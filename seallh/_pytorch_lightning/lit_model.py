from typing import Dict, Any, Optional, Callable, List, Tuple

import torch
import pytorch_lightning as pl
from seallh.experiment.utils import import_class


class LitModel(pl.LightningModule):
    """Lightning wrapper for generic hydra-config based experiments."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.net = self._create_model(cfg.model)
        self.losses = self._create_losses(cfg.train.get("criterion", {}))  # list[(name, fn, weight)]

        self._create_metrics(cfg.get("metrics", {}))


    # ----- Lightning required methods -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    def on_train_epoch_end(self):
        self._compute_and_log_metrics("train")

    def on_validation_epoch_end(self):
        self._compute_and_log_metrics("val")

    def on_test_epoch_end(self):
        self._compute_and_log_metrics("test")

    def configure_optimizers(self):
        optimizer = self._create_optimizer(self.cfg.train.optimizer, self.parameters())
        scheduler, meta = self._create_scheduler(self.cfg.train.scheduler, optimizer)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, **meta}}

    # ----- Core step -----

    def _step(self, batch, stage: str):

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")

        outputs = self(x)

        # Only calculate losses during training phase
        if stage == "train":
            total_loss = 0.0
            for name, fn, w in self.losses:
                l = fn(outputs, y)
                total_loss = total_loss + (w * l)
                self.log(f"{stage}/loss_{name}", l, prog_bar=(name == "main"))

            self.log(f"{stage}/loss", total_loss, prog_bar=True)
        else:
            # For validation and testing, we don't need loss computation
            total_loss = 0.0
        
        # Update metrics (accumulate, don't log yet)
        self._update_metrics(outputs, y, stage)
        
        return total_loss

    # ----- factories -----

    def _create_model(self, model_cfg: Dict[str, Any]):
        cls_path = model_cfg.get("class")
        if not cls_path:
            raise ValueError("cfg.model.class is required (e.g. 'projects.mnist.src.model.Model').")
        cls = import_class(cls_path)
        return cls(**model_cfg.get("args", {}))

    def _create_losses(self, crit_cfg: Any) -> List[Tuple[str, Callable, float]]:
        """
        Accepts:
          - single dict: {class: "...", args: {}, weight: 1.0, name: "main"}
          - list of dicts: [{class: "...", args: {}, weight: 0.5, name: ce}, ...]
        Returns list of (name, loss_fn, weight).
        """
        def build_one(cfg_item: Dict[str, Any], default_name: str) -> Tuple[str, Callable, float]:
            cls = import_class(cfg_item.get("class", "torch.nn.CrossEntropyLoss"))
            fn = cls(**cfg_item.get("args", {}))
            name = cfg_item.get("name", default_name)
            weight = float(cfg_item.get("weight", 1.0))
            return name, fn, weight

        # Handle OmegaConf types - check for list-like behavior first
        if hasattr(crit_cfg, '__iter__') and not hasattr(crit_cfg, 'keys'):
            # List-like (including ListConfig)
            out: List[Tuple[str, Callable, float]] = []
            for i, item in enumerate(crit_cfg):
                out.append(build_one(item, f"loss{i+1}"))
            return out
        elif hasattr(crit_cfg, 'keys') or isinstance(crit_cfg, dict):
            # Dict-like (including DictConfig)
            return [build_one(crit_cfg, "main")]
        else:
            # Fallback: single CrossEntropy
            cls = import_class("torch.nn.CrossEntropyLoss")
            return [("main", cls(), 1.0)]

    def _create_metrics(self, metrics_cfg: Dict[str, Any]) -> None:
        """Setup metrics for train, validation, and test stages."""
        from torchmetrics import MetricCollection
        
        # Handle both old format (flat) and new format (per-stage)
        for stage in ["train", "val", "test"]:
            if stage in metrics_cfg:
                stage_metrics = {}
                stage_cfg = metrics_cfg[stage]
                
                for name, spec in stage_cfg.items():
                    cls = import_class(spec["class"])
                    metric = cls(**spec.get("args", {}))
                    stage_metrics[name] = metric
                
                # Register MetricCollection as a module attribute
                if stage_metrics:
                    metric_collection = MetricCollection(stage_metrics)
                    setattr(self, f"{stage}_metrics", metric_collection)

    def _update_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, stage: str):
        """Update (accumulate) metrics for the given stage without logging."""
        metrics_attr = f"{stage}_metrics"
        
        if hasattr(self, metrics_attr):
            metrics_collection = getattr(self, metrics_attr)
            # Just update the metrics state, don't compute final values yet
            metrics_collection.update(outputs, targets)

    def _compute_and_log_metrics(self, stage: str):
        """Compute final metric values and log them at epoch end."""
        metrics_attr = f"{stage}_metrics"
        
        if hasattr(self, metrics_attr):
            metrics_collection = getattr(self, metrics_attr)
            
            # Compute final metric values for the epoch
            metric_values = metrics_collection.compute()
            
            # Log all computed metrics
            for metric_name, metric_value in metric_values.items():
                log_key = f"{stage}/{metric_name}"
                self.log(log_key, metric_value)
            
            # Reset metrics for next epoch
            metrics_collection.reset()

    def _create_optimizer(self, optimizer_cfg: Dict[str, Any], parameters):
        cls = import_class(optimizer_cfg.get("class", "torch.optim.AdamW"))
        return cls(parameters, **optimizer_cfg.get("args", {}))

    def _create_scheduler(self, scheduler_cfg: Dict[str, Any], optimizer):
        if not scheduler_cfg.get("enabled", False):
            return None, None
        cls_path = scheduler_cfg.get("class")
        if not cls_path:
            return None, None
        args = dict(scheduler_cfg.get("args", {}))
        if "OneCycleLR" in cls_path and args.get("total_steps") in (None, 0):
            args.pop("total_steps", None)
        cls = import_class(cls_path)
        sched = cls(optimizer, **args)
        interval = "step" if "OneCycleLR" in cls_path else "epoch"
        monitor = "train/loss" if "ReduceLROnPlateau" in cls_path else None
        meta = {"interval": interval, **({"monitor": monitor} if monitor else {})}
        return sched, meta
