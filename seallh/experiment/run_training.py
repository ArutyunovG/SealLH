from omegaconf import DictConfig, OmegaConf
import logging
import pytorch_lightning as pl

from seallh.experiment.setup_dataloaders import setup_dataloaders
from seallh.experiment.utils import import_class
from seallh._pytorch_lightning.lit_model import LitModel


def run_training(cfg: DictConfig, datasets_dict, clearml_task, pl_loggers=None):
    """Run the training phase."""
    logger = logging.getLogger("cpplhy.experiment")
    
    # Create model
    logger.info("Creating Lightning model...")
    model = LitModel(cfg)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = setup_dataloaders(datasets_dict, cfg.dataloader, splits=["train", "val"])
    
    # Get primary dataset name (usually the first one)
    primary_dataset = list(datasets_dict.keys())[0]
    primary_dataloaders = dataloaders.get(primary_dataset, {})
    
    train_loader = primary_dataloaders.get("train")
    val_loader = primary_dataloaders.get("val", primary_dataloaders.get("validation"))
    
    # Validate that we have at least a training dataloader
    if train_loader is None:
        raise ValueError(f"No training dataloader found for dataset '{primary_dataset}'. Available splits: {list(primary_dataloaders.keys())}")
    
    logger.info("Dataloaders created:")
    logger.info(f"  Train: {'+' if train_loader else '-'}")
    logger.info(f"  Validation: {'+' if val_loader else '-'}")
    
    # Get seed from dataloader config
    seed = cfg.dataloader.get("seed", 42)
    pl.seed_everything(seed)

    # Create trainer
    logger.info("Creating PyTorch Lightning Trainer...")
    
    # PyTorch Lightning expects either None, single logger, or list of loggers
    pl_logger = None if not pl_loggers else (pl_loggers[0] if len(pl_loggers) == 1 else pl_loggers)
    
    # Create trainer config (convert to dict to avoid struct mode issues)
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    
    # Handle callbacks if present
    callbacks = []
    if "callbacks" in trainer_cfg:
        callback_configs = trainer_cfg.pop("callbacks")
        for callback_cfg in callback_configs:
            if "class" in callback_cfg:
                callback_cls = import_class(callback_cfg["class"])
                callback_instance = callback_cls(**callback_cfg.get("args", {}))
                callbacks.append(callback_instance)
                logger.info(f"Created callback: {callback_cfg['class']}")
    
    # should go only to Trainer.fit() method
    weights_only=trainer_cfg.pop("weights_only", True)
    trainer = pl.Trainer(logger=pl_logger, callbacks=callbacks if callbacks else None, **trainer_cfg)
    
    logger.info("Starting training...")
    
    # Train the model
    if val_loader is not None:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            weights_only=weights_only
        )
    else:
        logger.info("No validation dataloader found, training without validation")
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            weights_only=weights_only
        )
