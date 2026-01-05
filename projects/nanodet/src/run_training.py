from pathlib import Path
import logging
import importlib.resources
import yaml

logger = logging.getLogger("projects.nanodet.run_training")


# Note: we no longer search for a local nanodet repo. The package should
# provide `nanodet.config` and `nanodet.tools` as installed package data.


def _generate_nanodet_config(cfg_path: str, created_datasets, template_name: str = "nanodet-plus-m_320.yml") -> Path:
    """Generate a nanodet YAML config at `cfg_path` by loading the packaged
    template and replacing COCO train/val paths using `created_datasets`.

    Returns the Path to the written config.
    """
    # Determine primary dataset and splits
    try:
        primary_dataset = list(created_datasets.keys())[0]
        ds_splits = created_datasets[primary_dataset]
    except Exception:
        raise RuntimeError("No created_datasets available to generate nanodet config")

    train_ds = ds_splits.get("train") or ds_splits.get("training")
    if not train_ds:
        raise RuntimeError("No training dataset found in created_datasets")
    val_ds = ds_splits.get("val") or ds_splits.get("validation") or ds_splits.get("test")

    # Load template YAML
    try:
        tpl_file = importlib.resources.files("nanodet.config").joinpath(template_name)
        with importlib.resources.as_file(tpl_file) as tfp:
            tpl = yaml.safe_load(tfp.read_text())
    except Exception as e:
        raise RuntimeError("Failed to load nanodet config template") from e

    tpl.setdefault("data", {}).setdefault("train", {})["img_path"] = str(train_ds.img_path)
    tpl.setdefault("data", {}).setdefault("train", {})["ann_path"] = str(train_ds.ann_path)

    if val_ds:
        tpl.setdefault("data", {}).setdefault("val", {})["img_path"] = str(val_ds.img_path)
        tpl.setdefault("data", {}).setdefault("val", {})["ann_path"] = str(val_ds.ann_path)

    out_path = Path(cfg_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(tpl))
    logger.info("Wrote generated nanodet config to %s", out_path)
    return out_path


def run_training(cfg, created_datasets, clearml_task, pl_loggers=None):
    """Run NanoDet training using existing datasets.

    Prefers `cfg.nanodet_cfg` (path to write/use a nanodet YAML). If not
    provided the function will raise. The function generates a nanodet
    config from the packaged template, adapts Seallh COCO datasets to
    NanoDet format and runs training in-process via PyTorch Lightning.
    """

    cfg_path = cfg.get("nanodet_cfg", None)
    if not cfg_path:
        raise RuntimeError("No nanodet config provided and no default config found in nanodet repo")

    # Generate nanodet config file from template using created_datasets
    _generate_nanodet_config(cfg_path, created_datasets)

    # Instead of launching the external script, run training in-process
    # using the Seallh -> NanoDet adapter so we can reuse created_datasets.

    # Make sure the installed `nanodet` package parent directory is on
    # sys.path. Some installed layouts require the package parent to be
    # discoverable so internal relative imports (e.g. `from ..model...`)
    # succeed.
    try:
        import sys
        import importlib.util
        from pathlib import Path

        spec = importlib.util.find_spec("nanodet")
        if spec:
            pkg_path = None
            if getattr(spec, "submodule_search_locations", None):
                pkg_path = Path(spec.submodule_search_locations[0])
            elif getattr(spec, "origin", None):
                pkg_path = Path(spec.origin).resolve().parent

            if pkg_path:
                parent_dir = str(pkg_path.parent)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                    logger.info("Added installed nanodet parent to sys.path: %s", parent_dir)
    except Exception:
        # best-effort only
        pass

    try:
        import torch
        import pytorch_lightning as pl
        from nanodet.util import cfg as nd_cfg, load_config, mkdir, NanoDetLightningLogger, env_utils
        from nanodet.data.collate import naive_collate
        from nanodet.evaluator import build_evaluator
        from nanodet.trainer.task import TrainingTask
    except Exception as e:
        import traceback
        logger.error("Nanodet import failed: %s", e)
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to import nanodet training internals: {e}") from e

    # Load nanodet config into its global cfg object
    load_config(nd_cfg, cfg_path)

    # Build NanoDet-compatible datasets using the adapter
    try:
        # try relative import first
        from .seallh_coco_to_nanodet import SeallhCOCOToNanoDetDataset
    except Exception:
        from seallh_coco_to_nanodet import SeallhCOCOToNanoDetDataset

    primary = list(created_datasets.keys())[0]
    splits = created_datasets[primary]
    train_src = splits.get("train") or splits.get("training")
    val_src = splits.get("val") or splits.get("validation") or splits.get("test")

    def _rel_or_abs(base, path):
        try:
            return str(Path(path).relative_to(base))
        except Exception:
            return str(path)

    # Prefer to use already-created dataset objects if they implement the
    # PyTorch Dataset protocol. Otherwise wrap the Seallh COCO metadata with
    # the adapter so NanoDet gets the expected BaseDataset behavior.
    try:
        # try relative import first
        from .seallh_coco_to_nanodet import SeallhCOCOToNanoDetDataset
    except Exception:
        from seallh_coco_to_nanodet import SeallhCOCOToNanoDetDataset


    # Create dataloaders using nanodet conventions
    train_loader = torch.utils.data.DataLoader(
        train_src,
        batch_size=nd_cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=nd_cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )

    if val_src:
        val_loader = torch.utils.data.DataLoader(
            val_src,
            batch_size=nd_cfg.device.batchsize_per_gpu,
            shuffle=False,
            num_workers=nd_cfg.device.workers_per_gpu,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=False,
        )
    else:
        val_loader = None

    # Prepare evaluator and training task
    evaluator = build_evaluator(nd_cfg.evaluator, val_src) if val_src else None
    task = TrainingTask(nd_cfg, evaluator)

    # logger + trainer
    logger_nd = NanoDetLightningLogger(nd_cfg.save_dir)
    logger_nd.dump_cfg(nd_cfg)

    # Set random seed if provided
    if cfg.get("seed") is not None:
        pl.seed_everything(int(cfg.seed))

    if nd_cfg.device.gpu_ids == -1:
        accelerator, devices, strategy, precision = ("cpu", None, "auto", nd_cfg.device.precision)
    else:
        accelerator, devices, strategy, precision = ("gpu", nd_cfg.device.gpu_ids, "auto", nd_cfg.device.precision)
    if devices and len(devices) > 1:
        strategy = "ddp"
        env_utils.set_multi_processing(distributed=True)

    trainer = pl.Trainer(
        default_root_dir=nd_cfg.save_dir,
        max_epochs=nd_cfg.schedule.total_epochs,
        check_val_every_n_epoch=nd_cfg.schedule.val_intervals,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=nd_cfg.log.interval,
        num_sanity_val_steps=0,
        callbacks=[],
        logger=logger_nd,
        benchmark=nd_cfg.get("cudnn_benchmark", True),
        gradient_clip_val=nd_cfg.get("grad_clip", 0.0),
        strategy=strategy,
        precision=precision,
    )

    logger.info("Starting in-process NanoDet training")
    trainer.fit(task, train_loader, val_loader)
