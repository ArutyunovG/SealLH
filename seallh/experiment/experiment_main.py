from omegaconf import DictConfig, OmegaConf
import logging

from seallh.experiment.setup_logging import setup_logging
from seallh.experiment.run_training import run_training
from seallh.experiment.run_testing import run_testing
from seallh.experiment.run_export import run_export
from seallh.experiment.external_repository_setuper import setup_repositories
from seallh.experiment.prepare_clearml_datasets import prepare_clearml_datasets
from seallh._clearml.task import ClearMLTask


def experiment_main(cfg: DictConfig) -> None:
    """Experiment main, receives config from main.py"""
    logger = logging.getLogger("seallh.experiment")
    logger.info("Entered experiment_main")
    pl_loggers = setup_logging(cfg)

    logger.info('Running experiment with config:')
    logger.info('============================================================')
    try:
        # Try to resolve and print the config
        resolved_config = OmegaConf.to_yaml(cfg, resolve=True)
        logger.info(resolved_config)
    except Exception as e:
        # Print unresolved config first, then reraise the exception
        logger.info(OmegaConf.to_yaml(cfg, resolve=False))
        logger.warning(f"Could not fully resolve config for logging: {e}")
        raise
    logger.info('============================================================')
   
    # Initialize ClearML Task with the full config (clearml.yaml is merged into root)
    clearml_task = ClearMLTask(clearml_config=cfg.clearml)

    # Set up external repositories (clone and install dependencies)
    external_repositories = setup_repositories(cfg)
    if external_repositories:
        logger.info(f"External repositories ready: {list(external_repositories.keys())}")

    # Prepare datasets from ClearML dataset configurations
    created_datasets = prepare_clearml_datasets(cfg)

    # Check which phases are enabled and orchestrate them
    phases = cfg.get("phases", {"training": False, "testing": False, "export": False})
    
    logger.info("Experiment phases configuration:")
    logger.info(f"  Training: {'+' if phases['training'] else '-'}")
    logger.info(f"  Testing: { '+' if phases['testing']  else '-'}")
    logger.info(f"  Export: {  '+' if phases['export']   else '-'}")
    
    # Run training phase if enabled
    if phases.get("training", False):
        logger.info("=== TRAINING PHASE ===")
        run_training(cfg, created_datasets, clearml_task, pl_loggers)
        logger.info("Training phase completed!")
    else:
        logger.info("Training phase skipped")
    
    # Run testing phase if enabled
    if phases.get("testing", False):
        logger.info("=== TESTING PHASE ===")
        run_testing(cfg, created_datasets, clearml_task, pl_loggers)
        logger.info("Testing phase completed!")
    else:
        logger.info("Testing phase skipped")
    
    # Run export phase if enabled
    if phases.get("export", False):
        logger.info("=== EXPORT PHASE ===")
        run_export(cfg, created_datasets, clearml_task, pl_loggers)
        logger.info("Export phase completed!")
    else:
        logger.info("Export phase skipped")
    
    logger.info("All experiment phases completed!")
