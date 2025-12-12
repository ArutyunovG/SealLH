from omegaconf import DictConfig, OmegaConf
import logging
import pytorch_lightning as pl
import os

from seallh.experiment.setup_dataloaders import setup_dataloaders
from seallh.experiment.utils import import_class
from seallh._pytorch_lightning.lit_model import LitModel


def run_testing(cfg: DictConfig, datasets_dict, clearml_task, pl_loggers=None):
    """Run model testing/evaluation using the best checkpoint."""
    logger = logging.getLogger("cpplhy.experiment")
    
    # Create model
    logger.info("Creating Lightning model for testing...")
    model = LitModel(cfg)
    
    # Create dataloaders for testing
    logger.info("Creating test dataloaders...")
    dataloaders = setup_dataloaders(datasets_dict, cfg.dataloader, splits=["test"])
    
    # Get primary dataset name (usually the first one)
    primary_dataset = list(datasets_dict.keys())[0]
    primary_dataloaders = dataloaders.get(primary_dataset, {})
    
    # Get test dataloader
    test_loader = primary_dataloaders.get("test")
    
    if test_loader is None:
        logger.warning("No test dataloader found. Available splits: %s", list(primary_dataloaders.keys()))
        logger.info("Testing phase skipped - no test data available")
        return
    
    logger.info(f"Using test split for testing with {len(test_loader.dataset)} samples")
    
    # Set up checkpoint path from config
    checkpoint_dir = cfg.paths.checkpoint_dir
    
    # Get checkpoint filename from ModelCheckpoint callback config
    checkpoint_filename = "best"  # default fallback
    for callback_cfg in cfg.trainer.callbacks:
        if callback_cfg.get("class") == "pytorch_lightning.callbacks.ModelCheckpoint":
            checkpoint_filename = callback_cfg.get("args", {}).get("filename", "best")
            break

    best_checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_filename}.ckpt")
    
    # Check if best checkpoint exists
    if not os.path.exists(best_checkpoint_path):
        logger.warning(f"Best checkpoint not found at: {best_checkpoint_path}")
        logger.info("Testing phase skipped - no trained model checkpoint available")
        return
    
    logger.info(f"Loading best checkpoint from: {best_checkpoint_path}")
    
    # Create trainer for testing
    logger.info("Creating PyTorch Lightning Trainer for testing...")
    
    # PyTorch Lightning expects either None, single logger, or list of loggers
    pl_logger = None if not pl_loggers else (pl_loggers[0] if len(pl_loggers) == 1 else pl_loggers)
    
    # Create minimal trainer config for testing (no training-specific settings)
    trainer_cfg = {
        "accelerator": cfg.trainer.get("accelerator", "auto"),
        "devices": cfg.trainer.get("devices", "auto"),
        "logger": pl_logger,
        "enable_checkpointing": False,  # No checkpointing during testing
        "enable_progress_bar": cfg.trainer.get("enable_progress_bar", True),
        "enable_model_summary": False,  # No need for model summary during testing
    }
    
    trainer = pl.Trainer(**trainer_cfg)
    
    logger.info("Starting testing...")
    
    # Load model from checkpoint and run testing
    test_results = trainer.test(
        model=model,
        dataloaders=test_loader,
        ckpt_path=best_checkpoint_path
    )
    
    # Log test results
    if test_results:
        logger.info("=== TEST RESULTS ===")
        for key, value in test_results[0].items():
            logger.info(f"{key}: {value}")
        logger.info("====================")
