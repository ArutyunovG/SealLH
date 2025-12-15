import logging
import os

from omegaconf import DictConfig
from typing import Optional, List, Any
from seallh.experiment.utils import import_class


def setup_logging(cfg: DictConfig) -> Optional[List[Any]]:
    """Setup logging configuration based on the config.
    
    Returns:
        List of PyTorch Lightning loggers or None if no loggers configured.
    """
    logger = logging.getLogger('seallh.experiment.setup_logging')

    # Configure Python logging based on config, with INFO as default
    log_level_str = cfg.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Setup logging to both console and file
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    
    # Clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # File handler - use paths from config if available, otherwise fallback
    if "paths" in cfg and "log_dir" in cfg.paths:
        log_file = f"{cfg.paths.log_dir}/experiment.log"
    else:
        log_file = cfg.get("log_file", "./logs/experiment.log")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Add handlers to root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Also configure PyTorch Lightning's internal logging
    logging.getLogger("pytorch_lightning").setLevel(log_level)
    
    logger = logging.getLogger('seallh.experiment.setup_logging')  # Re-get logger after setup
    logger.info(f'Log level set to: {log_level_str}')
    logger.info(f'Logging to console and file: {log_file}')
    
    # Setup PyTorch Lightning loggers if configured
    pl_loggers = []
    if "pl_logger" in cfg:
        logger.info("Setting up PyTorch Lightning loggers...")
        logger_configs = cfg.pl_logger
        # Handle both single logger (dict) and multiple loggers (list)
        if not isinstance(logger_configs, (list, tuple)):
            logger_configs = [logger_configs]
        
        for logger_cfg in logger_configs:
            if "class" in logger_cfg:
                logger_cls = import_class(logger_cfg["class"])
                pl_logger_instance = logger_cls(**logger_cfg.get("args", {}))
                pl_loggers.append(pl_logger_instance)
                logger.info(f"Created PyTorch Lightning logger: {logger_cfg['class']}")
    
    # Return loggers for use by the trainer
    return pl_loggers if pl_loggers else None
