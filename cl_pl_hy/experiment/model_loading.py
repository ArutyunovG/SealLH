"""
Model loading utilities for export phase.
This module provides default model loading functionality that can be overridden by projects.
"""
import logging
from cl_pl_hy._pytorch_lightning.lit_model import LitModel


def load_model_for_export(checkpoint_path, cfg):
    """
    Load model from checkpoint for export.
    
    This is the default implementation that loads the underlying model (.net) from LitModel.
    Projects can override this function to customize model loading behavior.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        cfg: Configuration object
        
    Returns:
        torch.nn.Module: The loaded model ready for export
    """
    logger = logging.getLogger("cpplhy.experiment.model_loading")
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load the Lightning module from checkpoint
    lit_model = LitModel.load_from_checkpoint(checkpoint_path, cfg=cfg)
    
    # Extract the underlying model (.net) for export
    model = lit_model.net
    model.eval()  # Set to evaluation mode
    
    logger.info("Model loaded successfully for export")
    
    return model


def get_model_loader(cfg):
    """
    Get the appropriate model loader function for the current project.
    
    This function first tries to import a project-specific model loader,
    and falls back to the default implementation if none exists.
    
    Args:
        cfg: Configuration object
        
    Returns:
        callable: Model loading function
    """
    logger = logging.getLogger("cpplhy.experiment.model_loading")
    
    try:
        # Try to import project-specific model loader
        project_loader_module = f"projects.{cfg.project_name}.src.model_loader"
        logger.info(f"Attempting to import model loader from: {project_loader_module}")
        
        from cl_pl_hy.experiment.utils import import_class
        project_loader = import_class(f"{project_loader_module}.load_model_for_export")
        
        logger.info("Using project-specific model loader")
        return project_loader
        
    except ImportError as e:
        logger.info(f"No project-specific model loader found ({e}), using default")
        return load_model_for_export
    except Exception as e:
        logger.warning(f"Failed to load project-specific model loader ({e}), using default")
        return load_model_for_export
