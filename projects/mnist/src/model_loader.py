"""
MNIST-specific model loading for export.
This demonstrates how projects can customize model loading behavior.
"""
import logging
import torch
from seallh._pytorch_lightning.lit_model import LitModel


def load_model_for_export(checkpoint_path, cfg):
    """
    MNIST-specific model loading for export with softmax output.
    
    This custom loader adds a softmax layer at the end to convert logits to probabilities,
    which is often desired for ONNX model deployment.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        cfg: Configuration object
        
    Returns:
        torch.nn.Module: The loaded model with softmax output, ready for export
    """
    logger = logging.getLogger("cpplhy.experiment.mnist_model_loading")
    
    logger.info(f"Loading MNIST model from checkpoint: {checkpoint_path}")
    
    # Load the Lightning module from checkpoint
    lit_model = LitModel.load_from_checkpoint(checkpoint_path, cfg=cfg)
    
    # Extract the underlying model (.net) for export
    base_model = lit_model.net
    base_model.eval()  # Set to evaluation mode
    
    # Create a wrapper model that adds softmax at the end
    class ModelWithSoftmax(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.softmax = torch.nn.Softmax(dim=1)  # Apply softmax along class dimension
            
        def forward(self, x):
            logits = self.base_model(x)
            probabilities = self.softmax(logits)
            return probabilities
    
    # Wrap the model with softmax
    model = ModelWithSoftmax(base_model)
    model.eval()
    
    logger.info("MNIST model loaded successfully for export with softmax output")
    logger.info(f"Base model architecture: {base_model.__class__.__name__}")
    logger.info("Added softmax layer for probability output")
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model
