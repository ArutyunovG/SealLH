"""
MNIST ONNX Model Visualization
User-defined visualization function for exported MNIST model.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import onnxruntime as ort
from pathlib import Path
import logging


def visualize_exported_model(onnx_path, cfg, datasets_dict):
    """
    Visualize the exported MNIST ONNX model with sample predictions.
    
    Args:
        onnx_path: Path to the exported ONNX model
        cfg: Configuration object
        datasets_dict: Dictionary of datasets
        
    Returns:
        str: Path to the generated visualization image
    """
    logger = logging.getLogger("seallh.experiment.visualization")
    
    try:
        # Load the ONNX model
        logger.info(f"Loading ONNX model for visualization: {onnx_path}")
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Debug: Print expected input shape
        input_info = ort_session.get_inputs()[0]
        logger.info(f"ONNX model expects input: name='{input_info.name}', shape={input_info.shape}, type={input_info.type}")
        
        # Get input shape from export config
        export_input_shape = cfg.export.input_shape
        logger.info(f"Using input shape from config: {export_input_shape}")
        
        # Get test dataset
        primary_dataset = list(datasets_dict.keys())[0]
        test_dataset = datasets_dict[primary_dataset].get('test')
        
        if test_dataset is None:
            logger.warning("No test dataset available for visualization")
            return None
        
        # Sample a few test images
        num_samples = 8
        samples = []
        labels = []
        predictions = []
        
        # Get random samples from test dataset
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        for idx in indices:
            sample, label = test_dataset[idx]
            
            # Convert to numpy
            if isinstance(sample, torch.Tensor):
                input_data = sample.numpy()
            else:
                input_data = np.array(sample)
            
            logger.debug(f"Original sample shape: {input_data.shape}")
            
            # Reshape according to export config input_shape
            # Add batch dimension and reshape to match export config
            if len(export_input_shape) == 1:  # Flattened input like [784]
                # Flatten the image and add batch dimension
                flattened = input_data.flatten()
                input_data = flattened.reshape(1, *export_input_shape)  # [1, 784]
            else:  # Multi-dimensional input like [1, 28, 28] or [3, 224, 224]
                # Ensure correct shape and add batch dimension if needed
                if input_data.shape != tuple(export_input_shape):
                    # Try to reshape to match export config
                    input_data = input_data.reshape(*export_input_shape)
                input_data = input_data.reshape(1, *export_input_shape)  # Add batch dimension
            
            logger.debug(f"Processed sample shape: {input_data.shape}")
            
            # Run inference
            ort_inputs = {input_info.name: input_data.astype(np.float32)}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Get prediction - model now outputs probabilities instead of logits
            probabilities = ort_outputs[0][0]  # Remove batch dimension
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            # Save original sample for visualization (reshape to 2D for display)
            if len(export_input_shape) == 1:  # Flattened input
                # Reshape back to 2D for visualization (assuming square image)
                img_size = int(np.sqrt(export_input_shape[0]))
                if img_size * img_size == export_input_shape[0]:
                    display_sample = sample.reshape(img_size, img_size) if hasattr(sample, 'reshape') else np.array(sample).reshape(img_size, img_size)
                else:
                    # If not square, use original sample for display
                    display_sample = sample.squeeze() if hasattr(sample, 'squeeze') else np.array(sample).squeeze()
            else:
                # Use original sample for display
                display_sample = sample.squeeze() if hasattr(sample, 'squeeze') else np.array(sample).squeeze()
            
            samples.append(display_sample)
            labels.append(label)
            predictions.append((prediction, confidence))
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle('MNIST ONNX Model Predictions (with Confidence)', fontsize=16)
        
        for i, (sample, label, (pred, conf)) in enumerate(zip(samples, labels, predictions)):
            row = i // 4
            col = i % 4
            
            axes[row, col].imshow(sample, cmap='gray')
            axes[row, col].set_title(f'True: {label}, Pred: {pred} ({conf:.2f})')
            axes[row, col].axis('off')
            
            # Color the title based on correctness
            if label == pred:
                axes[row, col].title.set_color('green')
            else:
                axes[row, col].title.set_color('red')
        
        plt.tight_layout()
        
        # Save visualization
        viz_dir = Path(cfg.paths.export_dir)
        viz_path = viz_dir / f"{cfg.project_name}_onnx_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved: {viz_path}")
        
        # Calculate accuracy
        correct = sum(1 for l, (p, _) in zip(labels, predictions) if l == p)
        accuracy = correct / len(labels)
        logger.info(f"Sample accuracy: {accuracy:.2%} ({correct}/{len(labels)})")
        
        # Calculate average confidence
        avg_confidence = np.mean([conf for _, conf in predictions])
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        
        return str(viz_path)
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return None
