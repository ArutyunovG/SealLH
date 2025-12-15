from omegaconf import DictConfig
import logging
import torch
import os
from pathlib import Path

from seallh.experiment.model_loading import get_model_loader


def run_export(cfg: DictConfig, datasets_dict, clearml_task, pl_loggers=None):
    """Run model export/conversion to various formats (ONNX, TorchScript, etc.)."""
    logger = logging.getLogger("seallh.experiment")
    
    logger.info("Starting export phase...")
    
    # Set up paths
    checkpoint_dir = cfg.paths.checkpoint_dir
    export_dir = cfg.paths.export_dir
    
    # Create export directory
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    
    # Get checkpoint filename from ModelCheckpoint callback config
    checkpoint_filename = "best"  # default fallback
    for callback_cfg in cfg.trainer.callbacks:
        if callback_cfg.get("class") == "pytorch_lightning.callbacks.ModelCheckpoint":
            checkpoint_filename = callback_cfg.get("args", {}).get("filename", "best")
            break
    
    best_checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_filename}.ckpt")
    
    # Check if checkpoint exists
    if not os.path.exists(best_checkpoint_path):
        logger.warning(f"Best checkpoint not found at: {best_checkpoint_path}")
        logger.info("Export phase skipped - no trained model checkpoint available")
        return
    
    # Load model using configurable model loader
    model_loader = get_model_loader(cfg)
    model = model_loader(best_checkpoint_path, cfg)
    
    # Determine device (use CPU for export to avoid device issues)
    device = torch.device("cpu")
    model = model.to(device)
    
    # Get input shape from config
    input_shape = cfg.export.input_shape
    logger.info(f"Using input shape for export: {input_shape}")
    
    # Create dummy input tensor (batch size = 1) on the same device as model
    dummy_input = torch.randn(1, *input_shape, device=device)
    
    # Export to ONNX format
    onnx_path = _export_onnx(model, dummy_input, export_dir, cfg, logger)
    
    # Upload to ClearML artifacts if export was successful
    if onnx_path and os.path.exists(onnx_path):
        _upload_to_clearml(onnx_path, clearml_task, cfg, logger)
        
        # Run visualization if available
        try:
            viz_path = _run_visualization(onnx_path, cfg, datasets_dict, logger)
        except Exception as e:
            raise RuntimeError(f"Visualization failed: {e}")
        
        if viz_path and os.path.exists(viz_path):
            _upload_visualization_to_clearml(viz_path, clearml_task, cfg, logger)
    else:
        raise RuntimeError("ONNX export failed, skipping ClearML upload")
    
    logger.info(f"Export completed! Files saved to: {export_dir}")


def _export_onnx(model, dummy_input, export_dir, cfg, logger):
    """Export model to ONNX format."""
    try:
        import torch.onnx
        
        onnx_path = os.path.join(export_dir, f"{cfg.project_name}.onnx")
        logger.info(f"Exporting to ONNX: {onnx_path}")
        
        # Get ONNX export settings
        onnx_cfg = cfg.export.onnx
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=onnx_cfg.get("export_params", True),
            opset_version=onnx_cfg.get("opset_version", 11),
            do_constant_folding=onnx_cfg.get("do_constant_folding", True),
            input_names=["input"],
            output_names=["output"]
        )
        
        logger.info(f"ONNX export successful: {onnx_path}")
        
        # Simplify ONNX model if requested
        if onnx_cfg.get("simplify", True):
            simplified_path = _simplify_onnx_model(onnx_path, logger)
            if simplified_path:
                onnx_path = simplified_path
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed")
        except ImportError:
            logger.warning("ONNX package not available for model validation")
        except Exception as e:
            logger.warning(f"ONNX model validation failed: {e}")
        
        return onnx_path
            
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise


def _simplify_onnx_model(onnx_path, logger):
    """Simplify ONNX model using onnxsim to optimize the graph."""
    try:
        import onnxsim
        import onnx
        
        logger.info(f"Simplifying ONNX model: {onnx_path}")
        
        # Load the original model
        model = onnx.load(onnx_path)
        
        # Get model info before simplification
        original_nodes = len(model.graph.node)
        
        # Simplify the model
        model_simplified, check = onnxsim.simplify(model)
        
        if check:
            # Get simplified model info
            simplified_nodes = len(model_simplified.graph.node)
            reduction = original_nodes - simplified_nodes
            
            # Create simplified model path
            simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
            
            # Save simplified model
            onnx.save(model_simplified, simplified_path)
            
            logger.info(f"ONNX simplification successful: {simplified_path}")
            logger.info(f"Nodes reduced: {original_nodes} → {simplified_nodes} (reduced by {reduction})")
            
            # Replace original with simplified version
            import os
            os.replace(simplified_path, onnx_path)
            logger.info("Replaced original ONNX model with simplified version")
            
            return onnx_path
        else:
            logger.warning("ONNX simplification check failed - keeping original model")
            return onnx_path
            
    except ImportError:
        logger.warning("onnxsim not available - skipping model simplification")
        return onnx_path
    except Exception as e:
        logger.error(f"ONNX simplification failed: {e}")
        logger.info("Keeping original non-simplified model")
        return onnx_path


def _upload_to_clearml(onnx_path, clearml_task, cfg, logger):
    """Upload exported ONNX model to ClearML artifacts."""
    try:
        if clearml_task and clearml_task.task:
            logger.info(f"Uploading ONNX model to ClearML artifacts: {onnx_path}")
            
            # Upload the ONNX file as an artifact
            artifact_name = f"{cfg.project_name}_model"
            clearml_task.task.upload_artifact(
                name=artifact_name,
                artifact_object=onnx_path,
                metadata={
                    "format": "onnx",
                    "input_shape": cfg.export.input_shape,
                    "opset_version": cfg.export.onnx.get("opset_version", 11),
                    "project": cfg.project_name
                }
            )
            
            logger.info(f"ONNX model uploaded to ClearML as artifact: {artifact_name}")
        else:
            logger.warning("ClearML task not available - skipping artifact upload")
            
    except Exception as e:
        logger.error(f"Failed to upload ONNX model to ClearML: {e}")


def _run_visualization(onnx_path, cfg, datasets_dict, logger):
    """Run project-specific visualization of the exported ONNX model."""
    try:
        # Try to import project-specific visualization function
        project_viz_module = f"projects.{cfg.project_name}.src.visualize_onnx"
        logger.info(f"Attempting to import visualization from: {project_viz_module}")
        
        from seallh.experiment.utils import import_class
        visualize_fn = import_class(f"{project_viz_module}.visualize_exported_model")
        
        logger.info("Running project-specific visualization...")
        viz_path = visualize_fn(onnx_path, cfg, datasets_dict)
        
        if viz_path:
            logger.info(f"Visualization completed: {viz_path}")
        else:
            logger.warning("Visualization function returned None")
            
        return viz_path
        
    except ImportError as e:
        logger.info(f"No project-specific visualization found ({e}), skipping visualization")
        return None
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise


def _upload_visualization_to_clearml(viz_path, clearml_task, cfg, logger):
    """Upload visualization image to ClearML artifacts."""
    try:
        if clearml_task and clearml_task.task:
            logger.info(f"Uploading visualization to ClearML: {viz_path}")
            
            # Upload as artifact
            artifact_name = f"{cfg.project_name}_visualization"
            clearml_task.task.upload_artifact(
                name=artifact_name,
                artifact_object=viz_path,
                metadata={
                    "type": "visualization",
                    "format": "png",
                    "description": "ONNX model prediction visualization",
                    "project": cfg.project_name
                }
            )
            
            # Also log as image for easy viewing in ClearML UI
            from PIL import Image
            import numpy as np
            
            img = Image.open(viz_path)
            
            # Convert RGBA to RGB if necessary to avoid format issues
            if img.mode == 'RGBA':
                # Create a white background and paste the image on it
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                img = background
            elif img.mode != 'RGB':
                # Convert any other mode to RGB
                img = img.convert('RGB')
            
            img_array = np.array(img)
            
            clearml_task.task.get_logger().report_image(
                title="ONNX Model Predictions",
                series="Exported Model Visualization",
                image=img_array,
                iteration=0
            )
            
            logger.info(f"Visualization uploaded to ClearML as artifact: {artifact_name}")
        else:
            logger.warning("ClearML task not available - skipping visualization upload")
            
    except Exception as e:
        logger.error(f"Failed to upload visualization to ClearML: {e}")
