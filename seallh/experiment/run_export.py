from omegaconf import DictConfig, OmegaConf
import logging
import torch
import os
from pathlib import Path
from typing import Dict, List, Tuple

from seallh.experiment.model_loading import get_model_loader

def run_export(cfg: DictConfig, datasets_dict, clearml_task, pl_loggers=None):
    """Run model export/conversion to various formats (ONNX, TorchScript, etc.)."""
    logger = logging.getLogger("seallh.experiment")
    
    logger.info("Starting export phase...")
    
    # Set up paths
    checkpoint_path = cfg.paths.checkpoint_path
    export_dir = cfg.paths.export_dir
    
    # Create export directory
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at: {checkpoint_path}")
        logger.info("Export phase skipped - no trained model checkpoint available")
        return
    
    # Load model using configurable model loader
    model_loader = get_model_loader(cfg)
    model = model_loader(checkpoint_path, cfg)
    
    # Determine device
    device_str = str(cfg.export.get("device", "cpu"))
    device = torch.device(device_str)
    model = model.to(device).eval()
    
    # Parse inputs/outputs from config
    input_names, input_shapes, input_dtypes = _parse_inputs(cfg.export)
    output_names = _parse_outputs(cfg.export)
    dynamic_axes = _parse_dynamic_axes(cfg.export)

    logger.info(f"Export inputs: {dict(zip(input_names, input_shapes))}")
    logger.info(f"Export outputs: {output_names}")
    
    # Create dummy inputs
    dummy_inputs = [torch.randn(s, dtype=dt, device=device) for s, dt in zip(input_shapes, input_dtypes)]
    dummy_input = tuple(dummy_inputs) if len(dummy_inputs) > 1 else dummy_inputs[0]
    
    # Export to ONNX format
    onnx_path = _export_onnx(model, dummy_input, export_dir, cfg, logger,
                             input_names=input_names, output_names=output_names,
                             dynamic_axes=dynamic_axes)
    
    # Upload to ClearML artifacts if export was successful
    if onnx_path and os.path.exists(onnx_path):
        clearml_task.upload_artifact(name=f"{cfg.project_name}_onnx_model", artifact_object=onnx_path)
        
        # Run visualization if available
        try:
            viz_path = _run_visualization(onnx_path, cfg, datasets_dict, logger)
        except Exception as e:
            raise RuntimeError(f"Visualization failed: {e}")
        
        if viz_path and os.path.exists(viz_path):
            _report_visualization(viz_path, clearml_task, cfg, pl_loggers, logger)
    else:
        raise RuntimeError("ONNX export failed, skipping ClearML upload")
    
    logger.info(f"Export completed! Files saved to: {export_dir}")


def _export_onnx(model, dummy_input, export_dir, cfg, logger,
                 input_names, output_names, dynamic_axes=None):
    """Export model to ONNX format."""
    try:
        import torch.onnx
        
        onnx_path = os.path.join(export_dir, f"{cfg.project_name}.onnx")
        logger.info(f"Exporting to ONNX: {onnx_path}")
        
        onnx_cfg = cfg.export.onnx
        
        export_kwargs = dict(
            export_params=onnx_cfg.get("export_params", True),
            opset_version=onnx_cfg.get("opset_version", 11),
            do_constant_folding=onnx_cfg.get("do_constant_folding", True),
            input_names=input_names,
            output_names=output_names,
        )
        if dynamic_axes:
            export_kwargs["dynamic_axes"] = dynamic_axes

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            **export_kwargs,
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


def _report_visualization(viz_path, clearml_task, cfg, pl_loggers, logger):
    """Report visualization image to ClearML."""
    from PIL import Image
    import numpy as np

    img = Image.open(viz_path)

    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = np.array(img)

    try:
        clearml_task.upload_artifact(
            name=f"{cfg.project_name}_visualization",
            artifact_object=viz_path,
        )
        clearml_task.report_image(
            title="ONNX Model Predictions",
            series="Exported Model Visualization",
            image=img_array,
        )
        logger.info("Visualization uploaded to ClearML")
    except Exception as e:
        logger.error(f"Failed to upload visualization to ClearML: {e}")


def _parse_inputs(export_cfg) -> Tuple[List[str], List[Tuple[int, ...]], List[torch.dtype]]:
    """Parse export.inputs config:
        export:
          inputs:
            - name: images
              shape: [1, 3, 640, 640]
              dtype: float32
    """

    _DTYPE_MAP = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "half": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "int64": torch.int64, "long": torch.int64,
        "int32": torch.int32, "int": torch.int32,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }

    inputs = export_cfg.inputs
    names, shapes, dtypes = [], [], []
    for item in inputs:
        names.append(str(item["name"]))
        shapes.append(tuple(item["shape"]))
        dtypes.append(_DTYPE_MAP.get(str(item.get("dtype", "float32")), torch.float32))
    return names, shapes, dtypes


def _parse_outputs(export_cfg) -> List[str]:
    """Parse export.outputs list."""
    return list(export_cfg.outputs)


def _parse_dynamic_axes(export_cfg) -> dict:
    """Parse dynamic_axes from config. Returns empty dict if not specified."""
    if "dynamic_axes" not in export_cfg:
        return {}
    dyn = export_cfg.dynamic_axes
    if isinstance(dyn, bool):
        return {} if not dyn else {}
    result = {}
    for name, axes in dyn.items():
        result[str(name)] = {int(k): str(v) for k, v in axes.items()}
    return result




