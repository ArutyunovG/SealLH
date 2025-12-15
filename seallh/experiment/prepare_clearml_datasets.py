from omegaconf import DictConfig
import logging

from seallh.experiment.utils import import_class
from seallh._clearml.dataset import ClearMLDataset


def prepare_clearml_datasets(cfg: DictConfig):
    """
    Prepare and create datasets from ClearML dataset configurations.
    
    Args:
        cfg: Configuration containing dataset definitions
        
    Returns:
        dict: Dictionary of created datasets organized by dataset_name and split
              Format: {dataset_name: {split_name: dataset_instance, ...}, ...}
    """
    logger = logging.getLogger("seallh.experiment.prepare_datasets")
    
    # Initialize ClearML Dataset manager
    logger.info("Initializing datasets...")
    clearml_dataset = ClearMLDataset(config=cfg.dataset)

    # Create datasets from ClearML datasets
    created_datasets = {}
    for dataset_name, info in clearml_dataset.dataset_info.items():
        ds_path = info["path"]
        cls_path = info["class"]
        instances = cfg.dataset[dataset_name].get("instances", [])

        logger.info(
            "Creating dataset '%s': path=%s, class=%s",
            dataset_name, ds_path, cls_path
        )

        DatasetCls = import_class(cls_path)
        created_datasets[dataset_name] = {}

        for inst in instances:
            split_name = inst.get("split", "train")
            args = inst.get("args", {})
            
            # Create dataset instance
            ds_obj = DatasetCls(root_dir=ds_path, **args)
            created_datasets[dataset_name][split_name] = ds_obj
            
            logger.info(
                "Created split '%s' of dataset '%s' with %d samples",
                split_name, dataset_name, len(ds_obj)
            )
    
    logger.info("Dataset preparation completed")
    return created_datasets
