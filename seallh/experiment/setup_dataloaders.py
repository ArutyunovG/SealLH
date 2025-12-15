import logging
from torch.utils.data import DataLoader


def setup_dataloaders(datasets_dict, dataloader_cfg, splits=None):
    """Create PyTorch DataLoaders from datasets for specified splits.
    
    Args:
        datasets_dict: Dictionary of datasets organized by dataset_name -> split_name -> dataset
        dataloader_cfg: Configuration for DataLoader parameters
        splits: List of dataset splits to create dataloaders for (e.g., ['train', 'val', 'test']).
                If None, creates dataloaders for all available splits.
        
    Returns:
        Dictionary of dataloaders organized by dataset_name -> split_name -> dataloader
    """
    logger = logging.getLogger("seallh.experiment.setup_dataloaders")
    dataloaders = {}
    
    # If no splits specified, use all available splits
    if splits is None:
        splits = set()
        for dataset_values in datasets_dict.values():
            splits.update(dataset_values.keys())
        splits = list(splits)
    
    # Normalize split names (handle 'validation' vs 'val')
    normalized_splits = set()
    for split in splits:
        normalized_splits.add(split)
        if split == "val":
            normalized_splits.add("validation")
        elif split == "validation":
            normalized_splits.add("val")
    
    for dataset_name, dataset_splits in datasets_dict.items():
        dataloaders[dataset_name] = {}
        
        for split_name, dataset in dataset_splits.items():
            # Only create dataloader if this split is in the requested splits
            if split_name not in normalized_splits:
                logger.debug(f"Skipping {split_name} dataloader for {dataset_name} (not in requested splits: {splits})")
                continue
                
            # Get appropriate dataloader config for this split
            if split_name == "train":
                dl_args = dataloader_cfg.get("train_args", {})
            elif split_name in ["val", "validation"]:
                dl_args = dataloader_cfg.get("val_args", {})
            elif split_name == "test":
                dl_args = dataloader_cfg.get("test_args", {})
            else:
                raise ValueError(f"Unsupported split name '{split_name}' for dataset '{dataset_name}'")
            
            # Create DataLoader
            dataloader = DataLoader(dataset, **dl_args)
            dataloaders[dataset_name][split_name] = dataloader
            
            # Log dataloader creation
            batch_size = dl_args.get("batch_size", "unknown")
            num_workers = dl_args.get("num_workers", "unknown")
            logger.info(f"Created {split_name} dataloader for {dataset_name}: "
                        f"batch_size={batch_size}, num_workers={num_workers}")
    
    return dataloaders
