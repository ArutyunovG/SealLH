from seallh.helpers.collate_fn import collate_fn

import torch
from torch.utils.data import ConcatDataset

import logging

logger = logging.getLogger("seallh.projects.ddq.setup_dataloader")

def setup_dataloader(cfg, created_datasets, split) -> torch.utils.data.DataLoader:

    logger.info(f"Concatenating {split} datasets")

    if split in ['val', 'valid']:
        split = 'validation'

    datasets_to_concat = []
    for dataset_name, dataset_dct in created_datasets.items():
        if split not in dataset_dct:
            logger.info(f"No {split} datasets found for dataset {dataset_name}")
            continue 
        datasets_to_concat.append(dataset_dct[split])
        
    if not datasets_to_concat:
        raise RuntimeError(f"No datasets found for split {split}")
    
    logger.info(f"Found {len(datasets_to_concat)} datasets for split {split} to concatenate")

    concat_dataset = ConcatDataset(datasets_to_concat)

    logger.info(f"Building {split} dataloader from config {cfg.dataloader}")

    if split == 'train':
        data_loader_cfg = cfg.dataloader.train_args
    elif split == 'validation':
        data_loader_cfg = cfg.dataloader.val_args
    elif split == 'test':
        data_loader_cfg = cfg.dataloader.test_args
    else:
        raise RuntimeError(f"Unsupported split {split}")

    batch_size = min(data_loader_cfg.batch_size, len(concat_dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset=concat_dataset,
        batch_size=batch_size,
        num_workers=data_loader_cfg.num_workers,
        pin_memory=data_loader_cfg.pin_memory,
        collate_fn=concat_dataset.collate_fn if hasattr(concat_dataset, "collate_fn") else collate_fn,
        shuffle=data_loader_cfg.shuffle,
        prefetch_factor=data_loader_cfg.prefetch_factor,
        persistent_workers=data_loader_cfg.persistent_workers
    )
    
    logger.info(f"Data loaders created.")

    return dataloader
