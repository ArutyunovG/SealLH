from __future__ import annotations
import os
import shutil
from typing import Optional, Sequence, Dict
import logging

from clearml import Dataset
from omegaconf import DictConfig

logger = logging.getLogger("seallh._clearml.dataset")

__all__ = ["ClearMLDataset"]


class ClearMLDataset:
    """
    ClearML Dataset manager that handles multiple datasets based on configuration.
    
    Loads all configured datasets and maintains a dictionary mapping dataset names to local paths.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize ClearMLDataset with configuration.
        
        Args:
            config: Configuration containing dataset specifications and local_copy_dir
        """
        self.config = config
        self.local_copy_dir = config.get("local_copy_dir")
        self.dataset_info = {}
        
        logger.info("Initializing ClearMLDataset manager")
        self._load_datasets()
    

    def _load_datasets(self) -> None:
        """Load all datasets specified in the configuration."""
        for dataset_name, dataset_spec in self.config.items():
            self._load_single_dataset(dataset_name, dataset_spec)


    def _load_single_dataset(self, dataset_name: str, dataset_spec: DictConfig) -> None:
        """
        Load a single dataset from its specification.
        
        Args:
            dataset_name: Name to use for the dataset (key in dataset_paths dict)
            dataset_spec: Dataset specification containing id/name/project/tags
        """
        try:
            logger.info("Loading dataset: %s", dataset_name)
            
            # Extract parameters from dataset specification
            dataset_id = dataset_spec.get("dataset_id")
            name = dataset_spec.get("name") 
            project = dataset_spec.get("project")
            tags = dataset_spec.get("tags", [])
            
            # Use dataset-specific local_copy_dir if provided, otherwise use global one
            local_copy_dir = dataset_spec.get("local_copy_dir") or "dataset"

            # Load the dataset using the existing get_local_dataset function
            local_path = self.get_local_dataset(
                local_copy_dir=local_copy_dir,
                dataset_id=dataset_id,
                name=name,
                project=project,
                tags=tags
            )
            
            self.dataset_info[dataset_name] = {
                "path": local_path,
                "class": dataset_spec.get("class")
            }

        except Exception as e:
            logger.error("Failed to load dataset '%s': %s", dataset_name, e)
            # Continue loading other datasets even if one fails


    def get_local_dataset(
        self,
        local_copy_dir: Optional[str] = None,
        dataset_id: Optional[str] = None,
        name: Optional[str] = None,
        project: Optional[str] = None,
        tags: Optional[Sequence[str]] = None
    ) -> str:
        """
            Download (if needed) and return a local readonly copy path of a ClearML Dataset.

            Args:
                local_copy_dir: Optional directory path where the dataset copy will be cached.
                            If None, uses ClearML's default cache directory.
                dataset_id: Optional ClearML dataset ID to retrieve. Takes precedence over name/project/tags.
                name: Optional dataset name. Required if dataset_id is not provided.
                project: Optional project name where the dataset resides. Used with name parameter.
                tags: Optional sequence of tags to filter the dataset. Used with name parameter.

            Usage:
                path = get_local_dataset(dataset_id="d123...")
                # or
                path = get_local_dataset(name="mnist", project="open")
                # or
                path = get_local_dataset(name="mnist", project="open", tags=["v1", "validated"])
                # or
                path = get_local_dataset(name="mnist", local_copy_dir="/custom/cache/dir")

            Returns:
                local_copy_dir path as string.
            """
        if not dataset_id and not name:
            raise ValueError("Provide either dataset_id or (name[, project, tags]).")

        if dataset_id:
            ds = Dataset.get(dataset_id=dataset_id)
            label = f"id={dataset_id}"
        else:
            ds = Dataset.get(dataset_name=name, dataset_project=project, dataset_tags=list(tags or []))
            label = f"name={name}, project={project}, tags={list(tags or [])}"

        logger.info(f"Resolving ClearML Dataset ({label})")

        if os.path.exists(local_copy_dir) and os.path.isdir(local_copy_dir):
            if ds.verify_dataset_hash(local_copy_dir):
                logger.warning(f"Local copy at '{local_copy_dir}' does not match remote. Deleting it and re-downloading.")
                shutil.rmtree(local_copy_dir)
                ds.get_mutable_local_copy(target_folder=local_copy_dir, overwrite=True)
            else:
                logger.info(f"Using up-to-date local copy of {name} at '{local_copy_dir}'")
        else:
            if os.path.exists(local_copy_dir):
                raise ValueError(f"Provided local_copy_dir '{local_copy_dir}' exists but is not a directory.")
            ds.get_mutable_local_copy(target_folder=local_copy_dir, overwrite=True)
       
        logger.info(f"Dataset {name} loaded successfully to: {local_copy_dir}")
        return local_copy_dir


    def get_dataset_names(self) -> Sequence[str]:
        """Return a list of all loaded dataset names."""
        return list(self.dataset_info.keys())


    def get_dataset_info(self, dataset_name: str) -> Dict[str, Optional[str]]:
        """
        Get dataset information by name.
        
        Args:
            dataset_name: Name of the dataset to retrieve info for.
        
        Returns:
            Dictionary with 'path' and 'class' keys, or raises KeyError if not found.
        """
        if dataset_name not in self.dataset_info:
            raise KeyError(f"Dataset '{dataset_name}' not found. Available: {list(self.dataset_info.keys())}")
        return self.dataset_info[dataset_name]
    
