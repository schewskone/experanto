import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .datasets import ChunkDataset
from .utils import (
    FastSessionDataLoader,
    LongCycler,
    MultiEpochsDataLoader,
    SessionConcatDataset,
)

logger = logging.getLogger(__name__)


def get_multisession_dataloader(
    paths: List[str],
    configs: Union[DictConfig, Dict, List[Union[DictConfig, Dict]]] = None,
    shuffle_keys: bool = False,
    **kwargs,
) -> DataLoader:
    """
    Create a multisession dataloader from a list of paths and corresponding configs.
    
    This function creates individual dataloaders for each session and cycles through them,
    providing data from different sessions in a round-robin fashion.
    
    Args:
        paths (List[str]): List of paths to the dataset folders. Each path should point
            to a folder containing experimental data for a single session.
        configs (Union[DictConfig, Dict, List[Union[DictConfig, Dict]]], optional): 
            Configuration for each dataset. If a single config is provided, it will be 
            applied to all datasets. If a list is provided, it should match the length 
            of paths. Each config should contain 'dataset' and 'dataloader' keys.
            Defaults to None.
        shuffle_keys (bool, optional): Whether to shuffle the order of session keys
            when cycling through dataloaders. Defaults to False.
        **kwargs: Additional keyword arguments for dataset and dataloader configuration.
            If 'config' is provided in kwargs, it will be used as configs parameter.
    
    Returns:
        DataLoader: A LongCycler dataloader that cycles through all session dataloaders.
    
    Example:
        >>> paths = ['path/to/session1', 'path/to/session2']
        >>> config = {'dataset': {...}, 'dataloader': {'batch_size': 32}}
        >>> loader = get_multisession_dataloader(paths, configs=config)
        >>> for batch in loader:
        ...     # Process batch from alternating sessions
        ...     pass
    """

    if configs is None and "config" in kwargs:
        configs = kwargs.pop("config")

    # Convert single config to list for uniform handling
    if isinstance(configs, (DictConfig, dict)):
        configs = [configs] * len(paths)

    dataloaders = {}
    for i, (path, cfg) in enumerate(zip(paths, configs)):
        # TODO use saved meta dict to find data key
        if "dynamic" in path:
            dataset_name = path.split("dynamic")[1].split("-Video")[0]
        elif "_gaze" in path:
            dataset_name = path.split("_gaze")[0].split("datasets/")[1]
        else:
            dataset_name = f"session_{i}"
        dataset = ChunkDataset(path, **cfg.dataset)
        dataloaders[dataset_name] = MultiEpochsDataLoader(
            dataset,
            **cfg.dataloader,
        )

    return LongCycler(dataloaders)


def get_multisession_concat_dataloader(
    paths: List[str],
    configs: Union[Dict, List[Dict]] = None,
    seed: Optional[int] = 0,
    dataloader_config: Optional[Dict] = None,
    **kwargs,
) -> "FastSessionDataLoader":
    """
    Create a multi-session dataloader that concatenates all sessions into a single dataset.
    
    Unlike get_multisession_dataloader which cycles through sessions, this function
    concatenates all sessions into a single dataset and returns (session_key, batch) pairs.
    
    Args:
        paths (List[str]): List of paths to dataset folders. Each path should point
            to a folder containing experimental data for a single session.
        configs (Union[Dict, List[Dict]], optional): Configuration for datasets.
            If a single dict is provided, it will be applied to all datasets.
            If a list is provided, it should match the length of paths.
            Defaults to None.
        seed (int, optional): Random seed for reproducibility. If provided, each dataset
            will use a deterministic seed based on the path hash. Defaults to 0.
        dataloader_config (Dict, optional): Configuration dictionary for the dataloader
            (e.g., batch_size, num_workers). If None, uses config from first dataset.
            Defaults to None.
        **kwargs: Additional keyword arguments. If 'config' is provided, it will be
            used as configs parameter.
    
    Returns:
        FastSessionDataLoader: A dataloader that returns (session_key, batch) tuples,
            or None if no valid datasets are found.
    
    Example:
        >>> paths = ['path/to/session1', 'path/to/session2']
        >>> config = {'dataset': {...}}
        >>> dataloader_cfg = {'batch_size': 16, 'num_workers': 4}
        >>> loader = get_multisession_concat_dataloader(
        ...     paths, configs=config, dataloader_config=dataloader_cfg, seed=42
        ... )
        >>> for session_key, batch in loader:
        ...     print(f"Processing batch from {session_key}")
    """
    if configs is None and "config" in kwargs:
        configs = kwargs.pop("config")

    # Convert single config to list for uniform handling
    if not isinstance(configs, list):
        configs = [configs] * len(paths)

    # Create datasets
    datasets = []
    session_names = []

    start_time = time.time()
    for i, (path, cfg) in enumerate(zip(paths, configs)):

        # Create dataset with deterministic seed
        path_hash = hash(path) % 10000
        dataset_seed = seed + path_hash if seed is not None else None

        # Set specific seed for this dataset if needed
        if hasattr(cfg.get("dataset", {}), "seed") and dataset_seed is not None:
            cfg["dataset"]["seed"] = dataset_seed
        if "dataset" in cfg:
            cfg = cfg["dataset"]
        try:
            # Assuming ChunkDataset is defined elsewhere
            dataset = ChunkDataset(path, **cfg)
            session_name = dataset.data_key

            # Only add datasets with non-zero length
            if len(dataset) > 0:
                datasets.append(dataset)
                session_names.append(session_name)
        except Exception as e:
            warnings.warn(f"Error creating dataset for {path}: {str(e)}")

    if not datasets:
        return None

    # Create the concatenated dataset
    concat_dataset = SessionConcatDataset(datasets, session_names)

    # Get dataloader config from the first config
    if dataloader_config is None:
        dataloader_config = dict(configs[0].get("dataloader", {}))

    # Create the dataloader with our simplified implementation
    return FastSessionDataLoader(dataset=concat_dataset, seed=seed, **dataloader_config)
