"""
Configuration Management
========================

This module provides default configuration objects for Experanto using Hydra and OmegaConf.

The configuration system uses YAML files to define default settings for datasets, dataloaders,
and modality-specific parameters. These configurations can be easily overridden programmatically
or through Hydra's command-line interface.

Default Configurations:
    - **DEFAULT_CONFIG**: Complete configuration including dataset and dataloader settings
    - **DEFAULT_DATASET_CONFIG**: Dataset-specific configuration
    - **DEFAULT_MODALITY_CONFIG**: Per-modality configuration (screen, responses, eye_tracker, etc.)
    - **DEFAULT_DATALOADER_CONFIG**: DataLoader configuration (batch size, workers, etc.)

Example:
    >>> from experanto.configs import DEFAULT_MODALITY_CONFIG
    >>> # Use default config
    >>> dataset = ChunkDataset("path/to/data", modality_config=DEFAULT_MODALITY_CONFIG)
    >>> 
    >>> # Or customize it
    >>> custom_config = DEFAULT_MODALITY_CONFIG.copy()
    >>> custom_config['screen']['sampling_rate'] = 60.0
    >>> dataset = ChunkDataset("path/to/data", modality_config=custom_config)

See Also:
    The default configuration is loaded from ``configs/default.yaml`` in the repository root.
"""

from pathlib import Path

from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf, open_dict

# get config relative to this file
script_dir = Path(__file__).parent
config_path = script_dir / ".." / "configs" / "default.yaml"
config_path = config_path.resolve()
cfg = OmegaConf.load(config_path)

DEFAULT_CONFIG = cfg
DEFAULT_DATASET_CONFIG = cfg.dataset
DEFAULT_MODALITY_CONFIG = cfg.dataset.modality_config
DEFAULT_DATALOADER_CONFIG = cfg.dataloader
