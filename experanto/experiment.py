from __future__ import annotations

import logging
import re
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .configs import DEFAULT_MODALITY_CONFIG
from .interpolators import Interpolator

log = logging.getLogger(__name__)


class Experiment:
    """
    Loads and manages experimental data from multiple recording devices.

    The Experiment class handles loading data from different modalities (e.g., screen, eye tracker,
    treadmill, neural responses) and provides unified access to interpolated data across devices.

    Example:
        >>> from experanto import Experiment
        >>> exp = Experiment(root_folder="path/to/data", cache_data=True)
        >>> values, valid = exp.interpolate(times=[0.0, 0.1, 0.2], device="screen")
    """

    def __init__(
        self,
        root_folder: str,
        modality_config: dict = DEFAULT_MODALITY_CONFIG,
        cache_data: bool = False,
    ) -> None:
        """
        Initialize an Experiment with data from a root folder.

        Args:
            root_folder (str): Path to the data folder containing device subfolders.
                Each subfolder should correspond to a recording modality (e.g., 'screen',
                'eye_tracker', 'responses').
            modality_config (dict, optional): Configuration dictionary for each modality.
                Keys are device names (e.g., 'screen', 'eye_tracker') and values are
                configuration dictionaries containing 'interpolation' parameters.
                Defaults to DEFAULT_MODALITY_CONFIG.
            cache_data (bool, optional): If True, loads and keeps all trial data in memory
                for faster access. If False, uses memory-mapped files when available.
                Defaults to False.
        """
        self.root_folder = Path(root_folder)
        self.devices = dict()
        self.start_time = np.inf
        self.end_time = -np.inf
        self.modality_config = modality_config
        self.cache_data = cache_data
        self._load_devices()

    def _load_devices(self) -> None:
        # Populate devices by going through subfolders
        # Assumption: blocks are sorted by start time
        device_folders = [d for d in self.root_folder.iterdir() if (d.is_dir())]

        for d in device_folders:
            if d.name not in self.modality_config:
                log.info(f"Skipping {d.name} data... ")
                continue
            log.info(f"Parsing {d.name} data... ")
            dev = Interpolator.create(
                d,
                cache_data=self.cache_data,
                **self.modality_config[d.name]["interpolation"],
            )
            self.devices[d.name] = dev
            self.start_time = dev.start_time
            self.end_time = dev.end_time
            log.info("Parsing finished")

    @property
    def device_names(self):
        """
        Get the names of all available recording devices.

        Returns:
            tuple: Tuple of device names (e.g., ('screen', 'eye_tracker', 'responses')).
        """
        return tuple(self.devices.keys())

    def interpolate(self, times: slice, device=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate data at specified time points for one or all devices.

        Args:
            times (slice or np.ndarray): Time points at which to interpolate data.
                Can be a numpy array of timestamps or a slice object.
            device (str, optional): Name of the specific device to interpolate.
                If None, interpolates data for all available devices. Defaults to None.

        Returns:
            tuple: A tuple of (values, valid) where:
                - values: Interpolated data. If device is None, returns dict with device names
                  as keys. Otherwise, returns numpy array.
                - valid: Boolean mask indicating which time points have valid data.

        Example:
            >>> # Interpolate all devices
            >>> values, valid = exp.interpolate(times=np.linspace(0, 1, 100))
            >>> screen_data = values['screen']
            >>>
            >>> # Interpolate specific device
            >>> values, valid = exp.interpolate(times=np.linspace(0, 1, 100), device='screen')
        """
        if device is None:
            values = {}
            valid = {}
            for d, interp in self.devices.items():
                values[d], valid[d] = interp.interpolate(times)
        elif isinstance(device, str):
            assert device in self.devices, "Unknown device '{}'".format(device)
            values, valid = self.devices[device].interpolate(times)
        return values, valid

    def get_valid_range(self, device_name) -> tuple:
        """
        Get the valid time range for a specific device.

        Args:
            device_name (str): Name of the device (e.g., 'screen', 'eye_tracker').

        Returns:
            tuple: A tuple of (start_time, end_time) representing the valid time interval
                for the specified device in seconds.
        """
        return tuple(self.devices[device_name].valid_interval)
