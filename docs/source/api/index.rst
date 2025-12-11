API Reference
=============

This section contains the complete API reference for Experanto, organized by module.
Each module provides tools for different aspects of experimental data handling.

.. contents:: Quick Navigation
   :depth: 2
   :local:

Core Classes
============

Experiment
----------

The ``Experiment`` class is the main entry point for loading and managing experimental data.

.. autoclass:: experanto.experiment.Experiment
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Datasets
========

Datasets provide PyTorch-compatible interfaces for loading chunked experimental data.

ChunkDataset
------------

Advanced dataset with flexible per-modality configuration.

.. autoclass:: experanto.datasets.ChunkDataset
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

SimpleChunkedDataset
--------------------

Simplified dataset for basic chunking operations.

.. autoclass:: experanto.datasets.SimpleChunkedDataset
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Dataloaders
===========

Dataloaders extend PyTorch's DataLoader for multi-session experiments.

.. autofunction:: experanto.dataloaders.get_multisession_dataloader

.. autofunction:: experanto.dataloaders.get_multisession_concat_dataloader

Interpolators
=============

Interpolators provide unified access to data from different recording modalities.

Base Interpolator
-----------------

.. autoclass:: experanto.interpolators.Interpolator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

SequenceInterpolator
--------------------

For sequential time-series data (neural responses, behavioral signals).

.. autoclass:: experanto.interpolators.SequenceInterpolator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

ScreenInterpolator
------------------

For visual stimulus data (images, videos).

.. autoclass:: experanto.interpolators.ScreenInterpolator
   :members:
   :undoc-members:
   :show-inheritance:

TimeIntervalInterpolator
------------------------

For interval-based data (behavioral events, trial markers).

.. autoclass:: experanto.interpolators.TimeIntervalInterpolator
   :members:
   :undoc-members:
   :show-inheritance:

Time Intervals
==============

Time interval utilities for managing valid data ranges.

TimeInterval
------------

.. autoclass:: experanto.intervals.TimeInterval
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __contains__

Interval Functions
------------------

.. autofunction:: experanto.intervals.uniquefy_interval_array

.. autofunction:: experanto.intervals.find_intersection_between_two_interval_arrays

.. autofunction:: experanto.intervals.find_intersection_across_arrays_of_intervals

Filters
=======

Filters identify valid time intervals by excluding problematic data regions.

Common Filters
--------------

.. automodule:: experanto.filters.common_filters
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
=========

Helper functions and classes for data processing.

Data Processing
---------------

.. autofunction:: experanto.utils.replace_nan_with_batch_mean

.. autofunction:: experanto.utils.add_behavior_as_channels

DataLoader Utilities
--------------------

.. autoclass:: experanto.utils.MultiEpochsDataLoader
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: experanto.utils.LongCycler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: experanto.utils.SessionConcatDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: experanto.utils.FastSessionDataLoader
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
=============

Configuration management using Hydra/OmegaConf.

.. automodule:: experanto.configs
   :members:
   :undoc-members:
   :show-inheritance:

