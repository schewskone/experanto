**Experanto**
======================================

**Experanto** is a Python package designed for interpolating recordings and stimuli in neuroscience experiments. It enables users to load single or multiple experiments and create efficient dataloaders for machine learning applications.

Key Features
------------

* **Multi-Modal Data Handling**: Seamlessly manage data from multiple recording devices (screen stimuli, neural responses, eye tracking, behavioral sensors)
* **Flexible Interpolation**: Support for various interpolation modes (nearest-neighbor, linear) with optional normalization
* **PyTorch Integration**: Native PyTorch dataset and dataloader implementations for deep learning workflows
* **Time-Based Filtering**: Advanced interval management for excluding invalid or corrupted data regions
* **Memory Efficient**: Support for memory-mapped files and optional data caching
* **Multi-Session Support**: Built-in tools for handling experiments with multiple recording sessions

Quick Start
-----------

.. code-block:: python

    from experanto import Experiment
    
    # Load experimental data
    exp = Experiment(root_folder="path/to/data", cache_data=True)
    
    # Interpolate data at specific times
    import numpy as np
    times = np.linspace(0, 10, 1000)
    values, valid = exp.interpolate(times, device="screen")

For more examples, see the :doc:`concepts/getting_started` guide.

Issues & Support
----------------

Issues with the package can be submitted at our `GitHub Issues page <https://github.com/schewskone/experanto/issues>`_.

------------

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   concepts/installation
   concepts/getting_started

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
        
   concepts/demo_data
   concepts/demo_experiment
   concepts/demo_configs
   concepts/demo_dataset
   concepts/demo_multisession

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

