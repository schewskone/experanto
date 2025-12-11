Use cases for Experanto
=======================

**Experanto** helps align temporal data from neuroscience experiments. It can handle visual stimuli, behavior, and neural responses.

Key Features
------------

It fixes usual issues such as:

- Different sampling rates for different devices
- Different formats for data types such as videos and images in the same experiments
- Temporal alignment of multi-modal data
- Efficient data loading for machine learning applications

Quick Start
-----------

After :doc:`installation`, you can get started with Experanto in three main ways:

1. **Load a single experiment** - Useful for testing and exploring data
   
   See :doc:`demo_experiment` for a detailed tutorial.

2. **Create a dataset** - For training machine learning models
   
   See :doc:`demo_dataset` for dataset creation and configuration.

3. **Load multiple sessions** - For multi-session experiments
   
   See :doc:`demo_multisession` for handling multiple experimental sessions.

Data Format
-----------

Experanto expects experiments to be organized in a specific directory structure with separate folders for each modality (screen, responses, behaviors, etc.). See :doc:`demo_data` for details on the expected data format.

Next Steps
----------

- Learn about :doc:`demo_configs` to customize your data loading pipeline
- Explore the :doc:`../api/index` for detailed API documentation
- Check out the example notebooks in the ``examples/`` directory of the repository

