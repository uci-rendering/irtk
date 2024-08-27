# Installation Guide

To get started with `irtk`, follow these steps:

1. Install [`PyTorch`](https://pytorch.org/).
   Choose a PyTorch version that is compatible with your intended differentiable renderer. Refer to the compatibility table below for guidance.

   ```{attention}
   The following compatibility table is based on our tested configurations. It assumes the latest version of each differentiable renderer.
   ```

   |                Differentiable Renderer                 | Compatible PyTorch Version |
   | :----------------------------------------------------: | :------------------------: |
   |   [psdr-jit](https://github.com/andyyankai/psdr-jit)   |            1.11            |
   |              psdr-enzyme (to be released)              |            TBD             |
   | [Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/) |          >= 1.11           |
   |   [nvdiffrast](https://nvlabs.github.io/nvdiffrast/)   |            TBD             |
   |          [PyTorch3D](https://pytorch3d.org/)           |            TBD             |
   |      [redner](https://github.com/BachiLi/redner)       |            TBD             |



2. Install the intended differentiable renderer. 

3. Install `irtk`
   Once PyTorch is installed, you can easily install `irtk` using pip:

   ```bash
   pip install irtk
   ```

This will set up your environment with both PyTorch and `irtk`, ready for your inverse rendering experiments.

