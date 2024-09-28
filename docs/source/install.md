# Installation Guide

To get started with `irtk`, follow these steps:

1. Install [`PyTorch`](https://pytorch.org/).

2. Install the intended differentiable renderer. We provide connectors for the following differentiable renderers:

  

   |                Differentiable Renderer                 |  CPU/GPU  | Render Method | Tested Version |
   | :----------------------------------------------------: | :-------: | :-----------: | :------------: |
   |   [psdr-jit](https://github.com/andyyankai/psdr-jit)   |    GPU    |  Ray Tracing  |     0.2.0      |
   |              psdr-enzyme (to be released)              |    CPU    |  Ray Tracing  |      TBD       |
   | [Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/) | CPU & GPU |  Ray Tracing  |     3.5.2      |
   |      [redner](https://github.com/BachiLi/redner)       | CPU & GPU |  Ray Tracing  |     latest     |
   |   [nvdiffrast](https://nvlabs.github.io/nvdiffrast/)   |    GPU    | Rasterization |     0.3.2      |
   |          [PyTorch3D](https://pytorch3d.org/)           |    GPU    | Rasterization |     0.7.8      |
    ```{attention}
   Last updated: 2024-09-27
   ```

3. Install `irtk`
   Once `PyTorch` and the differentiable renderer(s) are installed, you can install `irtk` using pip:

   ```bash
   pip install irtk
   ```

