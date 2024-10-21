# PyTorch3D

[PyTorch3D](https://pytorch3d.org/) is a differentiable rasterizer. 
It supports differentiation of rendered image w.r.t. arbitrary scene parameters in direct illuminated scenes.

In most cases, we set a point light to be always collocated to the camera, to ensure the consistency of the rendered image when compared to physics-based renderers. 

```{Attention}
By default in PyTorch3D, the radiation intensity of a light source does not decline with distance (i.e., the brightness of an object is always the same regardless of its position). Also, it does not natively support Microfacet BRDF. In order to ensure it has consistent behavior with other renderers, some modification to the source code is needed.
```

Supported scene components:

    HDRFilm, PerspectiveCamera, Mesh, DiffuseBRDF, MicrofacetBRDF, PointLight

## Integrator Configurations

PyTorch3D connector does not require to set an integrator explicitly. 

## Render Options

|   Option    | type  |                         Description                          |
| :---------: | :---: | :----------------------------------------------------------: |
|    npass    |  int  |                 Number of rendering passes.                  |
| light_power | float | Power of the collocated point light. If provided, the connector creates a point light collocated to the camera and sets its power. |

## Special Features