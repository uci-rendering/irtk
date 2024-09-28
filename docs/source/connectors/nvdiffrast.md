# nvdiffrast

[Nvdiffrast](https://nvlabs.github.io/nvdiffrast/) is a differentiable rasterizer. 
It supports differentiation of rendered image w.r.t. arbitrary scene parameters in direct illuminated scenes. 

Nvdiffrast provides modular low-level operations for differentiable rasterizing, but does not come with full support for shading. 
Therefore, we have adopted the structure from [Nvdiffmodeling](https://github.com/NVlabs/nvdiffmodeling) in our connector to handle shading and rendering at a higher level. 
In most cases, we set a point light to be always collocated to the camera, to ensure the consistency of the rendered image when compared to physics-based renderers. 

Supported scene components:

    HDRFilm, PerspectiveCamera, Mesh, DiffuseBRDF, MicrofacetBRDF, PointLight

## Integrator Configurations

Nvdiffrast connector does not require to set an integrator explicitly. 

## Render Options

|   Option    | type  |                         Description                          |
| :---------: | :---: | :----------------------------------------------------------: |
|    npass    |  int  |                 Number of rendering passes.                  |
| light_power | float | Power of the collocated point light. If provided, the connector creates a point light collocated to the camera and sets its power. |
|    scale    | float | The scale of the size of the whole scene, render_size = real_size * scale. [^scale] |

[^scale]: In our testing, nvdiffrast works best when the size of the scene is in the range of [-1, 1]. If the scene is too large, the renderer might have unexpected behaviors. Therefore, we provide this parameter as a workaround to scale down the scene if needed. The gradient will be automatically scaled up to the original scene once obtained, so this parameter should have no impact on any user code outside of the connector.

## Special Features