# Connector References

While we strive to unify different renderers, certain aspects remain unique to each and require specific handling:

1. Integrator Configurations: Each renderer has its own set of configurations for integrators, even for common types like path tracers.
   ```python
   # 'type' and 'config' are backend-specific
   integrator = irtk.scene.Integrator(type=<type>, config=<config>)
   ```

2. Renderer Options: Similarly, renderers have distinct sets of rendering options, such as samples per pixel (spp).
   ```python
   # 'connector_name' and 'render_options' are backend-specific
   renderer = irtk.renderer.Renderer(connector_name=<name>, render_options=<options>)
   ```

3. Renderer-Specific Features: We occasionally expose unique, useful features of a particular renderer through its corresponding connector. These features are not shared across other connectors.

For detailed information on specific connectors, please refer to the following sections:

```{toctree}
:maxdepth: 1

psdr-jit
psdr-enzyme
mitsuba
nvdiffrast
pytorch3d
redner
```