# run from ./inv-render-toolkit with `python -m examples.0_introduction.code`

import irtk
from irtk.scene import *
from irtk.renderer import Renderer
from irtk.io import write_image

scene = Scene()
# scene.set('cow', Mesh.from_file('./examples/data/meshes/cow.obj', mat_id='cow_tex'))
# scene.set('cow_tex', DiffuseBRDF.from_file('./examples/data/textures/cow_texture.png'))
# scene.set('sensor1', PerspectiveCamera.from_lookat(fov=60, origin=(0, 0, -2.7), target=(0, 0, 0), up=(0, 1, 0)))
# scene.set('envlight', EnvironmentLight((1.0, 1.0, 1.0)))

scene.set('armadillo', Mesh.from_file('./examples/data/meshes/armadillo.obj', mat_id='blue'))
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
# scene.set('blue', MicrofacetBRDF((0.2, 0.2, 0.9), (0, 0, 0), (0.08)))
scene.set('envlight', EnvironmentLight.from_file('./examples/data/envmaps/factory.exr'))
# scene.set('sensor2', PerspectiveCamera.from_lookat(fov=40, origin=(-1.5, 1.5, 1.5), target=(0, 0, 0), up=(0, 1, 0), near=1, far=50))
scene.set('sensor2', PerspectiveCamera.from_lookat(fov=40, origin=(1.5, 0, 1.5), target=(0, 0, 0), up=(0, 1, 0), near=1, far=50))

scene.set('film', HDRFilm(width=512, height=512))
scene.set('integrator', Integrator(type='path', config={
    'max_depth': 1,
    'hide_emitters': False
}))

render = Renderer('redner', render_options={
    'spp': 128,
    'npass': 1
})
image = render(scene)[0]

# write_image('output/cow_redner.png', image)
write_image('output/armadillo_redner.png', image)
