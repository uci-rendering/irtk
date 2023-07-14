# run from ./inv-render-toolkit with `python -m examples.2_pytorch3d_demo.code`

import ivt
from ivt.scene import *
from ivt.renderer import Renderer
from ivt.io import write_image

from pytorch3d.renderer import look_at_rotation

scene = Scene()
# scene.set('cow', Mesh.from_file('./examples/data/meshes/cow.obj', mat_id='cow_tex'))
# scene.set('cow_tex', DiffuseBRDF.from_file('./examples/data/textures/cow_texture.png'))
# scene.set('sensor1', PerspectiveCamera.from_lookat(fov=60, origin=(0, 0, -2.7), target=(0, 0, 0), up=(0, 1, 0)))

scene.set('armadillo', Mesh.from_file('./examples/data/meshes/armadillo.obj', mat_id='blue'))
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
scene.set('sensor2', PerspectiveCamera.from_lookat(fov=40, origin=(-1.5, 1.5, 1.5), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('film', HDRFilm(width=512, height=512))

render = Renderer('pytorch3d', render_options={
    'npass': 1
})

image = render(scene)[0]
write_image('output/armadillo_pytorch3d.png', image)