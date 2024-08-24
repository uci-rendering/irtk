# run from ./inv-render-toolkit with `python -m examples.2_pytorch3d_demo.code`

import irtk
from irtk.scene import *
from irtk.renderer import Renderer
from irtk.io import write_image, to_torch_f

scene = Scene()
# scene.set('cow', Mesh.from_file('./data/cow_mesh/cow.obj', mat_id='cow_tex'))
# scene.set('cow_tex', DiffuseBRDF.from_file('./data/cow_mesh/cow_texture.png'))
# scene.set('sensor1', PerspectiveCamera.from_lookat(fov=60, origin=(0, 0, -2.7), target=(0, 0, 0), up=(0, 1, 0)))

scene.set('armadillo', Mesh.from_file('./examples/data/meshes/armadillo.obj', mat_id='blue'))
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
# scene.set('blue', MicrofacetBRDF((0.2, 0.2, 0.9), (0, 0, 0), (0.08)))
scene.set('sensor2', PerspectiveCamera.from_lookat(fov=40, origin=(-1.5, 1.5, 1.5), target=(0, 0, 0), up=(0, 1, 0)))

scene.set('film', HDRFilm(width=512, height=512))

render = Renderer('nvdiffrast', render_options={
    'npass': 1,
    'light_power': 2.0
})

# render forward
print('Rendering scene...')
image = render(scene)
# write_image('./output/cow_nvdiffrast.png', image[..., :3])
write_image('./output/0_introduction/armadillo_nvdiffrast.png', image[..., :3])