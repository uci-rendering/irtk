# run from ./inv-render-toolkit with `python -m examples.2_pytorch3d_demo.code`

import ivt
from ivt.scene import *
from ivt.renderer import Renderer
from ivt.io import write_image, to_torch_f

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

# render forward
print('Rendering scene...')
image = render(scene)[0]
write_image('output/armadillo_pytorch3d.png', image[..., :3])

# render diff
print('Rendering diff...')
diff_x_offset = 0.01
diff_to_world = torch.eye(4)
diff_to_world[0, 3] = diff_x_offset
scene['armadillo']['to_world'] = to_torch_f(diff_to_world)
diff_image = render(scene)[0]
write_image('output/armadillo_pytorch3d_diff.png', (diff_image[..., :3] - image[..., :3]) * 100)

# render grad
print('Rendering grad...')
# TODO: can't set a component twice if it's cached
scene.set('film', HDRFilm(width=64, height=64))
grad_image = render.connector.renderGrad(scene, render.render_options)[0]
write_image('output/armadillo_pytorch3d_grad.png', grad_image)