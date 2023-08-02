# run from ./inv-render-toolkit with `python -m examples.2_pytorch3d_demo.code`

import irt
from irt.scene import *
from irt.renderer import Renderer
from irt.io import write_image, to_torch_f
from irt.utils import apply_pmkmp_cm

import os

mesh_target = 'armadillo'
renderer = 'pytorch3d'

if not os.path.exists(f'output/2_pytorch3d_demo'):
        os.makedirs(f'output/2_pytorch3d_demo')
file_prefix = f'output/2_pytorch3d_demo/{mesh_target}_{renderer}'

scene = Scene()
# scene.set('cow', Mesh.from_file('./examples/data/meshes/cow.obj', mat_id='cow_tex'))
# scene.set('cow_tex', DiffuseBRDF.from_file('./examples/data/textures/cow_texture.png'))
# scene.set('sensor', PerspectiveCamera.from_lookat(fov=60, origin=(0, 0, -2.7), target=(0, 0, 0), up=(0, 1, 0)))

scene.set('object', Mesh.from_file(f'./examples/data/meshes/{mesh_target}.obj', mat_id='blue'))
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
scene.set('sensor', PerspectiveCamera.from_lookat(fov=40, origin=(-1.5, 1.5, 1.5), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('film', HDRFilm(width=512, height=512))

render = Renderer('pytorch3d', render_options={
    'npass': 1
})

# render forward
print('Rendering scene...')
image = render(scene)[0]
write_image(f'{file_prefix}.png', image[..., :3])

# render grad using finite difference
print('Rendering diff...')
diff_x_offset = 0.01
diff_to_world = torch.eye(4)
diff_to_world[0, 3] = diff_x_offset
scene['object']['to_world'] = to_torch_f(diff_to_world)
diff_image = render(scene)[0]
diff_image = (diff_image[..., :3] - image[..., :3]) / diff_x_offset
diff_image_transformed = (diff_image.sum(-1) * 0.05).sigmoid()
diff_image = to_torch_f(apply_pmkmp_cm(diff_image_transformed.cpu().numpy()))
write_image(f'{file_prefix}_diff.png', diff_image)

# render grad using differentiable rendering
# print('Rendering grad...')
# # NOTE: can't set a component twice if it's cached
# scene.set('film', HDRFilm(width=256, height=256))
# grad_image = render.connector.renderGrad(scene, render.render_options)[0]
# write_image(f'{file_prefix}_grad.png', grad_image)