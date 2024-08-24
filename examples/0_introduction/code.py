import irtk
from irtk.scene import *
from irtk.renderer import Renderer
from irtk.io import write_image, to_torch_f
from irtk.utils import apply_pmkmp_cm

import os
import sys

mesh_target = 'armadillo'

if len(sys.argv) >= 2:
    renderer = sys.argv[1]
elif len(irtk.get_connector_list()) == 1:
    renderer = irtk.get_connector_list()[0]
else:
    print("Please specify backend renderer. Currently available backend(s):")
    print(irtk.get_connector_list())
    exit()

if not os.path.exists(f'output/0_introduction'):
        os.makedirs(f'output/0_introduction')
file_prefix = f'output/0_introduction/{mesh_target}_{renderer}'

scene = Scene()
# scene.set('cow', Mesh.from_file('./examples/data/meshes/cow.obj', mat_id='cow_tex'))
# scene.set('cow_tex', DiffuseBRDF.from_file('./examples/data/textures/cow_texture.png'))
# scene.set('sensor', PerspectiveCamera.from_lookat(fov=60, origin=(0, 0, -2.7), target=(0, 0, 0), up=(0, 1, 0)))

scene.set('object', Mesh.from_file(f'./examples/data/meshes/{mesh_target}.obj', mat_id='blue'))
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
scene.set('sensor', PerspectiveCamera.from_lookat(fov=40, origin=(-1.5, 1.5, 1.5), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('film', HDRFilm(width=512, height=512))

if renderer == 'psdr_jit':
    scene.set('integrator', Integrator(type='collocated', config={
        'intensity': 30
    }))

    render = Renderer('psdr_jit', render_options={
        'spp': 32,
        'sppe': 0,
        'sppse': 0,
        'log_level': 0,
        'npass': 1
    })

elif renderer == 'pytorch3d':
    render = Renderer(renderer, render_options={
        'npass': 1
    })
    
elif renderer == 'nvdiffrast':
    render = Renderer('nvdiffrast', render_options={
        'npass': 1,
        'light_power': 2.0
    })

elif renderer == 'mitsuba':
    scene.set('integrator', Integrator(type='path', config={
        'max_depth': 4,
        'hide_emitters': False
    }))
    
    render = Renderer('mitsuba', render_options={
        'spp': 128,
        'npass': 1
    })
elif renderer == 'redner':
    scene.set('integrator', Integrator(type='path', config={
        'max_depth': 1,
        'hide_emitters': False
    }))

    render = Renderer('redner', render_options={
        'spp': 128,
        'npass': 1
    })
    

# render forward
print('Rendering scene...')
image = render(scene)[0]
write_image(f'{file_prefix}.png', image[..., :3])

def img_transform(image):
    image_transformed = (image.sum(-1) * 0.1).sigmoid()
    image = to_torch_f(apply_pmkmp_cm(image_transformed.cpu().numpy()))
    return image

# render grad using finite difference
# print('Rendering diff...')
# diff_x_offset = 0.01
# diff_to_world = torch.eye(4)
# diff_to_world[0, 3] = diff_x_offset
# scene['object']['to_world'] = to_torch_f(diff_to_world)
# diff_image = render(scene)[0]
# diff_image = (diff_image[..., :3] - image[..., :3]) / diff_x_offset
# diff_image = img_transform(diff_image)
# write_image(f'{file_prefix}_diff.png', diff_image)

# # render grad using differentiable rendering
# print('Rendering grad...')
# # NOTE: can't set a component twice if it's cached
# scene.set('film', HDRFilm(width=256, height=256))
# grad_image = render.connector.renderGrad(scene, render.render_options)[0]
# grad_image = img_transform(grad_image)
# write_image(f'{file_prefix}_grad.png', grad_image)