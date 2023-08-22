# run from ./inv-render-toolkit with `python -m examples.4_mesh_optimization.code <backend>`

import irt
from irt.scene import *
from irt.renderer import Renderer
from irt.io import write_image, to_srgb, write_mesh
from irt.loss import l1_loss
from irt.sampling import sample_sphere
from irt.utils import Timer
from largesteps_optimizer import LargeStepsOptimizer

import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import sys
import os
import numpy as np

if len(sys.argv) >= 2:
    renderer = sys.argv[1]
elif len(irt.get_connector_list()) == 1:
    renderer = irt.get_connector_list()[0]
else:
    print("Please specify backend renderer. Currently available backend(s):")
    print(irt.get_connector_list())
    exit()

mesh_target = 'pig'
mesh_init = 'pig_smooth'
output_folder = 'pig_smooth'
num_ref_sensors = 50
sensor_radius = 3
num_epoch = 15
max_gif_duration = 5000 # ms

# define scene
scene = Scene()
scene.set('object', Mesh.from_file(f'./examples/data/meshes/{mesh_target}.obj', mat_id='blue', to_world=torch.diag(torch.Tensor([0.015, 0.015, 0.015, 1]))))
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
# scene.set('envlight', EnvironmentLight.from_file('./examples/data/envmaps/factory.exr'))
# scene.set('sensor_main', PerspectiveCamera.from_lookat(fov=40, origin=(-1.5, 1.5, 1.5), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('sensor_main', PerspectiveCamera.from_lookat(fov=40, origin=(2, 2, -2), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('film', HDRFilm(width=512, height=512))

for i, origin in enumerate(sample_sphere(num_ref_sensors, sensor_radius, 'fibonacci')):
    scene.set(f'sensor_{i}', PerspectiveCamera.from_lookat(fov=40, origin=origin, target=(0, 0, 0), up=(0, 1, 0)))

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

if output_folder == '':
    if not os.path.exists(f'output/4_mesh_optimization'):
        os.makedirs(f'output/4_mesh_optimization')
    file_prefix = f'output/4_mesh_optimization/{renderer}'
else:
    if not os.path.exists(f'output/4_mesh_optimization/{output_folder}'):
        os.makedirs(f'output/4_mesh_optimization/{output_folder}', exist_ok=True)
    file_prefix = f'output/4_mesh_optimization/{output_folder}/{renderer}'

# render ref
print('Rendering ref...')
images_ref = render(scene, sensor_ids=[ i for i in range(num_ref_sensors + 1) ])
write_image(file_prefix + '_ref.png', images_ref[0])

# ref_dir = f'output/4_mesh_optimization/{mesh_target}/ref'
# if not os.path.exists(ref_dir):
#     os.makedirs(ref_dir)
# for i, image in enumerate(images_ref):
#     write_image(ref_dir + f'/ref_{i}.png', image)

# render init
print('Rendering init...')
scene.clear_cache()
scene.components.pop('object')
scene.set('object', Mesh.from_file(f'./examples/data/meshes/{mesh_init}.obj', mat_id='blue'))

# v = scene['object']['f'][300].long()
# offset = torch.zeros_like(scene['object']['v'][v])
# offset[..., 0] = 0.05 *  torch.ones_like(offset[..., 0])
# scene['object']['v'][v] = scene['object']['v'][v] + offset
# scene['object']['v'] = scene['object']['v'] * 0.95
scene['object']['v'] = scene['object']['v'] * 1.5
verts = scene['object']['v'].clone()
verts.requires_grad_()
image_init = render(scene)[0]
write_image(file_prefix + '_init.png', image_init)

# optimization
# optimizer = torch.optim.Adam([verts], lr=0.0005)
# optimizer = LargeStepsOptimizer(verts, scene['object']['f'], lr=0.003, lmbda=1)
optimizer = LargeStepsOptimizer(verts, scene['object']['f'], lr=0.0005, lmbda=1)

losses = []
images_gif = []
duration_this_frame = 0
max_loss = 0
with Timer('Optimization') as timer:
    for i in range(num_epoch):

        ref_sensors = (np.random.permutation(num_ref_sensors) + 1).tolist()
        # ref_sensors = (np.arange(num_ref_sensors) + 1).tolist()
        
        for sensor_id in ref_sensors:
            optimizer.zero_grad()
            scene['object']['v'] = verts
            scene.configure()
            
            # NOTE: bug - only the first sensor has grad
            images_opt = render(scene, sensor_ids=[sensor_id, 0])
            
            loss = l1_loss(images_ref[sensor_id], images_opt[0])
            loss.backward()
            
            # visualize
            loss = l1_loss(images_ref[0], images_opt[1]).detach().cpu().item()
            losses.append(loss)
            if loss > max_loss:
                max_loss = loss
            duration_per_img = max_gif_duration / (num_epoch * num_ref_sensors)
            duration_this_frame += duration_per_img
            if duration_this_frame >= 20:
                image_gif = img_as_ubyte(to_srgb(images_opt[1]))
                images_gif.append(image_gif)
                duration_this_frame = 0
            
            print(f'Epoch {i+1}/{num_epoch}, cam {sensor_id}, loss: {loss:.4f}')

            optimizer.step()
        

# images_opt = render(scene, sensor_ids=[ i for i in range(num_ref_sensors + 1) ])
# # write_image(file_prefix + '_ref.png', images_ref[0])
# ref_dir = f'output/4_mesh_optimization/{mesh_target}/ref'
# if not os.path.exists(ref_dir):
#     os.makedirs(ref_dir)
# for i, image in enumerate(images_opt):
#     write_image(ref_dir + f'/ref_{i}.png', image)

imageio.mimsave(file_prefix + '_opt.gif', images_gif, duration=20, loop=0)
plt.title(f'Final loss: {losses[-1]:.4g}')
plt.xlabel('iter')
plt.ylabel('loss')
plt.ylim(bottom=0, top=max_loss * 1.05)
plt.plot(range(len(losses)), losses, label='loss')
plt.savefig(file_prefix + '_loss.png')
# write_mesh(file_prefix + '_opt.obj', scene['object']['v'], scene['object']['f'])
image_opt = render(scene)[0]
write_image(file_prefix + '_opt.png', image_opt)

scene.clear_cache()