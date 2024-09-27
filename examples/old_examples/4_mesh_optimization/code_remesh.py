# run from ./inv-render-toolkit with `python -m examples.4_mesh_optimization.code <backend>`

import irtk
from irtk.scene import *
from irtk.renderer import Renderer
from irtk.io import *
from irtk.loss import l1_loss
from irtk.sampling import sample_sphere
from irtk.utils import Timer
from irtk.metric import chamfer_distance
from irtk.connector import get_connector
from largesteps_optimizer import LargeStepsOptimizer

import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import sys
import os
import numpy as np
import time

if len(sys.argv) >= 2:
    renderer = sys.argv[1]
elif len(irtk.get_connector_list()) == 1:
    renderer = irtk.get_connector_list()[0]
else:
    print("Please specify backend renderer. Currently available backend(s):")
    print(irtk.get_connector_list())
    exit()

# configs
mesh_target = 'armadillo_highres'
mesh_init = 'sphere'
output_folder = 'armadillo_remesh'
num_ref_sensors = 100
num_epoch = 20
sensor_radius = 2

remesh_iter = 100
edge_length_scale = 0.8
init_lr = 1e-2
min_lr = 1e-4
lr_scale = 0.9
init_lmbda = 200
lmbda_scale = 1

num_errors = 50
error_interval = max(int(num_epoch * num_ref_sensors / num_errors), 1)
max_gif_duration = 5000 # ms

# define scene
scene = Scene()
scene.set('object', Mesh.from_file(f'./examples/data/meshes/{mesh_target}.obj', mat_id='blue', use_face_normal=False))
scene['object']['v'] = scene['object']['v'] * 0.007
scene['object']['v'] = scene['object']['v'] - torch.Tensor([0, 0.1, 0]).cuda()
scene['object']['uv'] = []
scene['object']['fuv'] = []
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
# scene.set('envlight', EnvironmentLight.from_file('./examples/data/envmaps/factory.exr'))
scene.set('sensor_main', PerspectiveCamera.from_lookat(fov=40, origin=(1.5, 1.5, -1.5), target=(0, 0, 0), up=(0, 1, 0)))
# scene.set('sensor_main', PerspectiveCamera.from_lookat(fov=40, origin=(2, 2, -2), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('film', HDRFilm(width=512, height=512))

for i, origin in enumerate(sample_sphere(num_ref_sensors, sensor_radius, 'fibonacci')):
    scene.set(f'sensor_{i}', PerspectiveCamera.from_lookat(fov=40, origin=origin, target=(0, 0, 0), up=(0, 1, 0)))

if renderer == 'psdr_jit':
    scene.set('integrator', Integrator(type='collocated', config={
        'intensity': 20
    }))

    render = Renderer('psdr_jit', render_options={
        'spp': 128,
        'sppe': 0,
        'sppse': 0,
        'log_level': 0,
        'npass': 1
    })
elif renderer == 'pytorch3d':
    render = Renderer(renderer, render_options={
        'npass': 1,
        'light_diffuse_color': (1.2, 1.2, 1.2)
    })
elif renderer == 'nvdiffrast':
        render = Renderer('nvdiffrast', render_options={
        'npass': 1,
        'light_power': 3.0
    })
elif renderer == 'mitsuba':
    scene.set('integrator', Integrator(type='path', config={
        'max_depth': 4,
        'hide_emitters': False
    }))
    
    render = Renderer('mitsuba', render_options={
        'spp': 128,
        'npass': 1,
        'point_light_intensity': 20
    })
elif renderer == 'redner':
    scene.set('integrator', Integrator(type='path', config={
        'max_depth': 1,
        'hide_emitters': False
    }))

    render = Renderer('redner', render_options={
        'spp': 64,
        'npass': 1,
        'light_intensity': [27.0, 27.0, 27.0]
    })

os.makedirs(f'output/4_mesh_optimization/{output_folder}/{renderer}/{renderer}_mesh', exist_ok=True)
file_prefix = f'output/4_mesh_optimization/{output_folder}/{renderer}/{renderer}'

# render ref
print('Rendering ref...')
images_ref = render(scene, sensor_ids=[ i for i in range(num_ref_sensors + 1) ])
write_image(file_prefix + '_ref.png', images_ref[0])

# render init
print('Rendering init...')
scene.clear_cache()
scene.components.pop('object')
scene.set('object', Mesh.from_file(f'./examples/data/meshes/{mesh_init}.obj', mat_id='blue', use_face_normal=False))
scene['object']['uv'] = []
scene['object']['fuv'] = []

# v = scene['object']['f'][300].long()
# offset = torch.zeros_like(scene['object']['v'][v])
# offset[..., 0] = 0.05 *  torch.ones_like(offset[..., 0])
# scene['object']['v'][v] = scene['object']['v'][v] + offset
# scene['object']['v'] = scene['object']['v'] * 0.95
# scene['object']['v'] = scene['object']['v'] * 1.5
scene['object']['v'].requires_grad_()
image_init = render(scene)[0]
write_image(file_prefix + '_init.png', image_init)

# optimization
print('Optimizing...')
timer_opt_start = time.time()
render.connector.render_time = 0
lmbda = init_lmbda
lr = init_lr
v = to_numpy(scene['object']['v'])
f = to_numpy(scene['object']['f'])
half_edge_length = gpytoolbox.halfedge_lengths(v, f).mean()

# optimizer = torch.optim.Adam([verts], lr=0.0005)
# optimizer = LargeStepsOptimizer(verts, scene['object']['f'], lr=0.003, lmbda=1)
# optimizer = LargeStepsOptimizer(scene['object']['v'], scene['object']['f'], lr=lr, lmbda=lmbda)

losses = []
images_gif = []
duration_this_frame = 0
iter = 0
max_loss = 0
min_loss_iter = 0
for i in range(num_epoch):

    ref_sensors = (np.random.permutation(num_ref_sensors) + 1).tolist()
    # ref_sensors = (np.arange(num_ref_sensors) + 1).tolist()
    
    for j, sensor_id in enumerate(ref_sensors):
        
        # remesh
        if iter % remesh_iter == 0:
            scene.clear_cache()
            
            v = to_numpy(scene['object']['v']).astype(np.float64)
            f = to_numpy(scene['object']['f']).astype(np.int32)
            curr_edge_length = gpytoolbox.halfedge_lengths(v, f).mean()
            desired_edge_length = min(curr_edge_length, half_edge_length * edge_length_scale)
            v, f = gpytoolbox.remesh_botsch(v, f, h=desired_edge_length)
            scene['object']['v'] = v
            scene['object']['f'] = f
            scene['object']['v'].requires_grad_()
            scene['object']['uv'] = []
            scene['object']['fuv'] = []
            
            lr = max(lr * lr_scale, min_lr)
            lmbda *= lmbda_scale
            optimizer = LargeStepsOptimizer(scene['object']['v'], scene['object']['f'], lr=lr, lmbda=lmbda)
        
        optimizer.zero_grad()
        scene['object'].mark_updated('v')
        scene.configure()
        
        # NOTE: psdr bug - only the first sensor has grad
        # image_opt, image_main = render(scene, sensor_ids=[sensor_id, 0])
        image_opt = render(scene, sensor_ids=[sensor_id])[0]
        image_main = render(scene)[0]
        
        loss = l1_loss(images_ref[sensor_id], image_opt)
        loss.backward()
        
        # visualize
        losses.append(loss.detach().cpu())
        if loss > max_loss:
            max_loss = loss.detach().cpu()
        # img_loss = l1_loss(images_ref[0], image_main).detach().cpu().item()
        if iter % error_interval == 0:
            write_mesh(file_prefix + f'_mesh/iter_{iter}.obj', scene['object']['v'], scene['object']['f'])
        
        duration_per_img = max_gif_duration / (num_epoch * num_ref_sensors)
        duration_this_frame += duration_per_img
        if duration_this_frame >= 20:
            image_gif = img_as_ubyte(to_srgb(image_main))
            images_gif.append(image_gif)
            duration_this_frame = 0
        
        print(f'Epoch {i+1}/{num_epoch}, cam {sensor_id}, img loss: {loss.detach().cpu():.4g}')

        optimizer.step()
        iter += 1

timer_opt_end = time.time()
render_time = render.connector.render_time
opt_time = timer_opt_end - timer_opt_start
print(f"Rendering elapsed time: {render_time:.4g} seconds.")
print(f"Optimization elapsed time: {opt_time:.4g} seconds.")

# images_opt = render(scene, sensor_ids=[ i for i in range(num_ref_sensors + 1) ])
# # write_image(file_prefix + '_ref.png', images_ref[0])
# ref_dir = f'output/4_mesh_optimization/{mesh_target}/ref'
# if not os.path.exists(ref_dir):
#     os.makedirs(ref_dir)
# for i, image in enumerate(images_opt):
#     write_image(ref_dir + f'/ref_{i}.png', image)

# mesh_opt = {
#     'v': scene['object']['v'].detach().cpu(),
#     'f': scene['object']['f'].detach().cpu()
# }
# final_loss = chamfer_distance(mesh_ref, mesh_opt, num_mesh_samples)
# losses.append(final_loss)
# if final_loss > max_loss:
#     max_loss = final_loss
# if final_loss < min_loss:
#     min_loss = final_loss
#     min_loss_iter = num_epoch * num_ref_sensors
# print(f"Final loss: {final_loss:.4g}")
# print(f"Best loss: {min_loss:.4g} at iter {min_loss_iter}")

plt.title('Image Loss')
plt.xlabel('iter')
plt.ylabel('loss')
plt.ylim(bottom=0, top=max_loss * 1.05)
plt.plot(range(iter), losses, label='loss')
plt.savefig(file_prefix + '_loss.png')

output_file = open(file_prefix + '.txt', 'w')
output_file.write(f"Rendering elapsed time: {render_time:.4g} seconds.\n")
output_file.write(f"Optimization elapsed time: {opt_time:.4g} seconds.\n")
# output_file.write(f"Final loss: {final_loss:.4g}\n")
# output_file.write(f"Best loss: {min_loss:.4g} at iter {min_loss_iter}\n")
output_file.close()

v = to_numpy(scene['object']['v']).astype(np.float64)
f = to_numpy(scene['object']['f']).astype(np.int32)
curr_edge_length = gpytoolbox.halfedge_lengths(v, f).mean()
desired_edge_length = min(curr_edge_length, half_edge_length * edge_length_scale)
v, f = gpytoolbox.remesh_botsch(v, f, h=desired_edge_length)
write_mesh(file_prefix + f'_mesh/iter_{iter}.obj', v, f)

imageio.mimsave(file_prefix + '_opt.gif', images_gif, duration=20, loop=0)

image_opt = render(scene)[0]
write_image(file_prefix + '_opt.png', image_opt)

scene.clear_cache()