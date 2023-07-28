# run from ./inv-render-toolkit with `python -m examples.4_mesh_optimization.code <backend>`

import ivt
from ivt.scene import *
from ivt.renderer import Renderer
from ivt.io import write_image, to_srgb, write_mesh
from ivt.loss import l1_loss
from ivt.sampling import sample_sphere

import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import sys
import os
import numpy as np

# from pytorch3d.loss import mesh_normal_consistency, mesh_edge_loss, mesh_laplacian_smoothing
# from pytorch3d.structures import Meshes

if len(sys.argv) >= 2:
    renderer = sys.argv[1]
elif len(ivt.get_connector_list()) == 1:
    renderer = ivt.get_connector_list()[0]
else:
    print("Please specify backend renderer. Currently available backend(s):")
    print(ivt.get_connector_list())
    exit()

mesh_target = 'armadillo'

# define scene
scene = Scene()
scene.set('armadillo', Mesh.from_file('./examples/data/meshes/cow.obj', mat_id='blue'))
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
# scene.set('sensor_main', PerspectiveCamera.from_lookat(fov=40, origin=(-1.5, 1.5, 1.5), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('sensor_main', PerspectiveCamera.from_lookat(fov=40, origin=(2, 2, -2), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('film', HDRFilm(width=512, height=512))

num_ref_sensors = 50
radius = 3
for i, origin in enumerate(sample_sphere(num_ref_sensors, radius, 'fibonacci')):
    scene.set(f'sensor_{i}', PerspectiveCamera.from_lookat(fov=40, origin=origin, target=(0, 0, 0), up=(0, 1, 0)))

if renderer == 'psdr_jit':
    scene.set('integrator', Integrator(type='collocated', config={
        'intensity': 10
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
        'npass': 1
    })

if not os.path.exists(f'output/4_mesh_optimization/{mesh_target}'):
    os.makedirs(f'output/4_mesh_optimization/{mesh_target}', )
file_prefix = f'output/4_mesh_optimization/{mesh_target}/{renderer}'

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
scene.components.pop('armadillo')
scene.set('sphere', Mesh.from_file('./examples/data/meshes/cow.obj', mat_id='blue'))
# v = scene['sphere']['f'][300].long()
# offset = torch.zeros_like(scene['sphere']['v'][v])
# offset[..., 0] = 0.05 *  torch.ones_like(offset[..., 0])
# scene['sphere']['v'][v] = scene['sphere']['v'][v] + offset
scene['sphere']['v'] = scene['sphere']['v'] * 0.95
verts = scene['sphere']['v'].clone()
verts.requires_grad_()
scene.clear_cache()
image_init = render(scene)[0]
write_image(file_prefix + '_init.png', image_init)

# for different imageio versions
if renderer == 'psdr_jit':
    writer = imageio.get_writer(file_prefix + '_opt.gif', mode='I', duration=30, loop=0)
elif renderer == 'pytorch3d':
    writer = imageio.get_writer(file_prefix + '_opt.gif', mode='I', duration=0.03)

# optimization
num_epoch = 5
num_iter = 0
num_ref_per_iter = 2
optimizer = torch.optim.Adam([verts], lr=0.0005)
losses = []
for i in range(num_epoch):

    ref_sensors = (np.random.permutation(num_ref_sensors) + 1).tolist()
    
    for sensor_id in ref_sensors:
        optimizer.zero_grad()
        scene['sphere']['v'] = verts
        scene.configure()
        
        images_opt = render(scene, sensor_ids=[0, sensor_id])
        image_gif = img_as_ubyte(to_srgb(images_opt[0]))
        writer.append_data(image_gif)
        
        loss = l1_loss(images_ref[sensor_id], images_opt[1])
        # if renderer == 'pytorch3d':
        #     mesh = Meshes([verts], [scene['sphere']['f']])
        #     loss += 0.01 * mesh_normal_consistency(mesh)
        #     loss += mesh_edge_loss(mesh)
        #     loss += mesh_laplacian_smoothing(mesh, method="uniform")
        loss.backward()
        losses.append(loss.detach().cpu().item())
        
        print(f'Epoch {i+1}/{num_epoch}, cam {sensor_id}, loss: {loss.detach().cpu():.4f}')

        optimizer.step()
        

# images_opt = render(scene, sensor_ids=[ i for i in range(num_ref_sensors + 1) ])
# # write_image(file_prefix + '_ref.png', images_ref[0])
# ref_dir = f'output/4_mesh_optimization/{mesh_target}/ref'
# if not os.path.exists(ref_dir):
#     os.makedirs(ref_dir)
# for i, image in enumerate(images_opt):
#     write_image(ref_dir + f'/ref_{i}.png', image)

writer.close()
plt.plot(range(len(losses)), losses, label='loss')
plt.savefig(file_prefix + '_loss.png')
write_mesh(file_prefix + '_opt.obj', scene['sphere']['v'], scene['sphere']['f'])
image_opt = render(scene)[0]
write_image(file_prefix + '_opt.png', image_opt)