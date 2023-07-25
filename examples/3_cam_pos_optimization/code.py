# run from ./inv-render-toolkit with `python -m examples.3_cam_pos_optimization.code <backend>`

import ivt
from ivt.scene import *
from ivt.renderer import Renderer
from ivt.io import write_image, to_srgb
from ivt.loss import l1_loss

import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import sys
import os

if len(sys.argv) < 2:
    print("Please specify backend renderer. Currently available backend(s):")
    print(ivt.get_connector_list())
    exit()
else:
    renderer = sys.argv[1]

scene = Scene()

scene.set('armadillo', Mesh.from_file('./examples/data/meshes/armadillo.obj', mat_id='blue'))
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
scene.set('sensor', PerspectiveCamera.from_lookat(fov=40, origin=(-1.5, 1.5, 1.5), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('film', HDRFilm(width=512, height=512))

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

if not os.path.exists('output/3_cam_pos_optimization'):
    os.mkdir('output/3_cam_pos_optimization')
file_prefix = f'output/3_cam_pos_optimization/armadillo_{renderer}'

# render ref
image = render(scene)[0]
write_image(file_prefix + '_ref.png', image[..., :3])

# render init
cam_pos = torch.Tensor([-1, 1, 0])
cam_pos.requires_grad_()
scene.set('sensor', PerspectiveCamera.from_lookat(fov=40, origin=cam_pos, target=(0, 0, 0), up=(0, 1, 0)))
image_init = render(scene)[0]
write_image(file_prefix + '_init.png', image_init[..., :3])

# for different imageio versions
if renderer == 'psdr_jit':
    writer = imageio.get_writer(file_prefix + '_opt.gif', mode='I', duration=30, loop=0)
elif renderer == 'pytorch3d':
    writer = imageio.get_writer(file_prefix + '_opt.gif', mode='I', duration=0.03)

# optimization
num_iter = 100
optimizer = torch.optim.Adam([cam_pos], lr=0.03)
losses = []
for i in range(num_iter):
    optimizer.zero_grad()
    scene.set('sensor', PerspectiveCamera.from_lookat(fov=40, origin=cam_pos, target=(0, 0, 0), up=(0, 1, 0)))
    scene.configure()
    
    image_opt = render(scene)[0]
    image_gif = img_as_ubyte(to_srgb(image_opt[..., :3]))
    writer.append_data(image_gif)
    
    loss = l1_loss(image, image_opt)
    loss.backward()
    losses.append(loss.detach().cpu().item())
    
    print(f'loss: {loss.detach().cpu():.4f}')
    
    optimizer.step()

writer.close()
plt.plot(range(len(losses)), losses, label='loss')
plt.savefig(file_prefix + '_loss.png')