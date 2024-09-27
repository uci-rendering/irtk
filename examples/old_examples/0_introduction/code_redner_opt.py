import irtk
from irtk.scene import *
from irtk.renderer import Renderer
from irtk.io import write_image, to_srgb
from irtk.loss import l1_loss

import matplotlib.pyplot as plt
import imageio
from skimage import img_as_ubyte
import sys
import os

renderer = 'redner'

# define scene
scene = Scene()
scene.set('object', Mesh.from_file('./examples/data/meshes/armadillo.obj', mat_id='blue'))
# scene['object']['v'] = scene['object']['v'] * 0.5
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
# scene.set('sensor', PerspectiveCamera.from_lookat(fov=40, origin=(-1.5, 1.5, 1.5), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('sensor', PerspectiveCamera.from_lookat(fov=40, origin=(-1, 1, 0), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('film', HDRFilm(width=512, height=512))

scene.set('integrator', Integrator(type='path', config={
    'max_depth': 1,
    'hide_emitters': False
}))

render = Renderer('redner', render_options={
    'spp': 64,
    'npass': 1,
    'light_intensity': [27.0, 27.0, 27.0]
})

file_prefix = f'output/0_introduction/armadillo_opt_{renderer}'

# render ref
image = render(scene)[0]
write_image(file_prefix + '_ref.png', image)

# render init
x_offset = torch.Tensor([0.3]).cuda()
x_offset.requires_grad_()
offset = torch.zeros((3)).cuda()
offset[0] += x_offset[0]
origin_verts = scene['object']['v'].clone()
scene['object']['v'] = origin_verts + offset
image_init = render(scene)[0]
write_image(file_prefix + '_init.png', image_init)

# for different imageio versions
writer = imageio.get_writer(file_prefix + '_opt.gif', mode='I', duration=30, loop=0)

# optimization
num_iter = 20
optimizer = torch.optim.Adam([x_offset], lr=0.03)
losses = []
for i in range(num_iter):
    optimizer.zero_grad()
    offset = torch.zeros((3)).cuda()
    offset[0] += x_offset[0]
    scene['object']['v'] = origin_verts + offset
    scene.configure()
    
    image_opt = render(scene)[0]
    image_gif = img_as_ubyte(to_srgb(image_opt))
    writer.append_data(image_gif)
    
    loss = l1_loss(image, image_opt)
    loss.backward()
    losses.append(loss.detach().cpu().item())
    
    print(f'loss: {loss.detach().cpu():.4f}')
    
    optimizer.step()

writer.close()
plt.plot(range(len(losses)), losses, label='loss')
plt.savefig(file_prefix + '_loss.png')