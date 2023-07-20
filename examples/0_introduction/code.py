# run from ./inv-render-toolkit with `python -m examples.0_introduction.code`

import ivt
from ivt.scene import *
from ivt.renderer import Renderer
from ivt.io import write_image
from ivt.loss import l1_loss

scene = Scene()
scene.set('armadillo', Mesh.from_file('./examples/data/meshes/armadillo.obj', mat_id='blue'))
scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
# scene.set('envlight', EnvironmentLight.from_file('./examples/data/envmaps/factory.exr'))
scene.set('sensor', PerspectiveCamera.from_lookat(fov=40, origin=(-1.5, 1.5, 1.5), target=(0, 0, 0), up=(0, 1, 0)))
scene.set('film', HDRFilm(width=512, height=512))
# scene.set('integrator', Integrator(type='path', config={
#     'max_depth': 1,
#     'hide_emitters': False
# }))
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
image = render(scene)[0]

write_image('output/armadillo_psdr.png', image)

# simple optimization
cam_offset_x = torch.Tensor([0.3])
cam_offset_x.requires_grad_()
to_world_offset = torch.zeros((4, 4)).to('cuda')
to_world_offset[0, 3] = cam_offset_x[0]
original_to_world = scene['sensor']['to_world']
scene['sensor']['to_world'] = original_to_world + to_world_offset
image_init = render(scene)[0]
write_image('output/armadillo_psdr_init.png', image_init)

num_iter = 20
optimizer = torch.optim.Adam([cam_offset_x], lr=0.05)
for i in range(num_iter):
    optimizer.zero_grad()
    to_world_offset = torch.zeros((4, 4)).to('cuda')
    to_world_offset[0, 3] = cam_offset_x[0]
    scene['sensor']['to_world'] = original_to_world + to_world_offset
    scene.configure()
    
    image_opt = render(scene)[0]
    # write_image(f'output/opt/armadillo_psdr_{i}.png', image_opt)
    loss = l1_loss(image, image_opt)
    loss.backward()

    print(f'loss: {loss[0].detach().cpu():.4f}')
    
    optimizer.step()