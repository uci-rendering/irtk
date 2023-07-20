# run from ./inv-render-toolkit with `python -m examples.2_pytorch3d_demo.code`

import ivt
from ivt.scene import *
from ivt.renderer import Renderer
from ivt.io import write_image
from ivt.loss import l1_loss

import pytorch3d.renderer as pr

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
image = render(scene)[0]
write_image('output/armadillo_pytorch3d.png', image[..., :3])

# render grad
# scene.set('film', HDRFilm(width=64, height=64))
# grad_image = render.connector.renderGrad(scene, render.render_options)[0]
# write_image('output/armadillo_pytorch3d_grad.png', grad_image)
# write_image('output/armadillo_pytorch3d_diff.png', grad_image - image)

# simple optimization
cam_offset_x = torch.Tensor([0.3])
cam_offset_x.requires_grad_()
to_world_offset = torch.zeros((4, 4)).to('cuda')
to_world_offset[0, 3] = cam_offset_x[0]
original_to_world = scene['sensor2']['to_world']
scene['sensor2']['to_world'] = original_to_world + to_world_offset
image_init = render(scene)[0]
write_image('output/armadillo_pytorch3d_init.png', image_init[..., :3])

num_iter = 20
optimizer = torch.optim.Adam([cam_offset_x], lr=0.05)
for i in range(num_iter):
    optimizer.zero_grad()
    to_world_offset = torch.zeros((4, 4)).to('cuda')
    to_world_offset[0, 3] = cam_offset_x[0]
    scene['sensor2']['to_world'] = original_to_world + to_world_offset
    scene.configure()
    
    image_opt = render(scene)[0]
    # write_image(f'output/opt/armadillo_pytorch3d_{i}.png', image_opt[..., :3])
    loss = l1_loss(image, image_opt)
    loss.backward()
    
    print(f'loss: {loss.detach().cpu():.4f}')
    
    optimizer.step()
    
    # cache, pytorch3d_params = render.connector.update_scene_objects(scene, render.render_options)
    # raster_settings = pr.RasterizationSettings(
    #     image_size=cache['film'], 
    #     blur_radius=0.0, 
    #     faces_per_pixel=1, 
    # )
    # renderer = pr.MeshRenderer(
    #     rasterizer=pr.MeshRasterizer(
    #         cameras=cache['camera'], 
    #         raster_settings=raster_settings
    #     ),
    #     shader=pr.SoftPhongShader(
    #         device='cuda', 
    #         cameras=cache['camera'],
    #         lights=cache['light']
    #     )
    # )
    # image_opt = renderer(cache['mesh'])[..., :3]
    # write_image(f'output/opt/armadillo_pytorch3d_{i}.png', image_opt)
    # loss = l1_loss(image, image_opt)
    # loss.backward()
    
    # with torch.no_grad():
    #     cam_offset_x += cam_offset_x.grad * 10
    
    # print(cam_offset_x.grad)
    # print(cam_offset_x)
    # print()
    
    