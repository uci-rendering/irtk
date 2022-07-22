from audioop import add
from copy import deepcopy
from ivt.io import write_png
from ivt.renderer import Renderer
from ivt.connector import ConnectorManager
from ivt.loss import l1_loss
from ivt.transform import *
from common import *
from time import time

from pathlib import Path
import torch
import numpy as np

tests = []
def add_test(func):
    def wrapper():
        print(f'Test ({func.__name__}) starts.\n')
        func()
        print(f'\nTest ({func.__name__}) ends.\n')
    tests.append(wrapper)
    
@add_test
def position_opt():
    # Optimize the x offset of a bunny

    # Config
    loss_func = l1_loss
    num_iters = 400

    output_path = Path('tmp_output', 'kai_test')
    output_path.mkdir(parents=True, exist_ok=True)


    # Get renderer
    render = Renderer('psdr_cuda', device='cuda', dtype=torch.float32)
    render.set_render_options({
        'spp': 16,
        'sppe': 8,
        'sppse': 4,
        'npass': 1,
        'log_level': 0,
    })

    # Make a simple scene
    opt_scene = kai_scene()
    target_scene = deepcopy(opt_scene)

    #  Init values
    rot_axis = torch.tensor([0,1,0], device='cuda', dtype=torch.float32)

    rot_angle0 = torch.tensor([30], device='cuda', dtype=torch.float32).requires_grad_()
    rot_angle1 = torch.tensor([-30], device='cuda', dtype=torch.float32).requires_grad_()
    rot_angle2 = torch.tensor([30], device='cuda', dtype=torch.float32).requires_grad_()
    rot_angle3 = torch.tensor([-10], device='cuda', dtype=torch.float32).requires_grad_()
    rot_angle4 = torch.tensor([30], device='cuda', dtype=torch.float32).requires_grad_()
    rot_angle5 = torch.tensor([-30], device='cuda', dtype=torch.float32).requires_grad_()


    trans_vec0 = torch.tensor([3, 0, 0], device='cuda', dtype=torch.float32).requires_grad_()
    trans_vec1 = torch.tensor([-3, 0, 0], device='cuda', dtype=torch.float32).requires_grad_()
    trans_vec2 = torch.tensor([3, 0, 0], device='cuda', dtype=torch.float32).requires_grad_()
    trans_vec3 = torch.tensor([-3, 0, 0], device='cuda', dtype=torch.float32).requires_grad_()
    trans_vec4 = torch.tensor([2, 0, 0], device='cuda', dtype=torch.float32).requires_grad_()
    trans_vec5 = torch.tensor([2, 1, 0], device='cuda', dtype=torch.float32).requires_grad_()

    # Get variable to change
    init_mat0 = opt_scene.param_map['meshes[0].to_world'].data.clone()
    opt_to_world0 = opt_scene.param_map['meshes[0].to_world']
    init_mat1 = opt_scene.param_map['meshes[1].to_world'].data.clone()
    opt_to_world1 = opt_scene.param_map['meshes[1].to_world']
    init_mat2 = opt_scene.param_map['meshes[2].to_world'].data.clone()
    opt_to_world2 = opt_scene.param_map['meshes[2].to_world']
    init_mat3 = opt_scene.param_map['meshes[3].to_world'].data.clone()
    opt_to_world3 = opt_scene.param_map['meshes[3].to_world']
    init_mat4 = opt_scene.param_map['meshes[4].to_world'].data.clone()
    opt_to_world4 = opt_scene.param_map['meshes[4].to_world']
    init_mat5 = opt_scene.param_map['meshes[5].to_world'].data.clone()
    opt_to_world5 = opt_scene.param_map['meshes[5].to_world']

    # Render target images
    target_images = render(target_scene)
    write_png(output_path / 'target.png', target_images)
    # Prepare for optimization

    params_opt = [
        {'params': trans_vec0, 'lr':  0.05},
        {'params': trans_vec1, 'lr':  0.05},
        {'params': trans_vec2, 'lr':  0.05},
        {'params': trans_vec3, 'lr':  0.05},
        {'params': trans_vec4, 'lr':  0.05},
        {'params': trans_vec5, 'lr':  0.05},
        {'params': rot_angle0, 'lr':  1.0},
        {'params': rot_angle1, 'lr':  1.0},
        {'params': rot_angle2, 'lr':  1.0},
        {'params': rot_angle3, 'lr':  1.0},
        {'params': rot_angle4, 'lr':  1.0},
        {'params': rot_angle5, 'lr':  1.0}
    ]
    optimizer = torch.optim.Adam(params_opt)

    # Start optimization
    for it in range(num_iters):
        t0 = time()

        optimizer.zero_grad()
        
        mat_trans0 = translate(trans_vec0)
        mat_rot0 = rotate(rot_axis, rot_angle0)
        mat_temp0 = torch.mm(mat_trans0, mat_rot0)
        mat0 = torch.mm(init_mat0, mat_temp0)
        opt_to_world0.set(mat0)

        mat_trans1 = translate(trans_vec1)
        mat_rot1 = rotate(rot_axis, rot_angle1)
        mat_temp1 = torch.mm(mat_trans1, mat_rot1)
        mat1 = torch.mm(init_mat1, mat_temp1)
        opt_to_world1.set(mat1)

        mat_trans2 = translate(trans_vec2)
        mat_rot2 = rotate(rot_axis, rot_angle2)
        mat_temp2 = torch.mm(mat_trans1, mat_rot2)
        mat2 = torch.mm(init_mat2, mat_temp2)
        opt_to_world2.set(mat2)

        mat_trans3 = translate(trans_vec3)
        mat_rot3 = rotate(rot_axis, rot_angle3)
        mat_temp3 = torch.mm(mat_trans3, mat_rot3)
        mat3 = torch.mm(init_mat3, mat_temp3)
        opt_to_world3.set(mat3)

        mat_trans4 = translate(trans_vec4)
        mat_rot4 = rotate(rot_axis, rot_angle4)
        mat_temp4 = torch.mm(mat_trans4, mat_rot4)
        mat4 = torch.mm(init_mat4, mat_temp4)
        opt_to_world4.set(mat4)

        mat_trans5 = translate(trans_vec5)
        mat_rot5 = rotate(rot_axis, rot_angle5)
        mat_temp5 = torch.mm(mat_trans5, mat_rot5)
        mat5 = torch.mm(init_mat5, mat_temp5)
        opt_to_world5.set(mat5)

        images = render(opt_scene, [opt_to_world0.data, opt_to_world1.data, opt_to_world2.data, opt_to_world3.data, opt_to_world4.data, opt_to_world5.data])

        write_png(output_path / f"{it}.png", images)
        
        loss = loss_func(target_images, images)
        loss.backward()
        t1 = time()

        print(f'[iter {it}/{num_iters}] loss: {loss.item()} time: {t1 - t0}')

        optimizer.step()

if __name__ == '__main__':
    for test in tests:
        test()