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
    num_iters = 10
    lr = 0.05

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
    trans_vec = torch.tensor([3, 0, 0], device='cuda', dtype=torch.float32).requires_grad_()
    # Get variable to change
    prev_mat = opt_scene.param_map['meshes[0].to_world'].data.clone()
    opt_to_world = opt_scene.param_map['meshes[0].to_world']

    # Render target images
    target_images = render(target_scene)
    write_png(output_path / 'target.png', target_images)
    # Prepare for optimization
    optimizer = torch.optim.Adam([trans_vec], lr=lr)

    # Start optimization
    for it in range(num_iters):
        t0 = time()

        optimizer.zero_grad()
        
        mat = torch.mm(prev_mat, translate(trans_vec))
        opt_to_world.set(mat)
        images = render(opt_scene, [opt_to_world.data])

        write_png(output_path / f"{it}.png", images)
        
        loss = loss_func(target_images, images)
        loss.backward()
        t1 = time()

        print(f'[iter {it}/{num_iters}] loss: {loss.item()} time: {t1 - t0} x: {trans_vec}')

        optimizer.step()

    print()
    print('Target: ')
    print(0)
    print('Optimized: ')
    print(trans_vec)

if __name__ == '__main__':
    for test in tests:
        test()