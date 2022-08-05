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
    
# @add_test # comment this to skip the test
def mat_opt():
    # Optimize the diffuse component of a bunny

    # Config
    loss_func = l1_loss
    num_iters = 100
    lr = 1e-2

    cm = ConnectorManager()

    # Test every connector available
    for cn in cm.get_availability_list():
        print(f'Optimizing with connector [{cn}]...')

        # Get renderer
        render = Renderer(cn, device='cuda', dtype=torch.float32)
        render.set_render_options(simple_render_options[cn])

        # Make a simple scene
        scene = simple_scene()
        target_scene = deepcopy(scene)

        # Target & init values
        target_v = (0.8, 0.8, 0.8)
        init_v = (0.5, 0.5, 0.5)
        
        # Render target images
        target_reflectance = target_scene.bsdfs[0]['reflectance']
        target_reflectance.set(target_v)
        target_images = render(target_scene)

        # Set the intial parameters
        reflectance = scene.bsdfs[0]['reflectance']
        reflectance.set(init_v)
        reflectance.set_requires_grad()

        # Prepare for optimization
        param_names = scene.get_requiring_grad()
        params = [scene.param_map[param_name].data for param_name in param_names]
        optimizer = torch.optim.Adam(params, lr=lr)
        num_params = len(param_names)

        # Start optimization
        for iter in range(num_iters):
            t0 = time()

            optimizer.zero_grad()

            images = render(scene, params)

            loss = loss_func(target_images, images)
            loss.backward()
            
            t1 = time()

            print(f'[iter {iter}/{num_iters}] loss: {loss.item()} time: {t1 - t0}')

            optimizer.step()
            
            for i in range(num_params):
                scene.param_map[param_names[i]].updated = True

        print()
        print('Target: ')
        print(target_v)
        print('Optimized: ')
        print(reflectance.data)

@add_test
def position_opt():
    # Optimize the x offset of a bunny

    # Config
    loss_func = l1_loss
    num_iters = 100
    lr = 0.05

    cm = ConnectorManager()

    # Test every connector available
    for cn in cm.get_availability_list():
        print(f'Optimizing with connector [{cn}]...')

        # Get renderer
        render = Renderer(cn, device='cuda', dtype=torch.float32)
        render.set_render_options(simple_render_options[cn])

        # Make a simple scene
        opt_scene = simple_scene()
        target_scene = deepcopy(opt_scene)

        #  Init values
        opt_x = torch.tensor(2, device='cuda', dtype=torch.float32).requires_grad_()
        x_axis = torch.tensor([1, 0, 0], device='cuda', dtype=torch.float32)
        
        # Get variable to change
        opt_to_world = opt_scene.param_map['meshes[0].to_world']

        # Render target images
        target_images = render(target_scene)

        # Prepare for optimization
        optimizer = torch.optim.Adam([opt_x], lr=lr)

        # Start optimization
        for iter in range(num_iters):
            t0 = time()

            optimizer.zero_grad()

            opt_to_world.set(translate(opt_x * x_axis))
            images = render(opt_scene, [opt_to_world.data])
            
            loss = loss_func(target_images, images)
            loss.backward()
            t1 = time()

            print(f'[iter {iter}/{num_iters}] loss: {loss.item()} time: {t1 - t0} x: {opt_x.item()}')

            optimizer.step()

        print()
        print('Target: ')
        print(0)
        print('Optimized: ')
        print(opt_x.item())

if __name__ == '__main__':
    for test in tests:
        test()