from copy import deepcopy
from ivt.io import write_png
from ivt.renderer import Renderer
from ivt.connector import ConnectorManager
from ivt.loss import l1_loss
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
    
@add_test # comment this to skip the test
def simple_mat_opt():
    # Config
    loss_func = l1_loss
    num_iters = 100
    lr = 1e-2

    cm = ConnectorManager()
    scene = simple_scene()

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
        target_reflectance.configure()
        target_images = render(target_scene)

        # Set the intial parameters
        reflectance = scene.bsdfs[0]['reflectance']
        reflectance.set(init_v)
        reflectance.requires_grad = True
        reflectance.configure()

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
                scene.param_map[param_names[i]].set(params[i])
                scene.param_map[param_names[i]].configure()

        print()
        print('Target: ')
        print(target_v)
        print('Optimized: ')
        print(reflectance.data)

if __name__ == '__main__':
    for test in tests:
        test()