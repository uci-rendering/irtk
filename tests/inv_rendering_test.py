from ivt.io import write_png
from ivt.renderer import Renderer
from ivt.loss import l1_loss
from common import *

from pathlib import Path
import torch
import numpy as np

tests = []
def add_test(func):
    def wrapper():
        print(f'Test ({func.__name__}) starts.\n')
        func()
        print(f'\nTest ({func.__name__}) ends.')
    tests.append(wrapper)
    
@add_test # comment this to skip the test
def simple_mat_opt():
    # Config
    loss_func = l1_loss
    num_iters = 100
    lr = 1e-2

    # Get renderer
    renderer = Renderer('psdr_enzyme', device='cpu', dtype=torch.float32)

    # Make a simple scene
    scene = simple_scene()

    # Target & init values
    target_v = np.array((0.8, 0.8, 0.8)).reshape(1, 1, 3)
    init_v = np.array((0.8, 0.5, 0.1)).reshape(1, 1, 3)
    
    # Render target images
    scene.bsdfs[0]['reflectance'].set(target_v)
    with torch.no_grad():
        target_images = renderer(scene, [])

    # Set the intial parameters
    reflectance = scene.bsdfs[0]['reflectance']
    reflectance.set(init_v)
    reflectance.requires_grad = True

    # Prepare for optimization
    reflectance_tensor = torch.tensor(reflectance.data, requires_grad=True)
    params = [reflectance_tensor]
    optimizer = torch.optim.Adam(params, lr=lr)
    param_names = scene.get_requiring_grad()
    num_params = len(param_names)

    # Start optimization
    for iter in range(num_iters):
        optimizer.zero_grad()

        images = renderer(scene, params)

        loss = loss_func(target_images, images)
        loss.backward()

        print(f'[iter {iter}/{num_iters}] loss: {loss.item()}')

        optimizer.step()
        
        for i in range(num_params):
            scene.param_map[param_names[i]].set(params[i])

    print()
    print('Target: ')
    print(target_v)
    print('Optimized: ')
    print(reflectance.data)
        



if __name__ == '__main__':
    for test in tests:
        test()