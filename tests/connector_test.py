from genericpath import exists
from ivt.io import write_png
from ivt.connector import ConnectorManager
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
        print(f'\nTest ({func.__name__}) ends.\n')
    tests.append(wrapper)
    
@add_test # comment this to skip the test
def renderC():
    output_path = Path('tmp_output', 'connector_test', 'renderC')

    cm = ConnectorManager()
    scene = simple_scene()
    
    # Test every connector available
    for cn in cm.get_availability_list():
        print(f'Rendering with connector [{cn}]...')
        render = Renderer(cn, device='cuda', dtype=torch.float32)
        render.set_render_options(simple_render_options[cn])

        # Render the images without gradient 
        with torch.no_grad():
            images = render(scene)

        # Write the images
        output_connector_path = output_path / cn
        output_connector_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            write_png(output_connector_path / f'{i}.png', image)


@add_test # comment this to skip the test
def renderC_vol():
    output_path = Path('tmp_output', 'connector_test', 'renderC_vol')

    cm = ConnectorManager()
    # scene = simple_scene()
    scene = vol_bunny_scene()
    
    # Test every connector available
    for cn in cm.get_availability_list():
        print(f'Rendering with connector [{cn}]...')
        render = Renderer(cn, device='cuda', dtype=torch.float32)
        # scene.add_render_options(simple_render_options[cn])
        scene.add_render_options(bunny_render_options[cn])

        # Render the images without gradient 
        with torch.no_grad():
            images = render(scene)

        # Write the images
        output_connector_path = output_path / cn
        output_connector_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            write_png(output_connector_path / f'{i}.png', image)
    return
    

@add_test
def renderD():
    output_path = Path('tmp_output', 'connector_test', 'renderD')
    output_path.mkdir(parents=True, exist_ok=True)

    cm = ConnectorManager()
    scene = simple_scene()

    # Test every connector available
    for cn in cm.get_availability_list():
        print(f'Rendering with connector [{cn}]...')
        render = Renderer(cn, device='cuda', dtype=torch.float32)
        render.set_render_options(simple_render_options[cn])

        # Render target images 
        target_images = render(scene)
        
        # Modify the parameters and set requires_grad 
        scene.param_map['bsdfs[0].reflectance'].set((0.5, 0.6, 0.7))
        scene.param_map['bsdfs[0].reflectance'].set_requires_grad()
        scene.param_map['meshes[0].vertex_positions'].data += 1e-4
        scene.param_map['meshes[0].vertex_positions'].set_requires_grad()

        # Get parameters that requires gradient
        param_names = scene.get_requiring_grad()
        params = [scene.param_map[param_name].data for param_name in param_names]

        
        # Render the new images 
        images = render(scene, params)

        # Get gradient
        loss = l1_loss(target_images, images)
        loss.backward()

        # Print the gradient
        for i, param_name in enumerate(param_names):
            print(f'gradient of {param_name}:')
            print(params[i].grad)
            print()

if __name__ == '__main__':
    for test in tests:
        test()