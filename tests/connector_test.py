from ivt.scene import Scene
from ivt.io import read_obj, write_png
from pathlib import Path
import torch
import numpy as np
from connectors import PSDREnzymeConnector

tests = []
def add_test(func):
    def wrapper():
        print(f'Test ({func.__name__}) starts.\n')
        func()
        print(f'\nTest ({func.__name__}) ends.')
    tests.append(wrapper)

@add_test
def render():
    meshes_path = Path('tests', 'scenes', 'bunny', 'meshes')
    output_path = Path('tmp_output')
    output_path.mkdir(parents=True, exist_ok=True)

    scene = Scene(backend='numpy')
    
    scene.add_integrator('direct')
    
    scene.add_render_options({
        'seed': 42,
        'num_samples': 32,
        'max_bounces': 1,
        'num_samples_primary_edge': 4,
        'num_samples_secondary_edge': 4,
        'quiet': False
    })
    
    scene.add_hdr_film(resolution=(512, 512))
    
    scene.add_perspective_camera(fov=45, origin=(0, 0, 30), target=(0, 0, 0), up=(0, 1, 0))
    
    v, f = read_obj(meshes_path / 'bunny.obj')
    scene.add_mesh(v, f, 0)
    
    v, f = read_obj(meshes_path / 'light_0.obj')
    scene.add_mesh(v, f, 0)
    
    v, f = read_obj(meshes_path / 'light_1.obj')
    scene.add_mesh(v, f, 0)
    
    scene.add_diffuse(torch.tensor((0.8, 0.8, 0.8)).reshape(1, 1, 3))
    
    scene.add_area_light(mesh_id=1, radiance=(50, 100, 80))
    scene.add_area_light(mesh_id=2, radiance=(100, 70, 50))
    

    connector = PSDREnzymeConnector()
    images = connector.renderC(scene)
    for i, image in enumerate(images):
        write_png(output_path / f'{i}.png', image)
    
    print('Done')

if __name__ == '__main__':
    for test in tests:
        test()