from ivt.scene import Scene
from ivt.io import read_obj, write_png
from ivt.connector import ConnectorManager
from ivt.loss import l1_loss

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
    
def simple_scene():
    meshes_path = Path('tests', 'scenes', 'bunny', 'meshes')
    
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
    scene.add_mesh(v, f, 1)
    
    v, f = read_obj(meshes_path / 'light_1.obj')
    scene.add_mesh(v, f, 1)
    
    scene.add_diffuse_bsdf(np.array((0.8, 0.8, 0.8)).reshape(1, 1, 3))
    scene.add_diffuse_bsdf(np.array((0.8, 0.8, 0.8)).reshape(1, 1, 3))
    
    scene.add_area_light(mesh_id=1, radiance=(50, 100, 80))
    scene.add_area_light(mesh_id=2, radiance=(100, 70, 50))
    
    return scene

@add_test # comment this to skip the test
def renderC():
    output_path = Path('tmp_output', 'connector_test', 'renderC')
    output_path.mkdir(parents=True, exist_ok=True)

    cm = ConnectorManager()
    assert cm.is_available('psdr_enzyme')
    connector = cm.get_connector('psdr_enzyme')

    scene = simple_scene()
    images = connector.renderC(scene)
    for i, image in enumerate(images):
        write_png(output_path / f'{i}.png', image)
    
    print('Done')
    
@add_test
def renderD():
    output_path = Path('tmp_output', 'connector_test', 'renderD')
    output_path.mkdir(parents=True, exist_ok=True)

    cm = ConnectorManager()
    assert cm.is_available('psdr_enzyme')
    connector = cm.get_connector('psdr_enzyme')

    scene = simple_scene()
    target_images = connector.renderC(scene)
    target_images = [torch.from_numpy(target_image).to('cpu') for target_image in target_images]
    
    scene.param_map['bsdfs[0].reflectance'].set(np.array((0.5, 0.6, 0.7)).reshape(1, 1, 3))
    scene.param_map['bsdfs[0].reflectance'].requires_grad = True
    
    scene.param_map['meshes[0].vertex_positions'].data += 1
    scene.param_map['meshes[0].vertex_positions'].requires_grad = True
    scene.param_map['meshes[0].vertex_positions'].configure()
    
    param_names = scene.get_requiring_grad()
    
    param_grads = connector.renderD(target_images, l1_loss, scene)
    
    for param_name, param_grad in zip(param_names, param_grads):
        print(f'gradient of {param_name}:')
        print(param_grad)
        print()
    
    print('Done')

if __name__ == '__main__':
    for test in tests:
        test()