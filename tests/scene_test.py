from ivt.scene import Scene
from pathlib import Path
import igl
import torch
import numpy as np

tests = []
def add_test(func):
    def wrapper():
        print(f'Test ({func.__name__}) starts.\n')
        func()
        print(f'\nTest ({func.__name__}) ends.')
    tests.append(wrapper)

@add_test
def different_backends():
    
    bunny_path = Path('data', 'meshes', 'bunny.obj')
    v, tc, n, f, ftc, fn = igl.read_obj(str(bunny_path))

    scene = Scene(device='cpu')
    scene.add_integrator('path')
    scene.add_render_options({
        'spp': 1
    })
    scene.add_hdr_film(resolution=(512, 512))
    scene.add_perspective_camera(fov=45, origin=(1, 0, 0), target=(0, 0, 0), up=(0, 0, 1))
    scene.add_mesh(v, f, 0)
    scene.add_diffuse_bsdf(torch.tensor((1, 0, 0)).reshape(1, 1, 3))
    scene.add_area_light(mesh_id=0, radiance=(10, 10, 10))
    
    print('backend: torch + cpu')
    print(scene)
    print()
    
    scene.device = 'cuda'
    scene.configure()
    print('backend: torch + cuda')
    print(scene)
    print()
    
    scene.backend = 'numpy'
    scene.ftype = np.float32
    scene.itype = np.int64
    scene.device = 'cpu'
    scene.configure()
    print('backend: numpy')
    print(scene)

if __name__ == '__main__':
    for test in tests:
        test()