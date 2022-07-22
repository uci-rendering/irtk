from ivt.scene import Scene
from ivt.io import read_obj
from ivt.transform import *

from pathlib import Path
import numpy as np

simple_render_options = {
    'psdr_enzyme': {
        'seed': 42,
        'num_samples': 4,
        'max_bounces': 1,
        'num_samples_primary_edge': 4,
        'num_samples_secondary_edge': 4,
        'quiet': False
    }, 

    'psdr_cuda': {
        'spp': 16,
        'sppe': 16,
        'sppse': 0,
        'npass': 1,
        'log_level': 0,
    }
}

def simple_scene(backend='torch', device='cuda'):
    meshes_path = Path('tests', 'scenes', 'bunny', 'meshes')
    
    scene = Scene(backend=backend, device=device)
    
    scene.add_integrator('direct')
    
    scene.add_hdr_film(resolution=(512, 512))
    
    scene.add_perspective_camera(fov=45, origin=(0, 0, 30), target=(0, 0, 0), up=(0, 1, 0))
    
    v, tc, n, f, ftc, fn = read_obj(meshes_path / 'bunny.obj')
    scene.add_mesh(v, f, 0)
    
    v, tc, n, f, ftc, fn = read_obj(meshes_path / 'light_0.obj')
    scene.add_mesh(v, f, 1)
    
    v, tc, n, f, ftc, fn = read_obj(meshes_path / 'light_1.obj')
    scene.add_mesh(v, f, 1)
    
    scene.add_diffuse_bsdf((0.8, 0.8, 0.8))
    scene.add_diffuse_bsdf((0.8, 0.8, 0.8))
    
    scene.add_area_light(mesh_id=1, radiance=(50, 100, 80))
    scene.add_area_light(mesh_id=2, radiance=(100, 70, 50))
    
    return scene

def kai_scene(backend='torch', device='cuda'):
    meshes_path = Path('tests', 'scenes', 'room')
    scene = Scene(backend=backend, device=device)
    scene.add_integrator('collocated', {'intensity': 2000})
    scene.add_hdr_film(resolution=(256, 256))
    scene.add_perspective_camera(fov=50, origin=(20,20,20), target=(0, 0, 0), up=(0, 1, 0))
    v, tc, n, f, ftc, fn = read_obj(meshes_path / 'chair.obj')

    to_world1 = translate(torch.tensor([0, 0, 10], device='cuda', dtype=torch.float32))
    temp = rotate(torch.tensor([0,1,0], device='cuda', dtype=torch.float32), 180)
    to_world1 = torch.mm(to_world1, temp)
    scene.add_mesh(v, f, 0, to_world=to_world1)

    to_world2 = translate(torch.tensor([0, 0, -10], device='cuda', dtype=torch.float32))
    scene.add_mesh(v, f, 0, to_world=to_world2)

    to_world3 = translate(torch.tensor([10, 0, 0], device='cuda', dtype=torch.float32))
    temp = rotate(torch.tensor([0,1,0], device='cuda', dtype=torch.float32), -90)
    to_world3 = torch.mm(to_world3, temp)

    scene.add_mesh(v, f, 0, to_world=to_world3)

    to_world4 = translate(torch.tensor([-10, 0, 0], device='cuda', dtype=torch.float32))
    temp = rotate(torch.tensor([0,1,0], device='cuda', dtype=torch.float32), 90)
    to_world4 = torch.mm(to_world4, temp)
    scene.add_mesh(v, f, 0, to_world=to_world4)

    v3,_,_,f3,_,_ = read_obj(meshes_path / 'table.obj')
    scene.add_mesh(v3, f3, 0)

    v4,_,_,f4,_,_ = read_obj(meshes_path / 'bunny_low.obj')
    to_world7 = translate(torch.tensor([0, 7, 0], device='cuda', dtype=torch.float32))
    scene.add_mesh(v4, f4, 0, to_world=to_world7)


    v1, tc1, n1, f1, ftc1, fn1 = read_obj(meshes_path / 'plane.obj')
    to_world5 = translate(torch.tensor([0, -0.01, 0], device='cuda', dtype=torch.float32))
    temp = scale(torch.tensor([5,1,5], device='cuda', dtype=torch.float32))
    to_world5 = torch.mm(to_world5, temp)
    scene.add_mesh(v1, f1, 1, to_world=to_world5, use_face_normal=True)

    scene.add_diffuse_bsdf((0.9, 0.5, 0.5))
    scene.add_diffuse_bsdf((0.5, 0.5, 0.5))
    return scene