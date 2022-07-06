from ivt.scene import Scene
from ivt.io import read_obj

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
        'spp': 1,
        'sppe': 1,
        'sppse': 1,
        'npass': 4,
        'log_level': 0,
    }
}

def simple_scene(backend='torch', device='cuda'):
    meshes_path = Path('tests', 'scenes', 'bunny', 'meshes')
    
    scene = Scene(backend=backend, device=device)
    
    scene.add_integrator('direct')
    
    scene.add_hdr_film(resolution=(512, 512))
    
    scene.add_perspective_camera(fov=45, origin=(0, 0, 30), target=(0, 0, 0), up=(0, 1, 0))
    
    v, f = read_obj(meshes_path / 'bunny.obj')
    scene.add_mesh(v, f, 0)
    
    v, f = read_obj(meshes_path / 'light_0.obj')
    scene.add_mesh(v, f, 1)
    
    v, f = read_obj(meshes_path / 'light_1.obj')
    scene.add_mesh(v, f, 1)
    
    scene.add_diffuse_bsdf((0.8, 0.8, 0.8))
    scene.add_diffuse_bsdf((0.8, 0.8, 0.8))
    
    scene.add_area_light(mesh_id=1, radiance=(50, 100, 80))
    scene.add_area_light(mesh_id=2, radiance=(100, 70, 50))
    
    return scene
