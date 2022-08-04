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
        'spp': 4,
        'sppe': 4,
        'sppse': 4,
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

bunny_render_options = {
    'psdr_enzyme': {
        'seed': 42,
        'num_samples': 10,
        'max_bounces': 10,
        'num_samples_primary_edge': 0,
        'num_samples_secondary_edge': 0,
        'quiet': False
    },
}
def vol_bunny_scene(backend='torch', device='cpu'):
    meshes_path = Path('tests', 'scenes', 'vol_bunny', 'meshes')

    scene = Scene(backend=backend, device=device)

    scene.add_integrator('volpath')

    scene.add_hdr_film(resolution=(256, 256))

    scene.add_perspective_camera(fov=13,
                                 origin=(201.868, 315.266, 383.194),
                                 target=(202.242, 315.857, 383.908),
                                 up=(-0.260633, 0.806177, -0.531178))

    scene.add_null_bsdf()
    scene.add_diffuse_bsdf((0., 0., 0.))

    scene.add_homogeneous_medium(sigmaT=0.7, albedo=(0.5, 0.7, 1.))

    v, _, _, f, _, _ = read_obj(meshes_path / 'bunny.obj')
    scene.add_mesh(v, f, 0, med_int_id=0)

    v, _, _, f, _, _ = read_obj(meshes_path / 'emitter.obj')
    scene.add_mesh(v, f, 0)

    scene.add_area_light(mesh_id=1, radiance=(100, 100, 100))

    return scene
