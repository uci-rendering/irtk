from ivt.scene import Scene
from ivt.io import *
from ivt.transform import *
from ivt.sampling import sample_sphere

from pathlib import Path

data_path = Path('data')
meshes_path = data_path / 'meshes'
cached_scenes_path = Path('cached_scenes')
cached_scenes_path.mkdir(parents=True, exist_ok=True)

scene_name = 'bunny'

scene = Scene(backend='torch', device='cuda')

num_sensors = 100
radius = 3
fov = 45
target = (0, 0, 0)
up = (0, 1, 0)
for origin in sample_sphere(num_sensors, radius, 'fibonacci'):
    scene.add_perspective_camera(fov, origin, target, up)

v, tc, n, f, ftc, fn = read_obj(meshes_path / 'bunny.obj')
scene.add_mesh(v, f, 0)

# light_0_to_world = translate([0, 4, 0]) @ rotate([1, 0, 0], 180) @ scale(0.1)
# v, tc, n, f, ftc, fn = read_obj(meshes_path / 'plane.obj')
# scene.add_mesh(v, f, 1, use_face_normal=True, to_world=light_0_to_world)

# light_1_to_world = translate([0, 4, 0]) @ scale(0.2)
# v, tc, n, f, ftc, fn = read_obj(meshes_path / 'plane.obj')
# scene.add_mesh(v, f, 1, use_face_normal=True, to_world=light_1_to_world)

scene.add_diffuse_bsdf((0.5, 0.6, 0.7))
scene.add_diffuse_bsdf((0.8, 0.8, 0.8))

# scene.add_area_light(mesh_id=1, radiance=(10000, 10000, 10000))
# scene.add_area_light(mesh_id=2, radiance=(10000, 10000, 10000))

torch.save(scene, cached_scenes_path / f'{scene_name}.pt')