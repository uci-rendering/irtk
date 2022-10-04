from ivt.scene import Scene
from ivt.io import *
from ivt.transform import *
from ivt.sampling import sample_sphere

from pathlib import Path

data_path = Path('data')
meshes_path = data_path / 'meshes'
cached_scenes_path = Path('cached_scenes')
cached_scenes_path.mkdir(parents=True, exist_ok=True)

scene_name = 'armadillo'

scene = Scene(device='cuda')

num_sensors = 104
radius = 2
fov = 45
target = (0, 0, 0)
up = (0, 1, 0)
for origin in sample_sphere(num_sensors, radius, 'fibonacci'):
    scene.add_perspective_camera(fov, origin, target, up)

scene.add_obj(meshes_path / 'armadillo.obj', 0)

scene.add_microfacet_bsdf((0.2, 0.6, 0.7), (0.04, 0.04, 0.04), 0.3)

torch.save(scene, cached_scenes_path / f'{scene_name}.pt')