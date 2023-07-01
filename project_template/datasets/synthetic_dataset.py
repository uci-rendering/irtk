from ivt.io import *
from ivt.config import *
from ivt.scene import *
from ivt.renderer import Renderer
from ivt.transform import *
from ivt.sampling import sample_sphere

import torch
from torch.utils.data import Dataset

from pathlib import Path
from copy import deepcopy
import gin

@gin.configurable
class SyntheticDataset(Dataset):
    def __init__(self, scene, target_path, render_tar):
        self.target_scene = scene
        self.target_path = Path(target_path)
        self.render_tar = render_tar

        self.num_sensors = len(self.target_scene.filter(PerspectiveCamera))
        self.w = self.target_scene['film.width']
        self.h = self.target_scene['film.height']

        if not self.target_path.exists():
            self.target_path.mkdir(parents=True)
            self.render_target_images()

    def __len__(self):
        return self.num_sensors

    def __getitem__(self, idx):
        return to_torch_f(read_image(self.target_path / f'{idx}.exr'))
    
    def render_target_images(self):
        print(f'rendering target images to {self.target_path}...')
        for i in range(self.num_sensors):
            image = self.render_tar(self.target_scene, sensor_ids=[i])[0]
            write_image(self.target_path / f'{i}.exr', image)
        self.target_scene.clear_cache()
        print(f'Done.')

    def get_rays(self, idx):
        sensor = self.target_scene[f'sensor {idx}']
        u = to_torch_f(torch.arange(self.w) + 0.5)
        v = to_torch_f(torch.arange(self.h) + 0.5)
        u, v = torch.meshgrid(u, v, indexing='xy')
        u = u.flatten() / self.w
        v = 1 - (v.flatten() / self.h)
        samples = torch.stack([u, v], dim=1)
        return sensor.get_rays(samples, self.w / self.h)
    
    def get_scene(self):
        return deepcopy(self.target_scene)

@gin.configurable
def simple_microfacet(
    film_res_w,
    film_res_h,
    mesh_path,
    envmap_path,
    diffuse,
    specular,
    roughness,
    num_sensors=100,
    radius=2,
    fov=45,
    target=(0, 0, 0),
    up=(0, 1, 0)):
    """
    Create a simple scene consisting of an object, which has a microfacet BRDF, and 
    some sensors sampled on the surface a sphere.
    """

    scene = Scene()

    for i, origin in enumerate(sample_sphere(num_sensors, radius, 'fibonacci')):
        scene.set(f'sensor {i}', PerspectiveCamera.from_lookat(fov, origin, target, up))

    read_mat = lambda mat : read_image(mat)[..., :3] if isinstance(mat, str) else mat
    diffuse = read_mat(diffuse)
    specular = read_mat(specular)
    roughness = read_mat(roughness)
    scene.set('mat', MicrofacetBRDF(diffuse, specular, roughness))
    if len(scene['mat.r'].shape) == 3 and scene['mat.r'].shape[2] == 3:
        scene['mat.r'] = scene['mat.r'][..., 0:1]

    scene.set('mesh', Mesh.from_file(mesh_path, 'mat', use_face_normal=False))

    scene.set('envmap', EnvironmentLight.from_file(envmap_path))
    if scene['envmap.radiance'].shape[2] == 4:
        scene['envmap.radiance'] = scene['envmap.radiance'][..., 0:3]
    
    scene.set('film', HDRFilm(film_res_w, film_res_h))

    scene.set('integrator', Integrator('path', {'max_depth': 3, 'hide_emitters': False}))

    return scene