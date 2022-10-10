from ivt.scene import Scene
from ivt.io import *
from ivt.transform import *
from ivt.sampling import sample_sphere
from pathlib import Path

import gin

@gin.configurable('simple_microfacet')
def make_scene(
    mesh_path,
    diffuse,
    specular,
    roughness,
    device,
    ftype,
    itype,
    num_sensors=104,
    radius=2,
    fov=45,
    target=(0, 0, 0),
    up=(0, 1, 0)):

    scene = Scene(device=device, ftype=ftype, itype=itype)

    for origin in sample_sphere(num_sensors, radius, 'fibonacci'):
        scene.add_perspective_camera(fov, origin, target, up)

    scene.add_obj(mesh_path, 0)

    read_mat = lambda mat : read_exr(mat) if isinstance(mat, str) else mat

    diffuse = read_mat(diffuse)
    specular = read_mat(specular)
    roughness = read_mat(roughness)
    scene.add_microfacet_bsdf(diffuse, specular, roughness)

    return scene
