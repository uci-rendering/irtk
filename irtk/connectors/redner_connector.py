from ..connector import Connector
from ..scene import *
from ..config import *
from ..io import write_mesh
from collections import OrderedDict

import pyredner
import torch
import time

# FIXME: Redner depends on OptiX Prime which fails with RTX 30 series.
pyredner.set_use_gpu(False)
pyredner.set_print_timing(False)

class RednerConnector(Connector, connector_name='redner'):

    backend = 'torch'
    device = 'cuda'
    ftype = torch.float32
    itype = torch.long

    def __init__(self):
        super().__init__()
        self.render_time = 0

    def update_scene_objects(self, scene, render_options):
        if 'redner' in scene.cached:
            cache = scene.cached['redner']
        else:
            cache = {}
            scene.cached['redner'] = cache

            cache['meshes'] = []
            cache['textures'] = {}
            cache['cameras'] = []
            cache['integrators'] = OrderedDict()
            cache['envlights'] = []
            cache['film'] = None
            cache['name_map'] = {}
        
        redner_params = []
        for name in scene.components:
            component = scene[name]
            redner_params += self.extensions[type(component)](name, scene)
        
        return scene.cached['redner'], redner_params

    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        cache, _ = self.update_scene_objects(scene, render_options)

        # Meshs
        objects = cache['meshes']

        images = []
        for sensor_id in sensor_ids:
            # Sensors
            # camera = pyredner.automatic_camera_placement(objects, resolution=(cache['film']['height'], cache['film']['width']))
            camera = pyredner.Camera(fov=to_torch_f([cache['cameras'][sensor_id]['fov']]).cpu(), 
                                     cam_to_world=cache['cameras'][sensor_id]['cam_to_world'].clone(),
                                     resolution=(cache['film']['height'], cache['film']['width']))
            if len(cache['cameras']) >= 1:
                camera.fov = to_torch_f([cache['cameras'][sensor_id]['fov']]).cpu()
                camera.cam_to_world = cache['cameras'][sensor_id]['cam_to_world'].clone()
                camera.cam_to_world[:3, 0] = -camera.cam_to_world[:3, 0]

            # Environment Lights
            if len(cache['envlights']) >= 1:
                redner_scene = pyredner.Scene(camera = camera, objects = objects, envmap = cache['envlights'][0])
            if 'light_intensity' in render_options:
                light = pyredner.generate_quad_light(position = camera.cam_to_world[:3, 3],
                                                    look_at = camera.cam_to_world[:3, 2] + camera.cam_to_world[:3, 3],
                                                    size = torch.tensor([1.0, 1.0]),
                                                    intensity = torch.tensor(render_options['light_intensity']))
                objects.append(light)
                redner_scene = pyredner.Scene(camera = camera, objects = objects)

            # redner Scene
            max_bounces = 1
            if len(cache['integrators']) >= 1:
                integrator = list(cache['integrators'].items())[0]
                max_bounces = integrator[1]['max_bounces']

            npass = render_options['npass']
            h, w, c = (cache['film']['height'], cache['film']['width'], 3)
        
            image = torch.zeros((h, w, c)).to(configs['device']).to(configs['ftype']).cpu()
            for i in range(npass):
                t = time.time()
                image_pass = render_pathtracing(redner_scene, max_bounces = max_bounces, num_samples = (render_options['spp'], 4))
                self.render_time += time.time() - t
                image += image_pass / npass
            images.append(image)
            
            # remove temp light
            if 'light_intensity' in render_options:
                objects.pop()

        return images
        
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        with torch.enable_grad():
            cache, redner_params = self.update_scene_objects(scene, render_options)

            # Meshs
            objects = cache['meshes']

            param_grads = [torch.zeros_like(scene[param_name]) for param_name in scene.requiring_grad]
            for sensor_index, sensor_id in enumerate(sensor_ids):
                # Sensors
                # camera = pyredner.automatic_camera_placement(objects, resolution=(cache['film']['height'], cache['film']['width']))
                camera = pyredner.Camera(fov=to_torch_f([cache['cameras'][sensor_id]['fov']]).cpu(), 
                                         cam_to_world=cache['cameras'][sensor_id]['cam_to_world'].clone(),
                                         resolution=(cache['film']['height'], cache['film']['width']))
                if len(cache['cameras']) >= 1:
                    camera.fov = to_torch_f([cache['cameras'][sensor_id]['fov']]).cpu()
                    camera.cam_to_world = cache['cameras'][sensor_id]['cam_to_world'].clone()
                    camera.cam_to_world[:3, 0] = -camera.cam_to_world[:3, 0]

                # Environment Lights
                if len(cache['envlights']) >= 1:
                    redner_scene = pyredner.Scene(camera = camera, objects = objects, envmap = cache['envlights'][0])
                if 'light_intensity' in render_options:
                    light = pyredner.generate_quad_light(position = camera.cam_to_world[:3, 3],
                                                    look_at = camera.cam_to_world[:3, 2] + camera.cam_to_world[:3, 3],
                                                    size = torch.tensor([1.0, 1.0]),
                                                    intensity = torch.tensor(render_options['light_intensity']))
                    objects.append(light)
                    redner_scene = pyredner.Scene(camera = camera, objects = objects)
                
                # redner Scene
                max_bounces = 1
                if len(cache['integrators']) >= 1:
                    integrator = list(cache['integrators'].items())[0]
                    max_bounces = integrator[1]['max_bounces']
                
                npass = render_options['npass']
                h, w, c = (cache['film']['height'], cache['film']['width'], 3)
            
                image_grad = image_grads[sensor_index] / npass
                for i in range(npass):
                    t = time.time()
                    image_pass = render_pathtracing(redner_scene, max_bounces = max_bounces, num_samples = (render_options['spp'], 4))
                    tmp = (image_grad[..., :3] * image_pass).sum(dim=2)
                    redner_grads = torch.autograd.grad(tmp, redner_params, torch.ones_like(tmp), retain_graph=True)
                    for param_grad, redner_grad in zip(param_grads, redner_grads):
                        param_grad += to_torch_f(torch.nan_to_num(redner_grad))
                        
                    self.render_time += time.time() - t
                        
                # remove temp light
                if 'light_intensity' in render_options:
                    objects.pop()

            return param_grads

@RednerConnector.register(Integrator)
def process_integrator(name, scene):
    integrator = scene[name]
    cache = scene.cached['redner']

    redner_integrator = {
        'type': integrator['type']
    }
    if integrator['type'] == 'path':
        redner_integrator['max_bounces'] = integrator['config']['max_depth']
        if 'hide_emitters' in integrator['config']:
            redner_integrator['hide_emitters'] = integrator['config']['hide_emitters']
        cache['integrators'][name] = redner_integrator
    else:
        raise RuntimeError(f"unrecognized integrator type: {integrator['type']}")

    return []

@RednerConnector.register(HDRFilm)
def process_hdr_film(name, scene):
    film = scene[name]
    cache = scene.cached['redner']

    cache['film'] = {
        'height': film['height'],
        'width': film['width']
    }

    return []

@RednerConnector.register(PerspectiveCamera)
def process_perspective_camera(name, scene):
    sensor = scene[name]
    cache = scene.cached['redner']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        redner_sensor = {
            'type': 'perspective',
            'clip_near': sensor['near'],
            'clip_far': sensor['far'],
            'fov': sensor['fov'],
            'cam_to_world': sensor['to_world'].cpu()
        }
        cache['cameras'].append(redner_sensor)
        cache['name_map'][name] = redner_sensor

    redner_sensor = cache['name_map'][name]
    
    # Update parameters
    updated = sensor.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == "to_world":
                redner_sensor['cam_to_world'] = sensor['to_world'].cpu()
            sensor.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    redner_params = []
    requiring_grad = sensor.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == "to_world":
                redner_sensor['cam_to_world'].requires_grad_()
                redner_params.append(redner_sensor['cam_to_world'])

    return redner_params

@RednerConnector.register(Mesh)
def process_mesh(name, scene):
    mesh = scene[name]
    cache = scene.cached['redner']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        # Create its material first
        mat_id = mesh['mat_id']
        if mat_id not in scene:
            raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
        brdf = scene[mat_id]
        RednerConnector.extensions[type(brdf)](mat_id, scene)

        verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(configs['device'])), dim=1)
        verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
        verts = verts.contiguous()
        
        if mesh['uv'].nelement() == 0:
            mesh['uv'] = torch.zeros((1, 2)).to(configs['device'])
        if mesh['fuv'].nelement() == 0:
            mesh['fuv'] = torch.zeros_like(mesh['f']).to(configs['device'])
        
        material = cache['textures'][mat_id]
        vts_normals = compute_vertex_normals(verts, mesh['f'].long())
        redner_mesh = pyredner.Object(vertices = verts.cpu(),
                                      indices = mesh['f'].cpu(),
                                      uvs = mesh['uv'].cpu(),
                                      uv_indices = mesh['fuv'].cpu(),
                                      normals = vts_normals.cpu(),
                                      normal_indices = mesh['f'].cpu(),
                                      material = material)

        cache['meshes'].append(redner_mesh)
        cache['name_map'][name] = redner_mesh

    redner_mesh = cache['name_map'][name]

    # Update parameters
    updated = mesh.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'v':
                verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(configs['device'])), dim=1)
                verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
                verts = verts.contiguous()
                if mesh['uv'].nelement() == 0:
                    mesh['uv'] = torch.zeros((1, 2)).to(configs['device'])
                if mesh['fuv'].nelement() == 0:
                    mesh['fuv'] = torch.zeros_like(mesh['f']).to(configs['device'])
                vts_normals = compute_vertex_normals(verts, mesh['f'].long())
                redner_mesh.vertices = verts.cpu()
                redner_mesh.normals = vts_normals.cpu()
            mesh.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    redner_params = []
    requiring_grad = mesh.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'v':
                redner_mesh.vertices.requires_grad = True
                redner_params.append(redner_mesh.vertices)

    return redner_params

@RednerConnector.register(DiffuseBRDF)
def process_diffuse_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['redner']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        kd = brdf['d']
        if kd.dim() == 4:
            kd = kd[0]
        if len(kd) > 3:
            # Flip texture
            kd = torch.flip(kd, [0])
        redner_brdf = pyredner.Material(diffuse_reflectance = kd.cpu())
        cache['textures'][name] = redner_brdf
        cache['name_map'][name] = redner_brdf

    redner_brdf = cache['name_map'][name]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                kd = brdf['d']
                if kd.dim() == 4:
                    kd = kd[0]
                if len(kd) > 3:
                    # Flip texture
                    kd = torch.flip(kd, [0])
                redner_brdf.diffuse_reflectance = kd.cpu()
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    redner_params = []
    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'd':
                redner_brdf.diffuse_reflectance.requires_grad = True
                redner_params.append(redner_brdf.diffuse_reflectance)

    return redner_params

@RednerConnector.register(MicrofacetBRDF)
def process_microfacet_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['redner']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        kd = brdf['d']
        if kd.dim() == 4:
            kd = kd[0]
        if len(kd) > 3:
            # Flip texture
            kd = torch.flip(kd, [0])
        ks = brdf['s']
        if ks.dim() == 4:
            ks = ks[0]
        if len(ks) > 3:
            # Flip texture
            ks = torch.flip(ks, [0])
        kr = brdf['r']
        if kr.dim() == 0:
            kr = to_torch_f([kr])
        elif kr.dim() == 4:
            kr = kr[0]
        if len(kr) > 3:
            # Flip texture
            kr = torch.flip(kr, [0])
        redner_brdf = pyredner.Material(diffuse_reflectance = kd.cpu(), 
                                        specular_reflectance = ks.cpu(),
                                        roughness = kr.cpu())
        cache['textures'][name] = redner_brdf
        cache['name_map'][name] = redner_brdf

    redner_brdf = cache['name_map'][name]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                kd = brdf['d']
                if kd.dim() == 4:
                    kd = kd[0]
                if len(kd) > 3:
                    # Flip texture
                    kd = torch.flip(kd, [0])
                redner_brdf.diffuse_reflectance = kd.cpu()
            elif param_name == 's':
                ks = brdf['s']
                if ks.dim() == 4:
                    ks = ks[0]
                if len(ks) > 3:
                    # Flip texture
                    ks = torch.flip(ks, [0])
                redner_brdf.specular_reflectance = ks.cpu()
            elif param_name == 'r':
                kr = brdf['r']
                if kr.dim() == 0:
                    kr = to_torch_f([kr])
                elif kr.dim() == 4:
                    kr = kr[0]
                if len(kr) > 3:
                    # Flip texture
                    kr = torch.flip(kr, [0])
                redner_brdf.roughness = kr.cpu()
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    redner_params = []
    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'd':
                redner_brdf.diffuse_reflectance.requires_grad = True
                redner_params.append(redner_brdf.diffuse_reflectance)
            elif param_name == 's':
                redner_brdf.specular_reflectance.requires_grad = True
                redner_params.append(redner_brdf.specular_reflectance)
            elif param_name == 'r':
                redner_brdf.roughness.requires_grad = True
                redner_params.append(redner_brdf.roughness)

    return redner_params

@RednerConnector.register(EnvironmentLight)
def process_environment_light(name, scene):
    emitter = scene[name]
    cache = scene.cached['redner']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        radiance = emitter['radiance']
        if radiance.dim() == 1:
            radiance = radiance.reshape(1, 1, 3)
        elif radiance.dim() == 4:
            radiance = radiance[0]
        redner_emitter = pyredner.EnvironmentMap(values = radiance.cpu(),
                                                 env_to_world = emitter['to_world'].cpu())

        cache['envlights'].append(redner_emitter)
        cache['name_map'][name] = redner_emitter

    redner_emitter = cache['name_map'][name]

    # Update parameters
    updated = emitter.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'radiance':
                radiance = emitter['radiance']
                if radiance.dim() == 1:
                    radiance = radiance.reshape(1, 1, 3)
                elif radiance.dim() == 4:
                    radiance = radiance[0]
                redner_emitter.values = radiance.cpu()
            elif param_name == 'to_world':
                redner_emitter.env_to_world = emitter['to_world'].cpu()
            emitter.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    redner_params = []
    requiring_grad = emitter.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'radiance':
                redner_emitter.values.requires_grad = True
                redner_params.append(redner_emitter.values)

    return redner_params

#----------------------------------------------------------------------------
# Helpers.
def compute_vertex_normals(verts, faces):
    """Computes the packed version of vertex normals from the packed verts
        and faces. This assumes verts are shared between faces. The normal for
        a vertex is computed as the sum of the normals of all the faces it is
        part of weighed by the face areas.
    """
    # Create a zeroed array with the same type and shape as verts
    verts_normals = torch.zeros_like(verts)

    # Create an indexed view into the verts array
    tris = verts[faces]

    # Calculate the normal for all triangles
    tri_normals = torch.cross(
        tris[:, 2] - tris[:, 1],
        tris[:, 0] - tris[:, 1],
        dim=1,
    )

    # Add the normals through indexed view
    verts_normals = verts_normals.index_add(
        0, faces[:, 0], tri_normals
    )
    verts_normals = verts_normals.index_add(
        0, faces[:, 1], tri_normals
    )
    verts_normals = verts_normals.index_add(
        0, faces[:, 2], tri_normals
    )

    # Normalize normals
    return torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )

from typing import Union, Tuple, Optional, List
import random
def render_pathtracing(scene: Union[pyredner.Scene, List[pyredner.Scene]],
                       max_bounces: int = 1,
                       num_samples: Union[int, Tuple[int, int]] = (4, 4),
                       device: Optional[torch.device] = None):
    seed = random.randint(0, 16777216)
    integrator = pyredner.integrators.WarpFieldIntegrator(
        num_samples = num_samples,
        max_bounces = max_bounces,
        sampler_type = pyredner.sampler_type.sobol,
    )
    scene_args = pyredner.RenderFunction.serialize_scene_class(\
        scene = scene,
        integrator = integrator,
        device = device)
    return pyredner.RenderFunction.apply(seed, *scene_args)