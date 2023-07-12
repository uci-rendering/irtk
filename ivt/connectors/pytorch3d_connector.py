from ..connector import Connector
from ..scene import *
from ..config import *
from ..io import write_mesh
from collections import OrderedDict

from pytorch3d.structures import Meshes, join_meshes_as_batch
import pytorch3d.renderer as pr
import torch

import time
import os

class PyTorch3DConnector(Connector, connector_name='pytorch3d'):

    backend = 'torch'
    device = 'cuda'
    ftype = torch.float32
    itype = torch.long

    def __init__(self):
        super().__init__()

    def update_scene_objects(self, scene, render_options):
        if 'pytorch3d' in scene.cached:
            cache = scene.cached['pytorch3d']
        else:
            cache = {}
            scene.cached['pytorch3d'] = cache

            cache['meshes'] = []
            cache['textures'] = dict()
            cache['cameras'] = []
            cache['raster_settings'] = None
            cache['lights'] = None

            cache['name_map'] = {}
            cache['integrators'] = OrderedDict()

        drjit_params = []
        for name in scene.components:
            component = scene[name]
            if type(component) in self.extensions:
                drjit_params += self.extensions[type(component)](name, scene)
            else:
                raise RuntimeError(f'Unsupported component for PyTorch3D: {component}')
        
        # change light here
        # lights = pr.AmbientLights(ambient_color=((0.5, 0.5, 0.5), ), device=device)
        lights = pr.PointLights(location=cache['cameras'][0].get_camera_center(), device=device)
        cache['lights'] = lights
        cache['mesh'] = join_meshes_as_batch(cache['meshes'])
        return scene.cached['pytorch3d'], drjit_params

    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        cache, _ = self.update_scene_objects(scene, render_options)

        npass = render_options['npass']
        images = []
        for sensor_id in sensor_ids:
            renderer = pr.MeshRenderer(
                rasterizer=pr.MeshRasterizer(
                    cameras=cache['cameras'][sensor_id], 
                    raster_settings=cache['raster_settings']
                ),
                shader=pr.SoftPhongShader(
                    device=device, 
                    cameras=cache['cameras'][sensor_id],
                    lights=cache['lights']
                )
            )
            image = None
            for i in range(npass):
                image_pass = renderer(cache['mesh'])[..., :3]
                if image:
                    image += image_pass / npass
                else:
                    image = image_pass / npass
            images.append(image)

        return images
        
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        pass

"""
@PSDRJITConnector.register(Integrator)
def process_integrator(name, scene):
    integrator = scene[name]
    cache = scene.cached['psdr_jit']

    if integrator['type'] == 'field':
        psdr_integrator = psdr_jit.FieldExtractionIntegrator(integrator['config']['type'])
        cache['integrators'][name] = psdr_integrator
    elif integrator['type'] == 'collocated':
        psdr_integrator = psdr_jit.CollocatedIntegrator(integrator['config']['intensity'])
        cache['integrators'][name] = psdr_integrator
    elif integrator['type'] == 'path':
        psdr_integrator = psdr_jit.PathTracer(integrator['config']['max_depth'])
        cache['integrators'][name] = psdr_integrator
        if 'hide_emitters' in integrator['config']:
            psdr_integrator.hide_emitters = integrator['config']['hide_emitters']
    else:
        raise RuntimeError(f"unrecognized integrator type: {integrator['type']}")

    return []
"""

@PyTorch3DConnector.register(HDRFilm)
def process_hdr_film(name, scene):
    film = scene[name]
    cache = scene.cached['pytorch3d']

    # TODO: check settings when renderD
    cache['raster_settings'] = pr.RasterizationSettings(
        image_size=(film['height'], film['width']), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    return []

@PyTorch3DConnector.register(PerspectiveCamera)
def process_perspective_camera(name, scene):
    sensor = scene[name]
    cache = scene.cached['pytorch3d']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        R = sensor['to_world'][:3, :3][None]
        T = -torch.mm(sensor['to_world'][:3, 3][None], R[0, ...])
        
        pytorch_sensor = pr.FoVPerspectiveCameras(
            znear=sensor['near'], 
            zfar=sensor['far'], 
            fov=sensor['fov'], 
            R=R, 
            T=T, 
            device=device
        )
        
        cache['cameras'] += pytorch_sensor
        cache['name_map'][name] = pytorch_sensor

    pytorch_sensor = cache['name_map'][name]
    
    return []
    
    # # Update parameters
    # updated = sensor.get_updated()
    # if len(updated) > 0:
    #     for param_name in updated:
    #         if param_name == "to_world":
    #             psdr_sensor.to_world = Matrix4fD(sensor['to_world'].reshape(1, 4, 4))
    #         sensor.params[param_name]['updated'] = False

    # # Enable grad for parameters requiring grad
    # drjit_params = []
    # requiring_grad = sensor.get_requiring_grad()
    # if len(requiring_grad) > 0:
    #     for param_name in requiring_grad:
    #         if param_name == "to_world":
    #             drjit_param = psdr_sensor.to_world
    #             drjit.enable_grad(drjit_param)
    #             drjit_params.append(drjit_param)

    # return drjit_params

@PyTorch3DConnector.register(Mesh)
def process_mesh(name, scene):
    mesh = scene[name]
    cache = scene.cached['pytorch3d']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        # Create its material first
        mat_id = mesh['mat_id']
        if mat_id not in scene:
            raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
        PyTorch3DConnector.extensions[type(scene[mat_id])](mat_id, scene)
        brdf = cache['textures'][mat_id]
        
        verts = mesh['v'].to(device)
        faces = mesh['f'].to(device)
        textures = pr.TexturesUV(brdf[None], mesh['fuv'][None].long(), mesh['uv'][None])
        pytorch3d_mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=textures
        )
        cache['meshes'] += pytorch3d_mesh
        cache['name_map'][name] = pytorch3d_mesh

    pytorch3d_mesh = cache['name_map'][name]
    
    return []

    # # Update parameters
    # updated = mesh.get_updated()
    # if len(updated) > 0:
    #     for param_name in updated:
    #         if param_name == 'v':
    #             if mesh['can_change_topology']:
    #                 psdr_mesh.load_raw(Vector3fC(mesh['v']), Vector3iC(mesh['f']))
    #             else:
    #                 psdr_mesh.vertex_positions = Vector3fC(mesh['v'])
    #                 psdr_mesh.face_indices = Vector3iC(mesh['f'])

    #         mesh.params[param_name]['updated'] = False

    # # Enable grad for parameters requiring grad
    # drjit_params = []
    # requiring_grad = mesh.get_requiring_grad()
    # if len(requiring_grad) > 0:
    #     for param_name in requiring_grad:
    #         if param_name == 'v':
    #             drjit_param = psdr_mesh.vertex_positions
    #             drjit.enable_grad(drjit_param)
    #             drjit_params.append(drjit_param)

    # return drjit_params

def convert_color(color, c, bitmap=True):
    if color.shape == ():
        color = color.tile(c)
    if color.shape == (c,):
        color = color.reshape(1, 1, c)
    h, w, _ = color.shape
    if c == 3:
        color = Vector3fD(color.reshape(-1, c))
        if bitmap:
            color = psdr_jit.Bitmap3fD(w, h, color)
    elif c == 1:
        color = FloatD(color.reshape(-1))
        if bitmap:
            color = psdr_jit.Bitmap1fD(w, h, color)
    else:
        raise RuntimeError("Not support bitmap channel number: {c}")
    return color

@PyTorch3DConnector.register(DiffuseBRDF)
def process_diffuse_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['pytorch3d']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        if brdf['d'].dim() == 1:
            pytorch3d_brdf = brdf['d'].unsqueeze(0).unsqueeze(0)
        elif brdf['d'].dim() == 2:
            pytorch3d_brdf = brdf['d'].unsqueeze(0)
        elif brdf['d'].dim() == 3:
            pytorch3d_brdf = brdf['d']
        else:
            raise RuntimeError(f"Unsupported diffuse brdf: {name}.")
        
        cache['textures'][name] = pytorch3d_brdf
        cache['name_map'][name] = pytorch3d_brdf

    pytorch3d_brdf = cache['name_map'][name]

    return []

    # # Update parameters
    # updated = brdf.get_updated()
    # if len(updated) > 0:
    #     for param_name in updated:
    #         if param_name == 'd':
    #             psdr_brdf.reflectance = convert_color(brdf['d'], 3)
    #         brdf.params[param_name]['updated'] = False

    # # Enable grad for parameters requiring grad
    # drjit_params = []
    # requiring_grad = brdf.get_requiring_grad()
    # if len(requiring_grad) > 0:
    #     for param_name in requiring_grad:
    #         if param_name == 'd':
    #             drjit_param = psdr_brdf.reflectance.data
    #             drjit.enable_grad(drjit_param)
    #             drjit_params.append(drjit_param)

    # return drjit_params

"""
@PSDRJITConnector.register(MicrofacetBRDF)
def process_microfacet_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['psdr_jit']
    psdr_scene = cache['scene']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        d = convert_color(brdf['d'], 3)
        s = convert_color(brdf['s'], 3)
        r = convert_color(brdf['r'], 1)

        psdr_bsdf = psdr_jit.MicrofacetBSDF(s, d, r)
        psdr_scene.add_BSDF(psdr_bsdf, name)
        cache['name_map'][name] = f"BSDF[id={name}]"

    psdr_brdf = psdr_scene.param_map[cache['name_map'][name]]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                psdr_brdf.diffuseReflectance.data = convert_color(brdf['d'], 3, bitmap=False)
            elif param_name == 's':
                psdr_brdf.specularReflectance.data = convert_color(brdf['s'], 3, bitmap=False)
            elif param_name == 'r':
                psdr_brdf.roughness.data = convert_color(brdf['r'], 1, bitmap=False)
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    drjit_params = []
    
    def enable_grad(drjit_param):
        drjit.enable_grad(drjit_param)
        drjit_params.append(drjit_param)

    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'd':
                enable_grad(psdr_brdf.diffuseReflectance.data)
            elif param_name == 's':
                enable_grad(psdr_brdf.specularReflectance.data)
            elif param_name == 'r':
                enable_grad(psdr_brdf.roughness.data)

    return drjit_params

@PSDRJITConnector.register(EnvironmentLight)
def process_environment_light(name, scene):
    emitter = scene[name]
    cache = scene.cached['psdr_jit']
    psdr_scene = cache['scene']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        radiance = convert_color(emitter['radiance'], 3)

        psdr_emitter = psdr_jit.EnvironmentMap()
        psdr_emitter.radiance = radiance
        psdr_emitter.set_transform(Matrix4fD(emitter['to_world'].reshape(1, 4, 4)))
        psdr_scene.add_EnvironmentMap(psdr_emitter)
        cache['name_map'][name] = f"Emitter[{psdr_scene.get_num_emitters() - 1}]"

    psdr_emitter = psdr_scene.param_map[cache['name_map'][name]]

    # Update parameters
    updated = emitter.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'radiance':
                psdr_emitter.radiance = convert_color(emitter['radiance'], 3)
            elif param_name == 'to_world':
                psdr_emitter.set_transform(Matrix4fD(emitter['to_world'].reshape(1, 4, 4)))
            emitter.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    drjit_params = []
    
    def enable_grad(drjit_param):
        drjit.enable_grad(drjit_param)
        drjit_params.append(drjit_param)

    requiring_grad = emitter.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'radiance':
                enable_grad(psdr_emitter.radiance.data)
                
    return drjit_params
"""