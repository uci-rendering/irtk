from ..connector import Connector
from ..scene import *
from ..config import *
from ..io import write_mesh
from collections import OrderedDict

import drjit
import psdr_jit
from drjit.scalar import Array3f
from drjit.cuda import Array3f as Vector3fC, Array3i as Vector3iC
from drjit.cuda.ad import Array3f as Vector3fD, Float32 as FloatD, Matrix4f as Matrix4fD, Matrix3f as Matrix3fD
from drjit.cuda.ad import Float32 as FloatD
import torch

import time
import os

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.label}] Elapsed time: {elapsed_time} seconds")

class PSDRJITConnector(Connector, connector_name='psdr_jit'):

    backend = 'torch'
    device = 'cuda'
    ftype = torch.float32
    itype = torch.long

    def __init__(self):
        super().__init__()

    def update_scene_objects(self, scene, render_options):
        if 'psdr_jit' in scene.cached:
            cache = scene.cached['psdr_jit']
        else:
            cache = {}
            scene.cached['psdr_jit'] = cache

            cache['scene'] = psdr_jit.Scene()
            cache['scene'].opts.spp = render_options['spp']
            cache['scene'].opts.sppe = render_options['sppe']
            cache['scene'].opts.sppse = render_options['sppse']
            cache['scene'].opts.log_level = render_options['log_level']

            cache['name_map'] = {}
            cache['integrators'] = OrderedDict()
            cache['configured'] = False

        drjit_params = []
        for name in scene.components:
            component = scene[name]
            drjit_params += self.extensions[type(component)](name, scene)

        if not cache['configured']:
            cache['scene'].configure() 
            cache['configured'] = True
        
        return scene.cached['psdr_jit'], drjit_params

    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        cache, _ = self.update_scene_objects(scene, render_options)

        cache['scene'].configure(sensor_ids)

        npass = render_options['npass']
        h, w, c = cache['film']['shape']
        if type(integrator_id) == int:
            integrator = list(cache['integrators'].values())[integrator_id]
        elif type(integrator_id) == str:
            integrator = cache['integrators'][integrator_id]
        else:
            raise RuntimeError('integrator_id is invalid: {integrator_id}')

        images = []
        for sensor_id in sensor_ids:
            image = torch.zeros((h * w, c)).to(device).to(ftype)
            for i in range(npass):
                image_pass = integrator.renderC(cache['scene'], sensor_id).torch()
                image += image_pass / npass
            image = image.reshape(h, w, c)
            images.append(image)

        return images
        
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        cache, drjit_params = self.update_scene_objects(scene, render_options)

        cache['scene'].configure(sensor_ids)

        npass = render_options['npass']
        h, w, c = cache['film']['shape']
        if type(integrator_id) == int:
            integrator = list(cache['integrators'].values())[integrator_id]
        elif type(integrator_id) == str:
            integrator = cache['integrators'][integrator_id]
        else:
            raise RuntimeError('integrator_id is invalid: {integrator_id}')
        
        param_grads = [torch.zeros_like(scene[param_name]) for param_name in scene.requiring_grad]

        for i, sensor_id in enumerate(sensor_ids):
            image_grad = Vector3fC(image_grads[i].reshape(-1, 3) / npass)
            for j in range(npass):
                image = integrator.renderD(cache['scene'], sensor_id)
                tmp = drjit.dot(image_grad, image)
                drjit.backward(tmp)

                for param_grad, drjit_param in zip(param_grads, drjit_params):
                    grad = drjit.grad(drjit_param).torch().to(device).to(ftype)
                    grad = torch.nan_to_num(grad).reshape(param_grad.shape)
                    param_grad += grad

        return param_grads

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

@PSDRJITConnector.register(HDRFilm)
def process_hdr_film(name, scene):
    film = scene[name]
    cache = scene.cached['psdr_jit']

    cache['film'] = {
        'shape': (film['height'], film['width'], 3)
    }
    cache['scene'].opts.width = film['width']
    cache['scene'].opts.height = film['height']

    return []

@PSDRJITConnector.register(PerspectiveCamera)
def process_perspective_camera(name, scene):
    sensor = scene[name]
    cache = scene.cached['psdr_jit']
    psdr_scene = cache['scene']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        psdr_sensor = psdr_jit.PerspectiveCamera(sensor['fov'], sensor['near'], sensor['far'])
        psdr_sensor.to_world = Matrix4fD(sensor['to_world'].reshape(1, 4, 4))
        psdr_scene.add_Sensor(psdr_sensor)
        cache['name_map'][name] = f"Sensor[{psdr_scene.num_sensors - 1}]"

    psdr_sensor = psdr_scene.param_map[cache['name_map'][name]]
    
    # Update parameters
    updated = sensor.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == "to_world":
                psdr_sensor.to_world = Matrix4fD(sensor['to_world'].reshape(1, 4, 4))
            sensor.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    drjit_params = []
    requiring_grad = sensor.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == "to_world":
                drjit_param = psdr_sensor.to_world
                drjit.enable_grad(drjit_param)
                drjit_params.append(drjit_param)

    return drjit_params

@PSDRJITConnector.register(Mesh)
def process_mesh(name, scene):
    mesh = scene[name]
    cache = scene.cached['psdr_jit']
    psdr_scene = cache['scene']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        # Create its material first
        mat_id = mesh['mat_id']
        if mat_id not in scene:
            raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
        brdf = scene[mat_id]
        PSDRJITConnector.extensions[type(brdf)](mat_id, scene)

        # TODO: Fix this workaround when psdr-jit updates psdr_mesh.load_raw()
        if mesh['can_change_topology']:
            psdr_mesh = psdr_jit.Mesh()
            psdr_mesh.load_raw(Vector3fC(mesh['v']), Vector3iC(mesh['f']))
            psdr_mesh.use_face_normal = mesh['use_face_normal']

            psdr_emitter = psdr_jit.AreaLight(mesh['radiance'].tolist()) if mesh['is_emitter'] else None
            psdr_scene.add_Mesh(psdr_mesh, mat_id, psdr_emitter)
        else:
            write_mesh('__psdr_jit_tmp__.obj', mesh['v'], mesh['f'], mesh['uv'], mesh['fuv'])
            psdr_emitter = psdr_jit.AreaLight(mesh['radiance']) if mesh['is_emitter'] else None
            psdr_scene.add_Mesh('__psdr_jit_tmp__.obj', mesh['to_world'].reshape(1, 4, 4), mat_id, psdr_emitter)
            os.remove('__psdr_jit_tmp__.obj')
        
        cache['name_map'][name] = f"Mesh[{psdr_scene.num_meshes - 1}]"

    psdr_mesh = psdr_scene.param_map[cache['name_map'][name]]

    # Update parameters
    updated = mesh.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'v':
                if mesh['can_change_topology']:
                    psdr_mesh.load_raw(Vector3fC(mesh['v']), Vector3iC(mesh['f']))
                else:
                    psdr_mesh.vertex_positions = Vector3fC(mesh['v'])
                    psdr_mesh.face_indices = Vector3iC(mesh['f'])

            mesh.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    drjit_params = []
    requiring_grad = mesh.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'v':
                drjit_param = psdr_mesh.vertex_positions
                drjit.enable_grad(drjit_param)
                drjit_params.append(drjit_param)

    return drjit_params

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

@PSDRJITConnector.register(DiffuseBRDF)
def process_diffuse_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['psdr_jit']
    psdr_scene = cache['scene']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        d = convert_color(brdf['d'], 3)
        psdr_bsdf = psdr_jit.DiffuseBSDF(d)
        psdr_scene.add_BSDF(psdr_bsdf, name)
        cache['name_map'][name] = f"BSDF[id={name}]"

    psdr_brdf = psdr_scene.param_map[cache['name_map'][name]]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                psdr_brdf.reflectance = convert_color(brdf['d'], 3)
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    drjit_params = []
    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'd':
                drjit_param = psdr_brdf.reflectance.data
                drjit.enable_grad(drjit_param)
                drjit_params.append(drjit_param)

    return drjit_params

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