from ..connector import Connector
from ..scene import *
from ..io import write_mesh
from collections import OrderedDict
from ..utils import Timer

import drjit
import psdr_jit
from drjit.scalar import Array3f
from drjit.cuda import Array3f as Vector3fC, Array3i as Vector3iC
from drjit.cuda.ad import Array3f as Vector3fD, Array1f as Vector1fD, Float32 as FloatD, Matrix4f as Matrix4fD, Matrix3f as Matrix3fD
from drjit.cuda.ad import Float32 as FloatD
import torch

import os
from ..utils import Timer
class PSDRJITConnector(Connector, connector_name='psdr_jit'):

    def __init__(self):
        super().__init__()
        
        self.debug = False

        self.default_render_options = {
            'spp': 64,
            'sppe': 0,
            'sppse': 0,
            'log_level': 0,
            'npass': 1,
            'seed': 0,
            'guiding_options': {
                'type': 'none'
            }
        }

        self.default_render_options = {
            'spp': 64,
            'sppe': 0,
            'sppse': 0,
            'log_level': 0,
            'npass': 1,
            'seed': 0,
            'guiding_options': {
                'type': 'none'
            }
        }

    def update_scene_objects(self, scene, render_options):
        if 'psdr_jit' in scene.cached:
            cache = scene.cached['psdr_jit']
        else:
            cache = {}
            scene.cached['psdr_jit'] = cache
            cache['scene'] = psdr_jit.Scene()
            cache['name_map'] = {}
            cache['integrators'] = OrderedDict()
            cache['configured'] = False

            def clean_up():
                drjit.flush_malloc_cache()
                drjit.flush_kernel_cache()

            cache['clean_up'] = clean_up

        for k in self.default_render_options:
            if k not in render_options:
                render_options[k] = self.default_render_options[k]

        psdr_scene = cache['scene']

        psdr_scene.opts.spp = render_options['spp']
        psdr_scene.opts.sppe = render_options['sppe']
        psdr_scene.opts.sppse = render_options['sppse']
        psdr_scene.opts.log_level = render_options['log_level']

        drjit_params = []
        for name in scene.components:
            component = scene[name]
            component_type = str.split(str(type(component)), '.')[-1][:-2]
            with Timer(f"'{component_type}' preparation", False):
                drjit_params += self.extensions[type(component)](name, scene)

        if not cache['configured']:
            psdr_scene.configure() 
            cache['configured'] = True
        
        return scene.cached['psdr_jit'], drjit_params

    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        with Timer('-- Prepare Scene', prt=self.debug, record=False):
            cache, _ = self.update_scene_objects(scene, render_options)
            psdr_scene = cache['scene']

            psdr_scene.configure(sensor_ids)

            npass = render_options['npass']
            h, w, c = cache['film']['shape']
            if type(integrator_id) == int:
                integrator = list(cache['integrators'].values())[integrator_id]
            elif type(integrator_id) == str:
                integrator = cache['integrators'][integrator_id]
            else:
                raise RuntimeError('integrator_id is invalid: {integrator_id}')

        with Timer('-- Backend Forward', prt=self.debug, record=False):
            images = []
            for sensor_id in sensor_ids:
                seed = render_options['seed']
                image = to_torch_f(torch.zeros((h * w, c)))
                for i in range(npass):
                    image_pass = integrator.renderC(psdr_scene, sensor_id, seed, cache['film']['pixel_idx']).torch().to(image)
                    image += image_pass / npass
                    seed += 1
                image = image.reshape(h, w, c)
                images.append(image)

        return images
        
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        with Timer('-- Prepare Scene', prt=self.debug, record=False):
            cache, drjit_params = self.update_scene_objects(scene, render_options)
            psdr_scene = cache['scene']
            
            psdr_scene.configure(sensor_ids)

            npass = render_options['npass']
            if type(integrator_id) == int:
                psdr_integrator = list(cache['integrators'].values())[integrator_id]
            elif type(integrator_id) == str:
                psdr_integrator = cache['integrators'][integrator_id]
            else:
                raise RuntimeError('integrator_id is invalid: {integrator_id}')
            
            param_grads = [torch.zeros_like(scene[param_name]) for param_name in scene.requiring_grad]

        with Timer('-- Backend Backward', prt=self.debug, record=False):
            for i, sensor_id in enumerate(sensor_ids):
                seed = render_options['seed']
                image_grad = Vector3fC(image_grads[i].reshape(-1, 3) / npass)
                self.preprocess_guiding(psdr_integrator, psdr_scene, sensor_id, render_options['guiding_options'], seed)
                for j in range(npass):
                    image = psdr_integrator.renderD(psdr_scene, sensor_id, seed, cache['film']['pixel_idx'])
                    tmp = drjit.dot(image_grad, image)
                    drjit.backward(tmp)

                    for param_grad, drjit_param in zip(param_grads, drjit_params):
                        grad = to_torch_f(drjit.grad(drjit_param).torch())
                        grad = torch.nan_to_num(grad).reshape(param_grad.shape)
                        param_grad += grad

                seed += 1

        return param_grads

    def preprocess_guiding(self, psdr_integrator, psdr_scene, sensor_id, guiding_options, seed):
        if isinstance(psdr_integrator, psdr_jit.PathTracer):
            if guiding_options['type'] == 'grid':
                psdr_integrator.preprocess_secondary_edges(psdr_scene, sensor_id, guiding_options['res'], guiding_options['nrounds'], seed)

    
    def forward_ad_mesh_translation(self, mesh_id, scene, render_options, sensor_ids=[0], integrator_id=0):
        cache, drjit_params = self.update_scene_objects(scene, render_options)
        psdr_scene = cache['scene']

        assert len(drjit_params) == 0
        assert len(sensor_ids) == 1

        P = FloatD(0.) 
        drjit.enable_grad(P) 
        psdr_mesh = psdr_scene.param_map[cache['name_map'][mesh_id]]
        psdr_mesh.set_transform(Matrix4fD([[1.,0.,0.,P],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],]))

        psdr_scene.configure(sensor_ids)

        seed = render_options['seed']
        npass = render_options['npass']
        h, w, c = cache['film']['shape']
        if type(integrator_id) == int:
            psdr_integrator = list(cache['integrators'].values())[integrator_id]
        elif type(integrator_id) == str:
            psdr_integrator = cache['integrators'][integrator_id]
        else:
            raise RuntimeError('integrator_id is invalid: {integrator_id}')

        
        image = to_torch_f(torch.zeros((h * w, c)))
        grad_image = to_torch_f(torch.zeros((h * w, c)))

        self.preprocess_guiding(psdr_integrator, psdr_scene, sensor_ids[0], render_options['guiding_options'], seed)
        
        for j in range(npass):
            drjit_image = psdr_integrator.renderD(psdr_scene, sensor_ids[0], seed)
            image += to_torch_f(drjit_image.torch()) / npass

            drjit.set_grad(P, 1.0)
            drjit.enqueue(drjit.ADMode.Forward, P)
            drjit.traverse(drjit.cuda.ad.Float, drjit.ADMode.Forward, drjit.ADFlag.ClearInterior)
            drjit_grad_image = drjit.grad(drjit_image)
            grad_image += to_torch_f(drjit_grad_image.torch()) / npass
            seed += 1

        image = image.reshape(h, w, c)
        grad_image = grad_image.reshape(h, w, c)
        return image, grad_image

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
    elif integrator['type'] == 'direct':
        psdr_integrator = psdr_jit.Direct(integrator['config']['mis'])
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
    psdr_scene = cache['scene']

    h = film['height']
    w = film['width']

    if film['crop_window'] is not None:
        h_lower, w_lower, h_upper, w_upper = film['crop_window']
        
        all_id = torch.arange(h * w).reshape(h, w)
        pixel_idx = all_id[h_lower:h_upper, w_lower:w_upper].flatten().numpy()

        cache['film'] = {
            'shape': (h_upper - h_lower, w_upper - w_lower, 3),
            'pixel_idx': pixel_idx
        }
    else:
        cache['film'] = {
            'shape': (h, w, 3),
            'pixel_idx': [-1]
        }

    psdr_scene.opts.width = w
    psdr_scene.opts.height = h

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

@PSDRJITConnector.register(PerspectiveCameraFull)
def process_perspective_camera_full(name, scene):
    sensor = scene[name]
    cache = scene.cached['psdr_jit']
    psdr_scene = cache['scene']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        psdr_sensor = psdr_jit.PerspectiveCamera(sensor['fx'], sensor['fy'], sensor['cx'], sensor['cy'], sensor['near'], sensor['far'])
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
            elif param_name == "fx":
                psdr_sensor.fx = sensor['fx']
            elif param_name == "fy":
                psdr_sensor.fy = sensor['fy']
            elif param_name == "cx":
                psdr_sensor.cx = sensor['cx']
            elif param_name == "cy":
                psdr_sensor.cy = sensor['cy']
            
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
        if 'mat_id' in mesh:
            mat_id = mesh['mat_id']
            if mat_id not in scene:
                raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
            brdf = scene[mat_id]
            PSDRJITConnector.extensions[type(brdf)](mat_id, scene)
        else:
            # Create a fake BSDF if the mesh is used as an emitter
            mat_id = f'{name}_null_bsdf'
            d = convert_color(to_torch_f([0, 0, 0]), 3)
            psdr_bsdf = psdr_jit.DiffuseBSDF(d)
            psdr_scene.add_BSDF(psdr_bsdf, mat_id)
            cache['name_map'][mat_id] = f"BSDF[id={mat_id}]"

        # TODO: Fix this workaround when psdr-jit updates psdr_mesh.load_raw()
        if mesh['can_change_topology']:
            psdr_mesh = psdr_jit.Mesh()
            psdr_mesh.load_raw(Vector3fC(mesh['v']), Vector3iC(mesh['f']))
            psdr_mesh.use_face_normal = mesh['use_face_normal']

            psdr_emitter = psdr_jit.AreaLight(mesh['radiance'].tolist()) if 'radiance' in mesh else None
            psdr_scene.add_Mesh(psdr_mesh, mat_id, psdr_emitter)
            psdr_scene.param_map[f"Mesh[{psdr_scene.num_meshes - 1}]"].set_transform(mesh['to_world'].reshape(1, 4, 4))
        else:
            write_mesh('__psdr_jit_tmp__.obj', mesh['v'], mesh['f'], mesh['uv'], mesh['fuv'])
            psdr_emitter = psdr_jit.AreaLight(mesh['radiance'].tolist()) if 'radiance' in mesh else None
            psdr_scene.add_Mesh('__psdr_jit_tmp__.obj', torch.eye(4).tolist(), mat_id, psdr_emitter)
            psdr_scene.param_map[f"Mesh[{psdr_scene.num_meshes - 1}]"].set_transform(mesh['to_world'].reshape(1, 4, 4))
            os.remove('__psdr_jit_tmp__.obj')
        
        cache['name_map'][name] = f"Mesh[{psdr_scene.num_meshes - 1}]"
        if psdr_emitter:
            cache['name_map'][name + ' emitter'] = f"Emitter[{psdr_scene.get_num_emitters() - 1}]"

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
            elif param_name == 'to_world':
                psdr_mesh.set_transform(Matrix4fD(mesh['to_world'].reshape(1, 4, 4)))
            elif param_name == 'radiance':
                psdr_emitter = psdr_scene.param_map[cache['name_map'][name + ' emitter']]
                psdr_emitter.radiance = mesh['radiance'].tolist()

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
            elif param_name == 'to_world':
                drjit_param = psdr_mesh.to_world_left
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

@PSDRJITConnector.register(PointLight)
def process_point_light(name, scene):
    pass

# Scene components specfic to psdr-jit
class MicrofacetBRDFPerVertex(ParamGroup):

    def __init__(self, d, s, r):
        super().__init__()
        
        self.add_param('d', to_torch_f(d), is_tensor=True, is_diff=True, help_msg='diffuse reflectance')
        self.add_param('s', to_torch_f(s), is_tensor=True, is_diff=True, help_msg='specular reflectance')
        self.add_param('r', to_torch_f(r), is_tensor=True, is_diff=True, help_msg='roughness')

@PSDRJITConnector.register(MicrofacetBRDFPerVertex)
def process_microfacet_brdf_per_vertex(name, scene):
    brdf = scene[name]
    cache = scene.cached['psdr_jit']
    psdr_scene = cache['scene']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        d = Vector3fD(brdf['d'])
        s = Vector3fD(brdf['s'])
        r = Vector1fD(brdf['r'])

        psdr_bsdf = psdr_jit.MicrofacetBSDFPerVertex(s, d, r)
        psdr_scene.add_BSDF(psdr_bsdf, name)
        cache['name_map'][name] = f"BSDF[id={name}]"

    psdr_brdf = psdr_scene.param_map[cache['name_map'][name]]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                psdr_brdf.diffuseReflectance = Vector3fD(brdf['d'])
            elif param_name == 's':
                psdr_brdf.specularReflectance = Vector3fD(brdf['s'])
            elif param_name == 'r':
                psdr_brdf.roughness= Vector1fD(brdf['r'])
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
                enable_grad(psdr_brdf.diffuseReflectance)
            elif param_name == 's':
                enable_grad(psdr_brdf.specularReflectance)
            elif param_name == 'r':
                enable_grad(psdr_brdf.roughness)

    return drjit_params
