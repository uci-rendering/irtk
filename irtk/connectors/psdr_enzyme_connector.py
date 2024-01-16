from ..connector import Connector
from ..scene import *
from ..io import write_mesh, to_numpy, to_torch_f
from collections import OrderedDict

import numpy as np

import psdr_cpu
from psdr_cpu import Properties
psdr_cpu.set_verbose(False)

from irtk.utils import Timer

class PSDREnzymeConnector(Connector, connector_name='psdr_enzyme'):

    def __init__(self):
        super().__init__()

        self.default_render_options = {
            'seed': 0,
            'spp': 64,
            'sppe': 0,
            'sppse': 0,
            'max_bounces': 1, 
            'quiet': True
        }

    def update_scene_objects(self, scene, render_options):
        if 'psdr_enzyme' not in scene.cached:
            cache = {}
            scene.cached['psdr_enzyme'] = cache
            cache['name_map'] = {}
            cache['integrators'] = OrderedDict()
            cache['scene'] = psdr_cpu.Scene()
            cache['ctx'] = {
                'cameras': [],
                'shapes': [],
                'bsdfs': [],
                'emitters': [],
                'phases': [],
                'media': []
            }
            cache['mat_id_map'] = {}
            cache['render_options'] = psdr_cpu.RenderOptions()
            cache['update_scene'] = True
        
        params = []
        for name in scene.components:
            component = scene[name]
            params += self.extensions[type(component)](name, scene)

        cache = scene.cached['psdr_enzyme']
        if cache['update_scene']:
            cache['scene'] = psdr_cpu.Scene(
                cache['ctx']['cameras'][0],
                cache['ctx']['shapes'],
                cache['ctx']['bsdfs'],
                cache['ctx']['emitters'],
                cache['ctx']['phases'],
                cache['ctx']['media'],
            )
            cache['update_scene'] = False

        for key in self.default_render_options:
            if key not in render_options:
                render_options[key] = self.default_render_options[key]

        for key in render_options:
            setattr(cache['render_options'], key, render_options[key])

        return cache, params
       
    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        cache, _ = self.update_scene_objects(scene, render_options)
        
        h, w, c = cache['film']['shape']
        integrator = list(cache['integrators'].values())[integrator_id]

        psdr_scene = cache['scene']

        images = []
        for sensor_id in sensor_ids:
            psdr_scene.camera = cache['ctx']['cameras'][sensor_id]
            psdr_scene.camera.width = w
            psdr_scene.camera.height = h
            psdr_scene.configure()
            image = to_torch_f(integrator.renderC(psdr_scene, cache['render_options']))
            image = image.reshape(h, w, c)
            images.append(image)

        return images
        
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        cache, params = self.update_scene_objects(scene, render_options)

        h, w, c = cache['film']['shape']
        integrator = list(cache['integrators'].values())[integrator_id]

        psdr_scene = cache['scene']

        param_grads = [torch.zeros_like(scene[param_name]) for param_name in scene.requiring_grad]

        for i, sensor_id in enumerate(sensor_ids):
            psdr_scene.camera = cache['ctx']['cameras'][sensor_id]
            psdr_scene.camera.width = w
            psdr_scene.camera.height = h
            psdr_scene.configure()
            psdr_scene_ad = psdr_cpu.SceneAD(psdr_scene)
            boundary_integrator = psdr_cpu.BoundaryIntegrator(psdr_scene)

            image_grad = to_numpy(image_grads[i]).reshape(-1, 1)
            integrator.renderD(psdr_scene_ad, cache['render_options'], image_grad)
            boundary_integrator.renderD(psdr_scene_ad, cache['render_options'], image_grad)

            for param_grad, param in zip(param_grads, params):
                enzyme_grad = eval(f'psdr_scene_ad.der.{param}')
                if isinstance(enzyme_grad, psdr_cpu.Bitmap):
                    enzyme_grad = enzyme_grad.m_data
                grad = to_torch_f(np.array(enzyme_grad))
                param_grad += torch.nan_to_num(grad).reshape_as(param_grad)

        return param_grads


@PSDREnzymeConnector.register(Integrator)
def process_integrator(name, scene):
    integrator = scene[name]
    cache = scene.cached['psdr_enzyme']

    integrator_dict = {
        'path2': psdr_cpu.Path2,
    }
    if integrator['type'] in integrator_dict:
        cache['integrators'][name] = integrator_dict[integrator['type']]()
    else:
        raise ValueError(f"integrator type [{integrator['type']}] is not supported.")

    return []

@PSDREnzymeConnector.register(HDRFilm)
def process_hdr_film(name, scene):
    film = scene[name]
    cache = scene.cached['psdr_enzyme']

    cache['film'] = {
        'shape': (film['height'], film['width'], 3)
    }

    return []

@PSDREnzymeConnector.register(PerspectiveCamera)
def process_perspective_camera(name, scene):
    camera = scene[name]
    cache = scene.cached['psdr_enzyme']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        camera_id = len(cache['ctx']['cameras'])
        props = Properties()
        props.set('width', 0)
        props.set('height', 0)
        props.set('fov', float(camera['fov']))
        props.set('to_world', to_numpy(camera['to_world']))
        props.set('rfilter', {'type': 'box'})
        psdr_camera = psdr_cpu.Camera(props)
        cache['ctx']['cameras'].append(psdr_camera)
        cache['name_map'][name] = ("cameras", camera_id)

    return []

@PSDREnzymeConnector.register(Mesh)
def process_mesh(name, scene):
    mesh = scene[name]
    cache = scene.cached['psdr_enzyme']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        shape_id = len(cache['ctx']['shapes'])

        props = Properties()
        props.setVectorX('vertices', to_numpy(mesh['v']))
        props.setVectorX3i('indices', to_numpy(mesh['f']))
        props.setVectorX2('uvs', to_numpy(mesh['uv']))
        props.setVectorX3i('uv_indices', to_numpy(mesh['fuv']))
        props.set('to_world', to_numpy(mesh['to_world']))
        props.set('use_face_normals', to_numpy(mesh['use_face_normal']))

        # Create the associated material if it is not exist yet
        mat_id = mesh['mat_id']
        if mat_id not in scene:
            raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
        if mat_id not in cache['mat_id_map']:
            brdf = scene[mat_id]
            PSDREnzymeConnector.extensions[type(brdf)](mat_id, scene)
        props.set('bsdf_id', cache['mat_id_map'][mat_id])
        
        # Create the area light associated with the mesh if needed
        if mesh['is_emitter']:
            emitter_id = len(cache['ctx']['emitters'])
            radiance = to_numpy(mesh['radiance']).reshape(3, 1)
            psdr_emitter = psdr_cpu.AreaLight(shape_id, radiance)
            cache['ctx']['emitters'].append(psdr_emitter)
            cache['name_map'][name + '_emitter'] = ("emitters", emitter_id)
            props.set('light_id', emitter_id)

        psdr_mesh = psdr_cpu.Shape(props)
        cache['ctx']['shapes'].append(psdr_mesh)
        cache['name_map'][name] = ('shapes', shape_id)
    
    group, idx = cache['name_map'][name]
    psdr_mesh = cache['ctx'][group][idx]

    # Update parameters
    updated = mesh.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'v':
                psdr_mesh.vertices = to_numpy(mesh['v'])
            elif param_name == 'f':
                psdr_mesh.indices = to_numpy(mesh['f'])
            elif param_name == 'to_world':
                psdr_mesh.to_world = to_numpy(mesh['to_world'])
            mesh.mark_updated(param_name, False)
        psdr_mesh.configure()
        cache['update_scene'] = True
    
    # Creating strings for accessing parameters requiring grad
    param_name_map = {
        'v': 'vertices'
    }
    requiring_grad = mesh.get_requiring_grad()
    params = [f'{group}[{idx}].{param_name_map[param_name]}' for param_name in requiring_grad]
    return params

def convert_color(color, c=3):
    if color.shape == ():
        color = color.tile(c)
    if color.shape == (c,):
        color = color.reshape(1, 1, c)
    w, h, _ = color.shape
    color = color.reshape(-1, 1)
    return psdr_cpu.Bitmap(to_numpy(color), to_numpy([h, w]))

@PSDREnzymeConnector.register(DiffuseBRDF)
def process_diffuse_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['psdr_enzyme']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        bsdf_id = len(cache['ctx']['bsdfs'])
        d = convert_color(brdf['d'], 3)
        psdr_bsdf = psdr_cpu.DiffuseBSDF()
        psdr_bsdf.reflectance = d 
        cache['ctx']['bsdfs'].append(psdr_bsdf)
        cache['name_map'][name] = ("bsdfs", bsdf_id)
        cache['mat_id_map'][name] = bsdf_id

    group, idx = cache['name_map'][name]
    psdr_brdf = cache['ctx'][group][idx]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                psdr_brdf.reflectance = convert_color(brdf['d'], 3)
            brdf.mark_updated(param_name, False)
        cache['update_scene'] = True
    
    # Creating strings for accessing parameters requiring grad
    param_name_map = {
        'd': 'reflectance'
    }
    requiring_grad = brdf.get_requiring_grad()
    params = [f'{group}[{idx}].{param_name_map[param_name]}' for param_name in requiring_grad]
    return params

@PSDREnzymeConnector.register(EnvironmentLight)
def process_environment_light(name, scene):
    emitter = scene[name]
    cache = scene.cached['psdr_enzyme']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        emitter_id = len(cache['ctx']['emitters'])
        radiance = convert_color(emitter['radiance'], 3)
        props = Properties()
        props.setBitmap('data', radiance)
        props.set('toWorld', to_numpy(emitter['to_world']))
        psdr_emitter = psdr_cpu.EnvironmentMap(props)
        cache['ctx']['emitters'].append(psdr_emitter)
        cache['name_map'][name] = ("emitters", emitter_id)

    return []

# # Scene components specfic to psdr-jit
# class MicrofacetBRDFPerVertex(ParamGroup):

#     def __init__(self, d, s, r):
#         super().__init__()
        
#         self.add_param('d', to_torch_f(d), is_tensor=True, is_diff=True, help_msg='diffuse reflectance')
#         self.add_param('s', to_torch_f(s), is_tensor=True, is_diff=True, help_msg='specular reflectance')
#         self.add_param('r', to_torch_f(r), is_tensor=True, is_diff=True, help_msg='roughness')

# @PSDREnzymeConnector.register(MicrofacetBRDFPerVertex)
# def process_microfacet_brdf_per_vertex(name, scene):
#     brdf = scene[name]
#     cache = scene.cached['psdr_enzyme']
#     psdr_scene = cache['scene']
    
#     # Create the object if it has not been created
#     if name not in cache['name_map']:
#         d = Vector3fD(brdf['d'])
#         s = Vector3fD(brdf['s'])
#         r = Vector1fD(brdf['r'])

#         psdr_bsdf = psdr_enzyme.MicrofacetBSDFPerVertex(s, d, r)
#         psdr_scene.add_BSDF(psdr_bsdf, name)
#         cache['name_map'][name] = f"BSDF[id={name}]"

#     psdr_brdf = psdr_scene.param_map[cache['name_map'][name]]

#     # Update parameters
#     updated = brdf.get_updated()
#     if len(updated) > 0:
#         for param_name in updated:
#             if param_name == 'd':
#                 psdr_brdf.diffuseReflectance = Vector3fD(brdf['d'])
#             elif param_name == 's':
#                 psdr_brdf.specularReflectance = Vector3fD(brdf['s'])
#             elif param_name == 'r':
#                 psdr_brdf.roughness= Vector1fD(brdf['r'])
#             brdf.params[param_name]['updated'] = False

#     # Enable grad for parameters requiring grad
#     drjit_params = []
    
#     def enable_grad(drjit_param):
#         drjit.enable_grad(drjit_param)
#         drjit_params.append(drjit_param)

#     requiring_grad = brdf.get_requiring_grad()
#     if len(requiring_grad) > 0:
#         for param_name in requiring_grad:
#             if param_name == 'd':
#                 enable_grad(psdr_brdf.diffuseReflectance)
#             elif param_name == 's':
#                 enable_grad(psdr_brdf.specularReflectance)
#             elif param_name == 'r':
#                 enable_grad(psdr_brdf.roughness)

#     return drjit_params