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
            # use assign instead of psdr_cpu.Scene()
            # avoid calling destructor
            cache['scene'].assign(
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

    def forward_ad_mesh_translation(self, mesh_id, scene, render_options, sensor_ids=[0], integrator_id=0):
        psdr_cpu.set_forward(True)

        cache, _ = self.update_scene_objects(scene, render_options)

        h, w, c = cache['film']['shape']
        integrator = list(cache['integrators'].values())[integrator_id]

        psdr_scene = cache['scene']

        group, idx = cache['name_map'][mesh_id]
        psdr_mesh = cache['ctx'][group][idx]
        psdr_mesh.setTranslation(np.array([1., 0., 0.]))
        psdr_mesh.requires_grad = True

        images = []
        grad_images = []
        for i, sensor_id in enumerate(sensor_ids):
            psdr_scene.camera = cache['ctx']['cameras'][sensor_id]
            psdr_scene.camera.width = w
            psdr_scene.camera.height = h
            psdr_scene.configure()

            image = to_torch_f(integrator.renderC(psdr_scene, cache['render_options']))
            image = image.reshape(h, w, c)
            images.append(image)

            psdr_scene_ad = psdr_cpu.SceneAD(psdr_scene)
            boundary_integrator = psdr_cpu.BoundaryIntegrator(psdr_scene)

            grad_image = integrator.forwardRenderD(psdr_scene_ad, cache['render_options'])
            grad_image += boundary_integrator.forwardRenderD(psdr_scene_ad, cache['render_options'])
            grad_image = grad_image.reshape(h, w, c)
            grad_images.append(grad_image)

        psdr_cpu.set_forward(False)

        return images, grad_images

@PSDREnzymeConnector.register(Integrator)
def process_integrator(name, scene):
    integrator = scene[name]
    cache = scene.cached['psdr_enzyme']

    integrator_dict = {
        'path2': psdr_cpu.Path2,
        'pathwas': psdr_cpu.PathWAS,
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
        # temporary fix of the to_world matrix 
        # psdr-enzyme uses left-hand coordinate 
        to_world = to_numpy(camera['to_world'].clone())
        to_world[:3, 0] *= -1 
        props.set('to_world', to_world)
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
        if 'mat_id' in mesh:
            mat_id = mesh['mat_id']
            if mat_id not in scene:
                raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
            if mat_id not in cache['mat_id_map']:
                brdf = scene[mat_id]
                PSDREnzymeConnector.extensions[type(brdf)](mat_id, scene)
            props.set('bsdf_id', cache['mat_id_map'][mat_id])
        
        # Create the area light associated with the mesh if needed
        if 'radiance' in mesh:
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

def color_to_bitmap(color, c=3):
    if color.shape == ():
        color = color.tile(c)
    if color.shape == (c,):
        color = color.reshape(1, 1, c)
    h, w, _ = color.shape
    color = color.reshape(-1, 1)
    return psdr_cpu.Bitmap(to_numpy(color), to_numpy([w, h]))

def color_to_spectrum(color):
    if color.shape == ():
        color = color.tile(3)
    return psdr_cpu.Spectrum3f(color[0], color[1], color[2])

@PSDREnzymeConnector.register(DiffuseBRDF)
def process_diffuse_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['psdr_enzyme']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        bsdf_id = len(cache['ctx']['bsdfs'])
        d = color_to_bitmap(brdf['d'], 3)
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
                psdr_brdf.reflectance = color_to_bitmap(brdf['d'], 3)
            brdf.mark_updated(param_name, False)
        cache['update_scene'] = True
    
    # Creating strings for accessing parameters requiring grad
    param_name_map = {
        'd': 'reflectance'
    }
    requiring_grad = brdf.get_requiring_grad()
    params = [f'{group}[{idx}].{param_name_map[param_name]}' for param_name in requiring_grad]
    return params

@PSDREnzymeConnector.register(RoughDielectricBSDF)
def process_rough_dielectric_bsdf(name, scene):
    bsdf = scene[name]
    cache = scene.cached['psdr_enzyme']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        bsdf_id = len(cache['ctx']['bsdfs'])
        psdr_bsdf = psdr_cpu.RoughDielectricBSDF(bsdf['alpha'], bsdf['i_ior'], bsdf['e_ior'])
        cache['ctx']['bsdfs'].append(psdr_bsdf)
        cache['name_map'][name] = ("bsdfs", bsdf_id)
        cache['mat_id_map'][name] = bsdf_id

    return []

@PSDREnzymeConnector.register(RoughConductorBRDF)
def process_rough_conductor_bsdf(name, scene):
    bsdf = scene[name]
    cache = scene.cached['psdr_enzyme']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        bsdf_id = len(cache['ctx']['bsdfs'])
        psdr_bsdf = psdr_cpu.RoughConductorBSDF(bsdf['alpha_u'].item(), to_numpy(bsdf['eta']), to_numpy(bsdf['k']))
        cache['ctx']['bsdfs'].append(psdr_bsdf)
        cache['name_map'][name] = ("bsdfs", bsdf_id)
        cache['mat_id_map'][name] = bsdf_id

    return []

@PSDREnzymeConnector.register(EnvironmentLight)
def process_environment_light(name, scene):
    emitter = scene[name]
    cache = scene.cached['psdr_enzyme']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        emitter_id = len(cache['ctx']['emitters'])
        radiance = color_to_bitmap(emitter['radiance'], 3)
        props = Properties()
        props.setBitmap('data', radiance)
        props.set('toWorld', to_numpy(emitter['to_world']))
        psdr_emitter = psdr_cpu.EnvironmentMap(props)
        cache['ctx']['emitters'].append(psdr_emitter)
        cache['name_map'][name] = ("emitters", emitter_id)

    return []

