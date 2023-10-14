from ..connector import Connector
from ..scene import *
from ..config import *
from ..io import write_mesh
from collections import OrderedDict

import drjit
from drjit.cuda import Array3f as Vector3fC, Array3i as Vector3iC
import mitsuba as mi
import torch

import time
import os

mi.set_variant('cuda_ad_rgb')

class MitsubaConnector(Connector, connector_name='mitsuba'):

    backend = 'torch'
    device = 'cuda'
    ftype = torch.float32
    itype = torch.long

    def __init__(self):
        super().__init__()
        self.render_time = 0

    def update_scene_objects(self, scene, render_options):
        if 'mitsuba' in scene.cached:
            cache = scene.cached['mitsuba']
        else:
            cache = {}
            scene.cached['mitsuba'] = cache

            cache['scene'] = {
                'type': 'scene',
            }
            cache['meshes'] = {}
            cache['textures'] = {}
            cache['cameras'] = OrderedDict()
            cache['integrators'] = OrderedDict()
            cache['envlights'] = {}
            cache['point_light'] = None
            cache['film'] = None
            cache['name_map'] = {}
        
        mitsuba_params = []
        for name in scene.components:
            component = scene[name]
            mitsuba_params += self.extensions[type(component)](name, scene)
        
        return scene.cached['mitsuba'], mitsuba_params

    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        mi.set_variant('cuda_ad_rgb')
        cache, _ = self.update_scene_objects(scene, render_options)

        # Mitsuba Scene
        mi_scene = cache['scene']
        mi_sensors = []
        # Sensors
        for sensor_name, sensor_value in cache['cameras'].items():
            sensor_value['film'] = cache['film']
            mi_sensors.append(mi.load_dict(sensor_value))
        # Integrators
        for integrator_name, integrator_value in cache['integrators'].items():
            mi_scene[integrator_name] = integrator_value
        # Meshs
        for mesh_name, mesh_value in cache['meshes'].items():
            mi_scene[mesh_name] = mesh_value
        # Environment Lights
        if len(cache['envlights']) >= 1:
            for envlight_name, envlight_value in cache['envlights'].items():
                mi_scene[envlight_name] = envlight_value
        # Point Light
        if cache['point_light']:
            mi_scene['emitter'] = cache['point_light']
        elif 'point_light_intensity' in render_options:
            mi_scene['emitter'] = {
                'type': 'point',
                'position': list(cache['cameras'].values())[0]['to_world'].translation(),
                'intensity': {
                    'type': 'rgb',
                    'value': render_options['point_light_intensity']
                }
            }
        elif 'plane_light_intensity' in render_options:
            mi_scene['emitter'] = {
                # 'type': 'obj',
                # 'filename': 'assets/meshes/plane_light.obj',
                'type': 'rectangle',
                # 'to_world': mi.ScalarTransform4f([
                #     [100, 0, 0, 0],
                #     [0, 0, -100, 0],
                #     [0, 100, 0, -300],
                #     [0, 0, 0, 1]
                # ]),
                'to_world': mi.ScalarTransform4f([
                    [50, 0, 0, 0],
                    [0, 50, 0, 0],
                    [0, 0, 50, -300],
                    [0, 0, 0, 1]
                ]),
                'focused-emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'rgb',
                        'value': render_options['plane_light_intensity']
                    }
                },
            }
        # print(mi_scene)
        loaded_mi_scene = mi.load_dict(mi_scene, False)
        # print(loaded_mi_scene)
        params = mi.traverse(loaded_mi_scene)
   
        images = []
        npass = render_options['npass']
        h, w, c = (cache['film']['height'], cache['film']['width'], 3)
        for sensor_id in sensor_ids:
            # change light pos according to sensor
            if 'point_light_intensity' in render_options:
                params['emitter.position'] = list(cache['cameras'].values())[sensor_id]['to_world'].translation()
                params.update()
            
            image = torch.zeros((h, w, c)).to(device).to(ftype)
            for i in range(npass):
                t = time.time()
                image_pass = mi.render(loaded_mi_scene, sensor=mi_sensors[sensor_id], spp=render_options['spp']).torch()
                self.render_time += time.time() - t
                image += image_pass / npass
            images.append(image)

        return images
        
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        with torch.enable_grad():
            mi.set_variant('cuda_ad_rgb')
            cache, mitsuba_params = self.update_scene_objects(scene, render_options)

            # Mitsuba Scene
            # mi_scene = cache['scene']
            mi_scene = {
                'type': 'scene',
            }
            mi_sensors = []
            # Sensors
            for sensor_name, sensor_value in cache['cameras'].items():
                sensor_value['film'] = cache['film']
                mi_sensors.append(mi.load_dict(sensor_value))
            # Integrators
            for integrator_name, integrator_value in cache['integrators'].items():
                mi_scene[integrator_name] = integrator_value
            # Meshs
            for mesh_name, mesh_value in cache['meshes'].items():
                mi_scene[mesh_name] = mesh_value
            # Environment Lights
            if len(cache['envlights']) >= 1:
                for envlight_name, envlight_value in cache['envlights'].items():
                    mi_scene[envlight_name] = envlight_value
            # Point Light
            if cache['point_light']:
                mi_scene['emitter'] = cache['point_light']
            elif 'point_light_intensity' in render_options:
                mi_scene['emitter'] = {
                    'type': 'point',
                    'position': list(cache['cameras'].values())[0]['to_world'].translation(),
                    'intensity': {
                        'type': 'rgb',
                        'value': render_options['point_light_intensity']
                    }
                }
            elif 'plane_light_intensity' in render_options:
                mi_scene['emitter'] = {
                    # 'type': 'obj',
                    # 'filename': 'assets/meshes/plane_light.obj',
                    'type': 'rectangle',
                    # 'to_world': mi.ScalarTransform4f([
                    #     [100, 0, 0, 0],
                    #     [0, 0, -100, 0],
                    #     [0, 100, 0, -300],
                    #     [0, 0, 0, 1]
                    # ]),
                    'to_world': mi.ScalarTransform4f([
                        [50, 0, 0, 0],
                        [0, 50, 0, 0],
                        [0, 0, 50, -300],
                        [0, 0, 0, 1]
                    ]),
                    'focused-emitter': {
                        'type': 'area',
                        'radiance': {
                            'type': 'rgb',
                            'value': render_options['plane_light_intensity']
                        }
                    },
                }
            # print(mi_scene)
            loaded_mi_scene = mi.load_dict(mi_scene, False)
            # print(loaded_mi_scene)
            params = mi.traverse(loaded_mi_scene)

            npass = render_options['npass']
            param_grads = [torch.zeros_like(scene[param_name]) for param_name in scene.requiring_grad]

            for i, sensor_id in enumerate(sensor_ids):
                # change light pos according to sensor
                if 'point_light_intensity' in render_options:
                    params['emitter.position'] = list(cache['cameras'].values())[sensor_id]['to_world'].translation()
                    params.update()
                
                image_grad = mi.TensorXf(image_grads[i] / npass)
                for j in range(npass):
                    t = time.time()
                    
                    image_pass = mi.render(loaded_mi_scene, params, sensor=mi_sensors[sensor_id], spp=render_options['spp'])
                    tmp = image_grad * image_pass
                    drjit.backward(tmp)
                    for param_grad, mitsuba_param in zip(param_grads, mitsuba_params):
                        if type(mitsuba_param) == dict:
                            backend_grad = drjit.grad(mitsuba_param['backend']).torch().to(device).to(ftype)
                            backend_grad = torch.nan_to_num(backend_grad).reshape(mitsuba_param['frontend'].shape)
                            tmp = backend_grad * mitsuba_param['frontend']
                            frontend_grad = torch.autograd.grad(tmp, scene[mitsuba_param['param']], torch.ones_like(tmp), retain_graph=True)
                            param_grad += frontend_grad[0]
                        else:
                            grad = drjit.grad(mitsuba_param).torch().to(device).to(ftype)
                            grad = torch.nan_to_num(grad).reshape(param_grad.shape)
                            param_grad += grad
                    
                    self.render_time += time.time() - t

            return param_grads

@MitsubaConnector.register(Integrator)
def process_integrator(name, scene):
    integrator = scene[name]
    cache = scene.cached['mitsuba']

    # TODO: support more integrator types
    mi_integrator = {
        'type': integrator['type']
    }
    if integrator['type'] == 'direct':
        cache['integrators'][name] = mi_integrator
    elif integrator['type'] == 'path':
        mi_integrator['max_depth'] = integrator['config']['max_depth']
        if 'hide_emitters' in integrator['config']:
            mi_integrator['hide_emitters'] = integrator['config']['hide_emitters']
        cache['integrators'][name] = mi_integrator
    else:
        raise RuntimeError(f"unrecognized integrator type: {integrator['type']}")

    return []

@MitsubaConnector.register(HDRFilm)
def process_hdr_film(name, scene):
    film = scene[name]
    cache = scene.cached['mitsuba']

    cache['film'] = {
        'type': 'hdrfilm',
        'width': film['width'],
        'height': film['height']
    }

    return []

@MitsubaConnector.register(PerspectiveCamera)
def process_perspective_camera(name, scene):
    sensor = scene[name]
    cache = scene.cached['mitsuba']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        mi_sensor = {
            'type': 'perspective',
            'near_clip': sensor['near'],
            'far_clip': sensor['far'],
            'fov': sensor['fov'],
            'to_world': mi.ScalarTransform4f(sensor['to_world'].cpu().numpy())
        }
        cache['cameras'][name] = mi_sensor
        cache['name_map'][name] = mi_sensor

    mi_sensor = cache['name_map'][name]
    
    # Update parameters
    updated = sensor.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == "to_world":
                mi_sensor['to_world'] = mi.ScalarTransform4f(sensor['to_world'].cpu().numpy())
            sensor.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    mitsuba_params = []
    requiring_grad = sensor.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == "to_world":
                mi_sensor['to_world'].requires_grad_()
                mitsuba_params.append(mi_sensor['to_world'])

    return mitsuba_params

@MitsubaConnector.register(Mesh)
def process_mesh(name, scene):
    mesh = scene[name]
    cache = scene.cached['mitsuba']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        # Create its material first
        mat_id = mesh['mat_id']
        if mat_id not in scene:
            raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
        brdf = scene[mat_id]
        MitsubaConnector.extensions[type(brdf)](mat_id, scene)

        verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(device)), dim=1)
        verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
        if mesh['uv'].nelement() == 0:
            mesh['uv'] = torch.zeros((1, 2)).to(device)
        if mesh['fuv'].nelement() == 0:
            verts_new = verts
            faces_new = mesh['f'].long()
            uvs_new = torch.zeros(verts.shape[0], 2).to(device)
        else:
            mesh['fuv'] = torch.zeros_like(mesh['f']).to(device)
            verts_new, faces_new, uvs_new = compute_texture_coordinates(verts, mesh['f'].long(), mesh['uv'], mesh['fuv'].long())

        # Set bsdf and emitter properties
        props = mi.Properties()
        props['bsdf'] = mi.load_dict(cache['textures'][mat_id])
        # if mesh['is_emitter']:
        #     props['emitter'] = mi.load_dict({
        #         'type': 'area',
        #         'radiance': {
        #             'type': 'rgb',
        #             'value': 20
        #         }
        #     })
            
        # Create mitsuba mesh
        mi_mesh = mi.Mesh(name, len(verts_new), len(faces_new), props=props, has_vertex_normals=True, has_vertex_texcoords=True)
        params = mi.traverse(mi_mesh)
        params['vertex_count'] = len(verts_new)
        params['face_count'] = len(faces_new)
        params['faces'] = drjit.ravel(mi.Vector3u(faces_new.cpu().numpy()))
        params['vertex_positions'] = drjit.ravel(mi.Vector3f(verts_new))
        # Compute vertex normals
        vts_normals = compute_vertex_normals(verts_new, faces_new.long())
        params['vertex_normals'] = drjit.ravel(mi.Vector3f(vts_normals))
        # Set vertex texture coordinates
        params['vertex_texcoords'] = drjit.ravel(mi.Vector2f(uvs_new))
        params.update()

        cache['meshes'][name] = mi_mesh
        cache['name_map'][name] = mi_mesh

    mi_mesh = cache['name_map'][name]

    # Update parameters
    updated = mesh.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'v' or param_name == 'to_world':
                verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(device)), dim=1)
                verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
                if mesh['uv'].nelement() == 0:
                    mesh['uv'] = torch.zeros((1, 2)).to(device)
                if mesh['fuv'].nelement() == 0:
                    verts_new = verts
                    faces_new = mesh['f'].long()
                    uvs_new = torch.zeros(verts.shape[0], 2).to(device)
                else:
                    verts_new, faces_new, uvs_new = compute_texture_coordinates(verts, mesh['f'].long(), mesh['uv'], mesh['fuv'].long())
                params = mi.traverse(mi_mesh)
                params['vertex_count'] = len(verts_new)
                params['face_count'] = len(faces_new)
                params['faces'] = drjit.ravel(mi.Vector3u(faces_new.cpu().numpy()))
                params['vertex_positions'] = drjit.ravel(mi.Vector3f(verts_new))
                # Compute vertex normals
                vts_normals = compute_vertex_normals(verts_new, faces_new.long())
                params['vertex_normals'] = drjit.ravel(mi.Vector3f(vts_normals))
                # Set vertex texture coordinates
                params['vertex_texcoords'] = drjit.ravel(mi.Vector2f(uvs_new))
                params.update()
                
            mesh.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    mitsuba_params = []
    requiring_grad = mesh.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'v':
                if mesh['fuv'].nelement() == 0 and torch.equal(mesh['to_world'], to_torch_f(torch.eye(4))):
                    params = mi.traverse(mi_mesh)
                    drjit.enable_grad(params['vertex_positions'])
                    mitsuba_params.append(params['vertex_positions'])
                    params.update()
                
                else:
                    # frontend
                    verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(device)), dim=1)
                    verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
                    
                    if mesh['fuv'].nelement() != 0:
                        verts_new, faces_new, uvs_new = compute_texture_coordinates(verts, mesh['f'].long(), mesh['uv'], mesh['fuv'].long())
                    else:
                        verts_new = verts
                    
                    # backend
                    params = mi.traverse(mi_mesh)
                    drjit.enable_grad(params['vertex_positions'])
                    
                    mitsuba_params.append({
                        'param': f'{name}.{param_name}',
                        'frontend': verts_new,
                        'backend': params['vertex_positions']
                    })
                    params.update()

    return mitsuba_params

@MitsubaConnector.register(DiffuseBRDF)
def process_diffuse_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['mitsuba']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        mi_brdf = {
            'type': 'diffuse',
            'reflectance': {}
        }
        kd = brdf['d']
        if kd.dim() == 4:
            kd = kd[0]
        if len(kd) <= 3:
            mi_brdf['reflectance']['type'] = 'rgb'
            mi_brdf['reflectance']['value'] = kd.cpu().numpy()
        else:
            # Flip texture
            kd = torch.flip(kd, [0])
            mi_brdf['reflectance']['type'] = 'bitmap'
            mi_brdf['reflectance']['bitmap'] = mi.Bitmap(kd.cpu().numpy())
        cache['textures'][name] = mi_brdf
        cache['name_map'][name] = mi_brdf

    mi_brdf = cache['name_map'][name]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                kd = brdf['d']
                if kd.dim() == 4:
                    kd = kd[0]
                if len(kd) <= 3:
                    mi_brdf['reflectance']['value'] = kd.cpu().numpy()
                else:
                    # Flip texture
                    kd = torch.flip(kd, [0])
                    mi_brdf['reflectance']['bitmap'] = mi.Bitmap(kd.cpu().numpy())
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    mitsuba_params = []
    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'd':
                if mi_brdf['reflectance']['type'] == 'rgb':
                    mi_brdf['reflectance']['value'].requires_grad = True
                    mitsuba_params.append(mi_brdf['reflectance']['value'])
                else:
                    mi_brdf['reflectance']['bitmap'].requires_grad = True
                    mitsuba_params.append(mi_brdf['reflectance']['bitmap'])

    return mitsuba_params

@MitsubaConnector.register(MicrofacetBRDF)
def process_microfacet_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['mitsuba']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        # TODO: support more BRDF types
        mi_brdf = {
            'type': 'roughplastic',
            'distribution': 'ggx',
            'diffuse_reflectance': {},
            'specular_reflectance': {},
            'alpha': 0.1
        }
        kd = brdf['d']
        if kd.dim() == 4:
            kd = kd[0]
        if len(kd) <= 3:
            mi_brdf['diffuse_reflectance']['type'] = 'rgb'
            mi_brdf['diffuse_reflectance']['value'] = kd.cpu().numpy()
        else:
            mi_brdf['diffuse_reflectance']['type'] = 'bitmap'
            mi_brdf['diffuse_reflectance']['bitmap'] = mi.Bitmap(kd.cpu().numpy())
        ks = brdf['s']
        if ks.dim() == 4:
            ks = ks[0]
        if ks.dim() == 0:
            mi_brdf['specular_reflectance']['type'] = 'rgb'
            mi_brdf['specular_reflectance']['value'] = ks.item()
        else:
            mi_brdf['specular_reflectance']['type'] = 'bitmap'
            mi_brdf['specular_reflectance']['bitmap'] = mi.Bitmap(ks.cpu().numpy())
        kr = brdf['r']
        if kr.dim() == 0:
            mi_brdf['alpha'] = float(kr.cpu().numpy())

        cache['textures'][name] = mi_brdf
        cache['name_map'][name] = mi_brdf

    mi_brdf = cache['name_map'][name]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                kd = brdf['d']
                if kd.dim() == 4:
                    kd = kd[0]
                if len(kd) <= 3:
                    mi_brdf['diffuse_reflectance']['value'] = kd.cpu().numpy()
                else:
                    mi_brdf['diffuse_reflectance']['bitmap'] = mi.Bitmap(kd.cpu().numpy())
            elif param_name == 's':
                ks = brdf['s']
                if ks.dim() == 4:
                    ks = ks[0]
                if len(ks) <= 3:
                    mi_brdf['specular_reflectance']['value'] = ks.cpu().numpy()
                else:
                    mi_brdf['specular_reflectance']['bitmap'] = mi.Bitmap(ks.cpu().numpy())
            elif param_name == 'r':
                kr = brdf['r']
                if kr.dim() == 0:
                    mi_brdf['alpha'] = float(kr.cpu().numpy())
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    mitsuba_params = []
    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'd':
                if mi_brdf['diffuse_reflectance']['type'] == 'rgb':
                    mi_brdf['diffuse_reflectance']['value'].requires_grad = True
                    mitsuba_params.append(mi_brdf['diffuse_reflectance']['value'])
                else:
                    mi_brdf['diffuse_reflectance']['bitmap'].requires_grad = True
                    mitsuba_params.append(mi_brdf['diffuse_reflectance']['bitmap'])
            elif param_name == 's':
                if mi_brdf['specular_reflectance']['type'] == 'rgb':
                    mi_brdf['specular_reflectance']['value'].requires_grad = True
                    mitsuba_params.append(mi_brdf['specular_reflectance']['value'])
                else:
                    mi_brdf['specular_reflectance']['bitmap'].requires_grad = True
                    mitsuba_params.append(mi_brdf['specular_reflectance']['bitmap'])
            elif param_name == 'r':
                mi_brdf['alpha'].requires_grad = True
                mitsuba_params.append(mi_brdf['alpha'])

    return mitsuba_params

@MitsubaConnector.register(EnvironmentLight)
def process_environment_light(name, scene):
    emitter = scene[name]
    cache = scene.cached['mitsuba']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        mi_emitter = {}
        radiance = emitter['radiance']
        if radiance.dim() == 4:
            radiance = radiance[0]
        if len(radiance) <= 3:
            mi_emitter['type'] = 'constant'
            mi_emitter['radiance'] = {}
            mi_emitter['radiance']['type'] = 'rgb'
            mi_emitter['radiance']['value'] = radiance.cpu().numpy()
        else:
            mi_emitter['type'] = 'envmap'
            mi_emitter['bitmap'] = mi.Bitmap(radiance.cpu().numpy())
            mi_emitter['to_world'] = mi.ScalarTransform4f(emitter['to_world'].cpu().numpy())

        cache['envlights'][name] = mi_emitter
        cache['name_map'][name] = mi_emitter

    mi_emitter = cache['name_map'][name]

    # Update parameters
    updated = emitter.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'radiance':
                radiance = emitter['radiance']
                if radiance.dim() == 4:
                    radiance = radiance[0]
                if len(radiance) <= 3:
                    mi_emitter['radiance']['value'] = radiance.cpu().numpy()
                else:
                    mi_emitter['bitmap'] = mi.Bitmap(radiance.cpu().numpy())
            elif param_name == 'to_world':
                mi_emitter['to_world'] = mi.ScalarTransform4f(emitter['to_world'].cpu().numpy())
            emitter.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    mitsuba_params = []
    requiring_grad = emitter.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'radiance':
                if mi_emitter['radiance']['type'] == 'rgb':
                    mi_emitter['radiance']['value'].requires_grad = True
                    mitsuba_params.append(mi_emitter['radiance']['value'])
                else:
                    mi_emitter['bitmap'].requires_grad = True
                    mitsuba_params.append(mi_emitter['bitmap'])

    return mitsuba_params

@MitsubaConnector.register(PointLight)
def process_point_light(name, scene):
    light = scene[name]
    cache = scene.cached['mitsuba']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        mi_point_light = {
            'type': 'point',
            'position': mi.ScalarVector3f(light['position'].cpu().numpy()),
            'intensity': {
                'type': 'rgb',
                'value': light['radiance'].item()
            }
        }
        
        cache['point_light'] = mi_point_light
        cache['name_map'][name] = mi_point_light

    mi_point_light = cache['name_map'][name]
    
    # Update parameters
    updated = light.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == "position":
                mi_point_light['position'] = mi.ScalarVector3f(light['position'].cpu().numpy())
            light.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    mitsuba_params = []
    requiring_grad = light.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == "position":
                drjit.enable_grad(mi_point_light['position'])
                mitsuba_params.append(mi_point_light['position'])
                
    return mitsuba_params

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

def collect(indices, attributes, mapping=False):
    i0 = indices[:, 0]
    i1 = indices[:, 1]
    i2 = indices[:, 2]
    a0 = attributes[i0, :]
    a1 = attributes[i1, :]
    a2 = attributes[i2, :]
    if mapping:
        return torch.cat([a0, a1, a2], axis=0), torch.cat([i0, i1, i2], axis=0)
    else:
        return torch.cat([a0, a1, a2], axis=0)

def compute_texture_coordinates(verts, faces, uv, fuv):
    # Cut faces
    verts_new, mapping = collect(faces, verts, mapping=True)
    uvs_new = collect(fuv, uv)
    # Calculate the corresponding indices
    n_indices = verts_new.shape[0] / 3
    i0 = torch.arange(n_indices, dtype=int)
    i1 = torch.arange(n_indices, dtype=int) + n_indices
    i2 = torch.arange(n_indices, dtype=int) + n_indices * 2
    i0 = i0.unsqueeze(1)
    i1 = i1.unsqueeze(1)
    i2 = i2.unsqueeze(1)
    faces_new = torch.cat([i0, i1, i2], axis=1).to(device)

    return verts_new, faces_new, uvs_new
