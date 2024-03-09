from ..connector import Connector
from ..scene import *
from ..config import *
from ..io import write_mesh
from collections import OrderedDict
from ..utils import Timer

import drjit
from drjit.cuda import Array3f as Vector3fC, Array3i as Vector3iC
import mitsuba as mi
import torch

import time
import os

mi.set_variant('cuda_ad_rgb')
mi.register_bsdf("MitsubaMicrofacetBSDF", lambda props: MitsubaMicrofacetBSDF(props))

class MitsubaConnector(Connector, connector_name='mitsuba'):

    debug = False

    def __init__(self):
        super().__init__()
        
        self.default_render_options = {
            'spp': 64,
            'npass': 1,
            'seed': 0
        }

    # """
    def update_scene_objects(self, scene, render_options):
        if 'mitsuba' in scene.cached:
            cache = scene.cached['mitsuba']
        else:
            cache = {}
            scene.cached['mitsuba'] = cache

            cache['meshes'] = {}
            cache['textures'] = {}
            cache['cameras'] = OrderedDict()
            cache['integrators'] = OrderedDict()
            cache['envlights'] = {}
            cache['point_light'] = None
            cache['film'] = None
            cache['name_map'] = {}
            
        for k in self.default_render_options:
            if k not in render_options:
                render_options[k] = self.default_render_options[k]
        
        mitsuba_params = []
        for name in scene.components:
            component = scene[name]
            mitsuba_params += self.extensions[type(component)](name, scene)
        
        # Mitsuba Scene
        mi_scene = {
            'type': 'scene',
            # 'rfilter': {
            #     'type': 'box',
            # }
        }
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
            # mi_scene['emitter'] = {
            #     'type': 'point',
            #     'position': list(cache['cameras'].values())[0]['to_world'].translation(),
            #     'intensity': {
            #         'type': 'rgb',
            #         'value': render_options['point_light_intensity']
            #     }
            # }
            mi_scene['emitter'] = {
                'type': 'obj',
                'filename': 'assets/meshes/plane_point_light.obj',
                'to_world': list(cache['cameras'].values())[0]['to_world'],
                'area_light': {
                    'type': 'area',
                    'radiance': {
                        'type': 'rgb',
                        'value': render_options['point_light_intensity']
                    }
                }
            }
            # mi_scene['emitter'] = {
            #     'type': 'constant',
            #     'radiance': 0.99,
            # }
        elif 'plane_light' in render_options:
            mi_scene['emitter'] = {
                # 'type': 'obj',
                # 'filename': 'assets/meshes/plane_4f.obj',
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f(render_options['plane_light']['to_world']),
                'focused-emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'rgb',
                        'value': render_options['plane_light']['radiance']
                    }
                },
            }
        # print(mi_scene)
        # print("loading scene dict...", end='')

        # print(loaded_mi_scene)
        cache['scene'] = mi_scene
            
        return scene.cached['mitsuba'], mitsuba_params
    # """
    
    """
    def update_scene_objects(self, scene, render_options):
        if 'mitsuba' in scene.cached:
            cache = scene.cached['mitsuba']
        else:
            cache = {}
            scene.cached['mitsuba'] = cache

            cache['meshes'] = {}
            cache['textures'] = {}
            cache['cameras'] = OrderedDict()
            cache['integrators'] = OrderedDict()
            cache['envlights'] = {}
            cache['point_light'] = None
            cache['film'] = None
            cache['name_map'] = {}
        
        from mitsuba import ScalarTransform4f as T
            
        sensor = {
            'type': 'perspective',
            'fov': 45,
            'to_world': T.look_at(target=[0, 0, 0], origin=[2.4206145913796355, 0.0, 4.375], up=[0, 1, 0]),
            'film': {
                'type': 'hdrfilm',
                'width': 256, 'height': 256,
                'filter': {'type': 'gaussian'},
                'sample_border': True
            }
        }
        scene_dict = {
            'type': 'scene',
            'integrator': {
                'type': 'direct_reparam',
            },
            'sensor': sensor,
            'emitter': {
                'type': 'envmap',
                'filename': "../scenes/textures/envmap2.exr",
            },
            'shape': {
                'type': 'ply',
                'filename': "../scenes/meshes/suzanne.ply",
                'bsdf': {'type': 'diffuse'}
            }
        }

        scene_target = mi.load_dict(scene_dict)
        cache['scene'] = scene_target
        
        mitsuba_params = []

        return scene.cached['mitsuba'], mitsuba_params
    """

    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        mi.set_variant('cuda_ad_rgb')
        
        with Timer(f"-- Prepare Scene", prt=self.debug, record=False):
            cache, _ = self.update_scene_objects(scene, render_options)

            mi_scene = cache['scene']
            loaded_mi_scene = mi.load_dict(mi_scene)
            params = mi.traverse(loaded_mi_scene)

            # Sensors
            mi_sensors = []
            for sensor_name, sensor_value in cache['cameras'].items():
                sensor_value['film'] = cache['film']
                sensor_value['sampler'] = {
                    'type': 'independent',
                    'sample_count': render_options['spp']
                }
                mi_sensors.append(mi.load_dict(sensor_value))

        with Timer('-- Backend Forward', prt=self.debug, record=False):
            images = []
            npass = render_options['npass']
            h, w, c = (cache['film']['height'], cache['film']['width'], 3)
            # h, w, c = (256, 256, 3)
            for sensor_id in sensor_ids:
                seed = render_options['seed']
                
                # change light pos according to sensor
                if 'point_light_intensity' in render_options:
                    # params['emitter.position'] = list(cache['cameras'].values())[sensor_id]['to_world'].translation()
                    # params.update()
                    # mi_scene['emitter']['to_world'] = list(cache['cameras'].values())[sensor_id]['to_world']
                    emitter_to_world = list(cache['cameras'].values())[sensor_id]['to_world']
                    emitter_offset = emitter_to_world.translation().numpy() * 0.01
                    mi_scene['emitter']['to_world'] = mi.ScalarTransform4f.translate(emitter_offset) @ emitter_to_world
                    loaded_mi_scene = mi.load_dict(mi_scene)
                
                image = torch.zeros((h, w, c)).to(configs['device']).to(configs['ftype'])
                for i in range(npass):
                    # image_pass = mi.render(loaded_mi_scene, sensor=mi_sensors[sensor_id], spp=render_options['spp'], seed=seed).torch()
                    image_pass = mi.render(loaded_mi_scene, params, sensor=mi_sensors[sensor_id], seed=seed).torch()
                    # image_pass = mi.render(loaded_mi_scene, spp=render_options['spp']).torch()
                    image += image_pass / npass
                    seed += 1
                images.append(image)

        # params = mi.traverse(loaded_mi_scene)
        # drjit.enable_grad(params['shape.vertex_positions'])
        # mitsuba_params = [params['shape.vertex_positions']]
        # # print(params)
        # npass = 1
        # image_grad = mi.TensorXf(torch.ones(256, 256, 3).to(configs['device']) / npass)
        # for j in range(npass):
            
        #     # image_pass = mi.render(loaded_mi_scene, params, sensor=mi_sensors[sensor_id], spp=render_options['spp'])
        #     image_pass = mi.render(loaded_mi_scene, params, spp=128)
        #     tmp = image_grad * image_pass
        #     drjit.backward(tmp)
        #     for mitsuba_param in mitsuba_params:
        #         grad = drjit.grad(mitsuba_param).torch().to(configs['device']).to(configs['ftype'])
        #         grad = torch.nan_to_num(grad)

        return images
        
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        with torch.enable_grad():
            mi.set_variant('cuda_ad_rgb')
            
            with Timer(f"-- Prepare Scene", prt=self.debug, record=False):
                cache, mitsuba_params = self.update_scene_objects(scene, render_options)
            
                # """
                # Sensors
                mi_sensors = []
                for sensor_name, sensor_value in cache['cameras'].items():
                    sensor_value['film'] = cache['film']
                    sensor_value['sampler'] = {
                        'type': 'independent',
                        'sample_count': render_options['spp']
                    }
                    mi_sensors.append(mi.load_dict(sensor_value))
                
                mi_scene = cache['scene']
                loaded_mi_scene = mi.load_dict(mi_scene)
                params = mi.traverse(loaded_mi_scene)

                npass = render_options['npass']
                param_grads = [torch.zeros_like(scene[param_name]) for param_name in scene.requiring_grad]

            with Timer('-- Backend Backward', prt=self.debug, record=False):
                for i, sensor_id in enumerate(sensor_ids):
                    seed = render_options['seed']
                    
                    # change light pos according to sensor
                    if 'point_light_intensity' in render_options:
                        # params['emitter.position'] = list(cache['cameras'].values())[sensor_id]['to_world'].translation()
                        # params.update()
                        emitter_to_world = list(cache['cameras'].values())[sensor_id]['to_world']
                        emitter_offset = emitter_to_world.translation().numpy() * 0.01
                        mi_scene['emitter']['to_world'] = mi.ScalarTransform4f.translate(emitter_offset) @ emitter_to_world
                        loaded_mi_scene = mi.load_dict(mi_scene)
                        params = mi.traverse(loaded_mi_scene)
                    
                    image_grad = mi.TensorXf(image_grads[i] / npass)
                    for j in range(npass):
                        # image_pass = mi.render(loaded_mi_scene, params, sensor=mi_sensors[sensor_id], spp=render_options['spp'], seed=seed)
                        image_pass = mi.render(loaded_mi_scene, params, sensor=mi_sensors[sensor_id], seed=seed)
                        tmp = image_grad * image_pass
                        drjit.backward(tmp)
                        for param_grad, mitsuba_param in zip(param_grads, mitsuba_params):
                            if type(mitsuba_param) == dict:
                                backend_grad = drjit.grad(mitsuba_param['backend']).torch().to(configs['device']).to(configs['ftype'])
                                backend_grad = torch.nan_to_num(backend_grad).reshape(mitsuba_param['frontend'].shape)
                                tmp = backend_grad * mitsuba_param['frontend']
                                frontend_grad = torch.autograd.grad(tmp, scene[mitsuba_param['param']], torch.ones_like(tmp), retain_graph=True)
                                param_grad += frontend_grad[0]
                            else:
                                grad = drjit.grad(mitsuba_param).torch().to(configs['device']).to(configs['ftype'])
                                grad = torch.nan_to_num(grad).reshape(param_grad.shape)
                                param_grad += grad
                        
                        drjit.flush_malloc_cache()
                        seed += 1
                
                torch.cuda.synchronize()
                        
            # """
            
            """
            loaded_mi_scene = cache['scene']
            params = mi.traverse(loaded_mi_scene)
            drjit.enable_grad(params['shape.vertex_positions'])
            mitsuba_params = [params['shape.vertex_positions']]
            # print(params)
            npass = 1
            image_grad = mi.TensorXf(torch.ones(256, 256, 3).to(configs['device']) / npass)
            for j in range(npass):
                
                # image_pass = mi.render(loaded_mi_scene, params, sensor=mi_sensors[sensor_id], spp=render_options['spp'])
                image_pass = mi.render(loaded_mi_scene, params, spp=128)
                tmp = image_pass
                drjit.backward(tmp)
                for mitsuba_param in mitsuba_params:
                    grad = drjit.grad(mitsuba_param).torch().to(configs['device']).to(configs['ftype'])
                    grad = torch.nan_to_num(grad)
            return grad
            """
            return param_grads

    def forward_ad_mesh_translation(self, mesh_id, scene, render_options, sensor_ids=[0], integrator_id=0):
        mi.set_variant('cuda_ad_rgb')
        cache, mitsuba_params = self.update_scene_objects(scene, render_options)
        
        assert len(mitsuba_params) == 0
        assert len(sensor_ids) == 1
        
        mi_scene = cache['scene']
        loaded_mi_scene = mi.load_dict(mi_scene)
        params = mi.traverse(loaded_mi_scene)
        
        # Sensors
        mi_sensors = []
        for sensor_name, sensor_value in cache['cameras'].items():
            sensor_value['film'] = cache['film']
            sensor_value['sampler'] = {
                'type': 'independent',
                'sample_count': render_options['spp']
            }
            mi_sensors.append(mi.load_dict(sensor_value))

        seed = render_options['seed']
        
        with Timer('Forward AD'):
            npass = render_options['npass']
            h, w, c = (cache['film']['height'], cache['film']['width'], 3)
            # h, w, c = (256, 256, 3)
            sensor_id = sensor_ids[0]
            # change light pos according to sensor
            if 'point_light_intensity' in render_options:
                # params['emitter.position'] = list(cache['cameras'].values())[sensor_id]['to_world'].translation()
                # params.update()
                # mi_scene['emitter']['to_world'] = list(cache['cameras'].values())[sensor_id]['to_world']
                emitter_to_world = list(cache['cameras'].values())[sensor_id]['to_world']
                emitter_offset = emitter_to_world.translation().numpy() * 0.01
                mi_scene['emitter']['to_world'] = mi.ScalarTransform4f.translate(emitter_offset) @ emitter_to_world
                loaded_mi_scene = mi.load_dict(mi_scene)
            
            image = to_torch_f(torch.zeros((h, w, c)))
            grad_image = to_torch_f(torch.zeros((h, w, c)))
            for i in range(npass):
                
                P = mi.Float(0.0)
                drjit.enable_grad(P)
                verts = drjit.unravel(mi.Point3f, params[f'{mesh_id}.vertex_positions'])
                t = mi.Transform4f.translate([P, 0.0, 0.0])
                params[f'{mesh_id}.vertex_positions'] = drjit.ravel(t @ verts)
                params.update()
                
                drjit.forward(P, drjit.ADFlag.ClearEdges)
                
                # drjit_image = mi.render(loaded_mi_scene, params, sensor=mi_sensors[sensor_id], spp=render_options['spp'], seed=seed)
                drjit_image = mi.render(loaded_mi_scene, params, sensor=mi_sensors[sensor_id], seed=seed)
                image += to_torch_f(drjit_image.torch()) / npass

                drjit.forward_to(drjit_image)
                drjit_grad_image = drjit.grad(drjit_image)
                grad_image += to_torch_f(drjit_grad_image.torch()) / npass
                seed += 1
        
        return image, grad_image
        

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
    # elif integrator['type'] == 'direct_reparam':
    #     cache['integrators'][name] = mi_integrator
    # elif integrator['type'] == 'prb_reparam':
    #     mi_integrator['max_depth'] = integrator['config']['max_depth']
    #     cache['integrators'][name] = mi_integrator
    # elif integrator['type'] == 'direct_projective':
    #     mi_integrator['type'] = 'direct_projective'
    #     for key in integrator['config'].keys():
    #         mi_integrator[key] = integrator['config'][key]
    #     cache['integrators'][name] = mi_integrator
    # elif integrator['type'] == 'prb_projective':
    #     mi_integrator['type'] = 'prb_projective'
    #     for key in integrator['config'].keys():
    #         mi_integrator[key] = integrator['config'][key]
    #     cache['integrators'][name] = mi_integrator
    else:
        mi_integrator['type'] = integrator['type']
        for key in integrator['config'].keys():
            mi_integrator[key] = integrator['config'][key]
        cache['integrators'][name] = mi_integrator
        # raise RuntimeError(f"unrecognized integrator type: {integrator['type']}")

    return []

@MitsubaConnector.register(HDRFilm)
def process_hdr_film(name, scene):
    film = scene[name]
    cache = scene.cached['mitsuba']

    cache['film'] = {
        'type': 'hdrfilm',
        'width': film['width'],
        'height': film['height'],
        'sample_border': True,
        # 'filter': {
        #     'type': 'box'
        # }
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

        # code for texture
        verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(configs['device'])), dim=1)
        verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
        if mesh['uv'].nelement() == 0:
            mesh['uv'] = torch.zeros((1, 2)).to(configs['device'])
        if mesh['fuv'].nelement() == 0:
            verts_new = verts
            faces_new = mesh['f'].long()
            uvs_new = torch.zeros(verts.shape[0], 2).to(configs['device'])
        else:
            mesh['fuv'] = torch.zeros_like(mesh['f']).to(configs['device'])
            verts_new, faces_new, uvs_new = compute_texture_coordinates(verts, mesh['f'].long(), mesh['uv'], mesh['fuv'].long())

        # write_mesh('__mitsuba_tmp__.obj', mesh['v'], mesh['f'], mesh['uv'], mesh['fuv'])
        write_mesh('__mitsuba_tmp__.obj', verts_new, faces_new, mesh['uv'], mesh['fuv'])
        mi_bsdf = mi.load_dict(cache['textures'][mat_id])
        mi_mesh_dict = {
            'type': 'obj',
            'filename': '__mitsuba_tmp__.obj',
            'bsdf': mi_bsdf,
            'face_normals': mesh['use_face_normal'],
        }
        if mesh['is_emitter']:
            mi_mesh_dict['area_light'] = {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': mesh['radiance'].tolist()
                }
            }
        
        mi_mesh = mi.load_dict(mi_mesh_dict)
        os.remove('__mitsuba_tmp__.obj')
        
        params = mi.traverse(mi_mesh)
        # params['vertex_count'] = len(verts_new)
        # params['face_count'] = len(faces_new)
        params['faces'] = drjit.ravel(mi.Vector3u(faces_new.cpu().numpy()))
        params['vertex_positions'] = drjit.ravel(mi.Vector3f(verts_new))
        # Compute vertex normals
        # vts_normals = compute_vertex_normals(verts_new, faces_new.long())
        # params['vertex_normals'] = drjit.ravel(mi.Vector3f(vts_normals))
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
                verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(configs['device'])), dim=1)
                verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
                # verts = torch.matmul(mesh['v'], mesh['to_world'][0:3, 0:3].transpose(0, 1))
                # verts = verts + mesh['to_world'][0:3, 3]
                if mesh['uv'].nelement() == 0:
                    mesh['uv'] = torch.zeros((1, 2)).to(configs['device'])
                if mesh['fuv'].nelement() == 0:
                    verts_new = verts
                    faces_new = mesh['f'].long()
                    uvs_new = torch.zeros(verts.shape[0], 2).to(configs['device'])
                else:
                    verts_new, faces_new, uvs_new = compute_texture_coordinates(verts, mesh['f'].long(), mesh['uv'], mesh['fuv'].long())
                params = mi.traverse(mi_mesh)
                # params['vertex_count'] = len(verts_new)
                # params['face_count'] = len(faces_new)
                params['faces'] = drjit.ravel(mi.Vector3u(faces_new.cpu().numpy()))
                params['vertex_positions'] = drjit.ravel(mi.Vector3f(verts_new))
                # Compute vertex normals
                # vts_normals = compute_vertex_normals(verts_new, faces_new.long())
                # params['vertex_normals'] = drjit.ravel(mi.Vector3f(vts_normals))
                # Set vertex texture coordinates
                params['vertex_texcoords'] = drjit.ravel(mi.Vector2f(uvs_new))
                params.update()
                
                mesh.params['v']['updated'] = False
                mesh.params['to_world']['updated'] = False
                updated = mesh.get_updated()
                
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
                    verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(configs['device'])), dim=1)
                    verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
                    # verts = torch.matmul(mesh['v'], mesh['to_world'][0:3, 0:3].transpose(0, 1))
                    # verts = verts + mesh['to_world'][0:3, 3]
                    
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
def process_mitsuba_microfacet_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['mitsuba']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        # TODO: support more BRDF types
        mi_brdf = {
            'type': 'MitsubaMicrofacetBSDF',
            'diffuse': brdf['d'].cpu().numpy(),
            'specular': brdf['s'].cpu().item(),
            'roughness': brdf['r'].cpu().item()
        }

        cache['textures'][name] = mi_brdf
        cache['name_map'][name] = mi_brdf
    
    mi_brdf = cache['name_map'][name]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                mi_brdf['diffuse'] = brdf['d'].cpu().numpy()
            elif param_name == 's':
                mi_brdf['specular'] = brdf['s'].cpu().item()
            elif param_name == 'r':
                mi_brdf['roughness'] = brdf['r'].cpu().item()
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    mitsuba_params = []
    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            pass

    return mitsuba_params

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

@MitsubaConnector.register(SmoothDielectricBRDF)
def process_smooth_dielectric_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['mitsuba']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        mi_brdf = {
            'type': 'dielectric',
            'int_ior': brdf['int_ior'],
            'ext_ior': brdf['ext_ior'],
            'specular_reflectance': {
                'type': 'srgb',
                'color': brdf['s_reflect'].cpu().numpy()
            },
            'specular_transmittance': {
                'type': 'srgb',
                'color': brdf['s_transmit'].cpu().numpy()
            }
        }
        
        cache['textures'][name] = mi_brdf
        cache['name_map'][name] = mi_brdf

    mi_brdf = cache['name_map'][name]

    return []

@MitsubaConnector.register(RoughConductorBRDF)
def process_rough_conductor_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['mitsuba']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        mi_brdf = {
            'type': 'roughconductor',
            'eta': {
                'type': 'srgb',
                'color': brdf['eta'].cpu().numpy()
            },
            'k': {
                'type': 'srgb',
                'color': brdf['k'].cpu().numpy()
            },
            'specular_reflectance': {
                'type': 'srgb',
                'color': brdf['s'].cpu().numpy()
            },
            'distribution': 'ggx',
            'alpha_u': brdf['alpha_u'].cpu().item(),
            'alpha_v': brdf['alpha_v'].cpu().item(),
        }
        
        cache['textures'][name] = mi_brdf
        cache['name_map'][name] = mi_brdf

    mi_brdf = cache['name_map'][name]

    return []

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
        if len(radiance.reshape(-1)) <= 3:
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
                if len(radiance.reshape(-1)) <= 3:
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
    faces_new = torch.cat([i0, i1, i2], axis=1).to(configs['device'])

    return verts_new, faces_new, uvs_new


class MitsubaMicrofacetBSDF(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        self.m_diffuse = props['diffuse']
        self.m_specular = props['specular']
        self.m_roughness = props['roughness']

        # Set the BSDF flags
        reflection_flags   = mi.BSDFFlags.Reflection  | mi.BSDFFlags.FrontSide
        self.m_components  = [reflection_flags]
        self.m_flags = reflection_flags
        # reflection_flags   = mi.BSDFFlags.GlossyReflection   | mi.BSDFFlags.FrontSide
        # diffuse_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        # self.m_components  = [reflection_flags, diffuse_flags]
        # self.m_flags = reflection_flags | diffuse_flags
        
    def sample_visible_11(self, cos_theta_i, sample):
        # print('sample_visible_11')
        p = mi.warp.square_to_uniform_disk_concentric(sample)
        s = 0.5 * (1.0 + cos_theta_i)
        p.y = drjit.lerp(drjit.safe_sqrt(1.0 - drjit.sqr(p.x)), p.y, s)
        x = p.x
        y = p.y
        z = drjit.safe_sqrt(1.0 - drjit.squared_norm(p))
        sin_theta_i = drjit.safe_sqrt(1.0 - drjit.sqr(cos_theta_i))
        norm = drjit.rcp(sin_theta_i * y + cos_theta_i * z)
        return mi.Vector2f(cos_theta_i * y - sin_theta_i * z, x) * norm
    
    def smith_g1(self, v, m, m_alpha_u, m_alpha_v):
        # print('smith_g1')
        xy_alpha_2 = drjit.sqr(m_alpha_u * v.x) + drjit.sqr(m_alpha_v * v.y)
        tan_theta_alpha_2 = xy_alpha_2 / drjit.sqr(v.z)
        result = 2.0 / (1.0 + drjit.sqrt(1.0 + tan_theta_alpha_2))
        # if xy_alpha_2 == 0.0:
        #     result = 1.0
        # if drjit.dot(v, m) * mi.Frame3f.cos_theta(v) <= 0.0:
        #     result = 0.0
        # masked(result, eq(xy_alpha_2, 0.f)) = 1.f;
        # masked(result, dot(v, m) * Frame<ad>::cos_theta(v) <= 0.f) = 0.f;
        return result
    
    def distr_eval(self, m, m_alpha_u, m_alpha_v):
        # print('distr_eval')
        alpha_uv = m_alpha_u * m_alpha_v
        cos_theta = mi.Frame3f.cos_theta(m)
        cos_theta_2 = drjit.sqr(cos_theta)
        result = drjit.rcp(drjit.pi * alpha_uv * drjit.sqr(drjit.sqr(m.x / m_alpha_u) + drjit.sqr(m.y / m_alpha_v) + drjit.sqr(m.z)))
        return drjit.select(result * cos_theta > 1e-5, result, 0.0)
    
    def distr_sample(self, wi, sample2, m_alpha_u, m_alpha_v):
        # print('disr_sample')
        wi_p = drjit.normalize(mi.Vector3f(
            m_alpha_u * wi.x,
            m_alpha_v * wi.y,
            wi.z
        ))

        sin_phi = mi.Frame3f.sin_phi(wi_p)
        cos_phi = mi.Frame3f.cos_phi(wi_p)
        cos_theta = mi.Frame3f.cos_theta(wi_p)
        # sample2 = mi.Vector2f(_sample1, _sample2.x)
        slope = self.sample_visible_11(cos_theta, sample2)

        slope = mi.Vector2f(
            (cos_phi * slope.x - sin_phi * slope.y) * m_alpha_u,
            (sin_phi * slope.x + cos_phi * slope.y) * m_alpha_v
        )
        m = drjit.normalize(mi.Vector3f(-slope.x, -slope.y, 1))

        # Compute probability density of the sampled position
        pdf = self.smith_g1(wi, m, m_alpha_u, m_alpha_v) * drjit.abs(drjit.dot(wi, m)) * self.distr_eval(m, m_alpha_u, m_alpha_v) / drjit.abs(mi.Frame3f.cos_theta(wi))
        pdf = drjit.detach(pdf)
        return m, pdf
    
    def sample(self, ctx, si, sample1, sample2, active):
        # print('sample')
        bs = mi.BSDFSample3f()
        cos_theta_i =  mi.Frame3f.cos_theta(si.wi)
        alpha_u = self.m_roughness
        alpha_v = self.m_roughness
        distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_u, alpha_v)
        m, m_pdf = distr.sample(si.wi, sample2)
        # m, m_pdf = self.distr_sample(si.wi, sample2, alpha_u, alpha_v)
        bs.wo = m * 2.0 * drjit.dot(si.wi, m) - si.wi
        bs.eta = 1.0
        bs.pdf = (m_pdf / (4.0 * drjit.dot(bs.wo, m)))
        bs.sampled_component = 0
        bs.sampled_type = mi.BSDFFlags.Reflection
        
        active = (cos_theta_i > 0.0) & drjit.neq(bs.pdf, 0.0) & (mi.Frame3f.cos_theta(bs.wo) > 0.0) & active
        
        value = self.eval(ctx, si, bs.wo, active) / bs.pdf
        
        return (drjit.detach(bs), drjit.select(active, value, 0.0))

    def eval(self, ctx, si, wo, active):
        # print('eval')
        
        cos_theta_nv = mi.Frame3f.cos_theta(si.wi)    # view dir
        cos_theta_nl = mi.Frame3f.cos_theta(wo)    # light dir
        
        active = active & (cos_theta_nv > 0.0) & (cos_theta_nl > 0.0)
        
        diffuse = mi.Vector3f(self.m_diffuse) / drjit.pi

        H = drjit.normalize(si.wi + wo)
        cos_theta_nh = mi.Frame3f.cos_theta(H)
        cos_theta_vh = drjit.dot(H, si.wi)

        F0 = self.m_specular
        roughness = self.m_roughness
        alpha = drjit.sqr(roughness)
        k = drjit.sqr(roughness + 1.0) / 8.0

        # GGX NDF term
        tmp = alpha / (cos_theta_nh * cos_theta_nh * (drjit.sqr(alpha) - 1.0) + 1.0)
        ggx = tmp * tmp / drjit.pi

        # Fresnel term
        coeff = cos_theta_vh * (-5.55473 * cos_theta_vh - 6.8316)
        fresnel = F0 + (1.0 - F0) * drjit.power(2.0, coeff)

        # Geometry term
        smithG1 = cos_theta_nv / (cos_theta_nv * (1.0 - k) + k)
        smithG2 = cos_theta_nl / (cos_theta_nl * (1.0 - k) + k)
        smithG = smithG1 * smithG2

        numerator = ggx * smithG * fresnel
        denominator = 4.0 * cos_theta_nl * cos_theta_nv
        specular = numerator / (denominator + 1e-6)

        value = (diffuse + specular) * cos_theta_nl
        
        return drjit.select(active, value, 0.0)

    def pdf(self, ctx, si, wo, active):
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        m = drjit.normalize(wo + si.wi)
        
        active = active & (cos_theta_i > 0.0) & (cos_theta_o > 0.0) & (drjit.dot(si.wi, m) > 0.0) & (drjit.dot(wo, m) > 0.0)

        alpha_u = self.m_roughness
        alpha_v = self.m_roughness
        distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, alpha_u, alpha_v)

        # result = (self.distr_eval(m, alpha_u, alpha_v) * self.smith_g1(si.wi, m, alpha_u, alpha_v) / (4.0 * cos_theta_i))
        result = (distr.eval(m) * distr.smith_g1(si.wi, m) / (4.0 * cos_theta_i))
        return drjit.detach(result)

    def eval_pdf(self, ctx, si, wo, active):
        eval_value = self.eval(ctx, si, wo, active)
        pdf_value = self.pdf(ctx, si, wo, active)
        return eval_value, pdf_value

    def traverse(self, callback):
        callback.put_parameter('diffuse', self.m_diffuse, mi.ParamFlags.Differentiable)
        callback.put_parameter('specular', self.m_specular, mi.ParamFlags.Differentiable)
        callback.put_parameter('roughness', self.m_roughness, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        print("🏝️ there is nothing to do here 🏝️")

    def to_string(self):
        return ('MicrofacetBSDF[\n'
                '    diffuse=%s,\n'
                '    specular=%s,\n'
                '    roughness=%s,\n'
                ']' % (self.m_diffuse, self.m_specular, self.m_roughness))