from ..connector import Connector
from ..scene import *
from ..config import *
from ..io import write_mesh
from collections import OrderedDict
from ..utils import Timer

import nvdiffrast.torch as dr
import torch
import numpy as np
import math
import time

class NvdiffrastConnector(Connector, connector_name='nvdiffrast'):

    def __init__(self):
        super().__init__()

    def update_scene_objects(self, scene, render_options):
        if 'nvdiffrast' in scene.cached:
            cache = scene.cached['nvdiffrast']
        else:
            cache = {}
            scene.cached['nvdiffrast'] = cache

            cache['meshes'] = []
            cache['textures'] = dict()
            cache['cameras'] = []
            cache['point_light'] = None
            cache['film'] = None

            cache['name_map'] = {}

        nvdiffrast_params = []
        for name in scene.components:
            component = scene[name]
            if type(component) in self.extensions:
                nvdiffrast_params += self.extensions[type(component)](name, scene)
            else:
                raise RuntimeError(f'Unsupported component for nvdiffrast: {component}')
        
        return scene.cached['nvdiffrast'], nvdiffrast_params
   
    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        cache, _ = self.update_scene_objects(scene, render_options)

        # Render.
        self.glctx = dr.RasterizeCudaContext()

        images = []
        for sensor_id in sensor_ids:
            # Modelview and modelview + projection matrices.
            to_world = cache['cameras'][sensor_id]['to_world'].clone()
            to_world[:3, 0]  = -to_world[:3, 0]
            to_world[:3, 2]  = -to_world[:3, 2]
            to_world[:3, 3]  = -to_world[:3, 3] * render_options['scale']
            to_world[:3, :3] = torch.transpose(to_world[:3, :3], 0, 1)
            to_world[:3, 3]  = torch.matmul(to_world[:3, :3], to_world[:3, 3])
            a_mv = to_world
            # a_mv = lookAt(np.array([0, 0, -2.7]), np.array([0, 0, 0]), np.array([0, 1, 0]))
            # a_mv = lookAt(np.array([-1.5, 1.5, 1.5]), np.array([0, 0, 0]), np.array([0, 1, 0]))
            # print(a_mv)
            # proj_mtx = projection(math.tan(cache['cameras'][sensor_id]['fov']/2.0*math.pi/180), cache['cameras'][sensor_id]['near'], cache['cameras'][sensor_id]['far'])
            proj_mtx = projection(math.tan(cache['cameras'][sensor_id]['fov']/2.0*math.pi/180))
            a_mvp = torch.matmul(proj_mtx.to(configs['device']), a_mv.to(configs['device']))
            if cache['point_light']:
                a_lightpos = cache['point_light']['position']
                a_light_power = cache['point_light']['radiance']
            else:
                a_lightpos = torch.inverse(a_mv)[None, :3, 3]
                a_light_power = render_options['light_power']
            a_campos = torch.inverse(a_mv)[None, :3, 3]

            name    = cache['meshes'][0]['mat_id']
            pos_idx = cache['meshes'][0]['f'].int()
            vtx_pos = cache['meshes'][0]['v'].float() * render_options['scale']
            uv_idx  = cache['meshes'][0]['fuv'].int()
            vtx_uv  = cache['meshes'][0]['uv'].float()
            # Compute vertex normals
            vts_normals = compute_vertex_normals(vtx_pos, pos_idx.long())
            # Material
            bsdf = cache['textures'][name]['bsdf']
            if len(cache['textures'][name]['d']) <= 3:
                kd = to_torch_f(cache['textures'][name]['d'])
                if 's' in cache['textures'][name]:
                    ks = to_torch_f([cache['textures'][name]['s'].item(), cache['textures'][name]['r'].item(), 0])
                    ks = ks.unsqueeze(0).unsqueeze(0)
                else:
                    ks = to_torch_f((0, 0, 0))
            else:
                # Flip texture
                kd = torch.flip(cache['textures'][name]['d'], [0])
                ks = to_torch_f((0, 0, 0))
            
            image = None
            npass = render_options['npass']
            with Timer('Forward'):
                for i in range(npass):
                    # Basic render
                    # tex = kd
                    # image_pass = render(glctx, a_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex, cache['film'], enable_mip=True, max_mip_level=9)
                    
                    # Shading
                    a_mvp = a_mvp[None, ...]
                    params = {'mvp' : a_mvp, 'lightpos' : a_lightpos, 'campos' : a_campos, 'resolution' : cache['film'], 'time' : 0}
                    material = {
                        'kd': Texture2D(kd),
                        'ks': Texture2D(ks),
                        'bsdf': bsdf
                    }
                    nv_mesh =  NvMesh(vtx_pos, pos_idx.long(), vts_normals, pos_idx.long(), vtx_uv, uv_idx.long(), v_weights=None, bone_mtx=None, material=material)
                    nv_mesh_tng = compute_tangents(nv_mesh)

                    image_pass = render_mesh(self.glctx, nv_mesh_tng.eval(params), a_mvp, a_campos, a_lightpos, a_light_power, cache['film'][1], 
                        num_layers=1, spp=1, background=None, min_roughness=0.08)

                    image_pass.squeeze_(0)
                    if image:
                        image += image_pass / npass
                    else:
                        image = image_pass / npass
                
                images.append(image)
        
        return images
    
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        with torch.enable_grad():
            cache, nv_params = self.update_scene_objects(scene, render_options)

            # Render.
            glctx = dr.RasterizeCudaContext()

            param_grads = [torch.zeros_like(scene[param_name]) for param_name in scene.requiring_grad]
            for img_index, sensor_id in enumerate(sensor_ids):
                # Modelview and modelview + projection matrices.
                to_world = cache['cameras'][sensor_id]['to_world'].clone()
                to_world[:3, 0]  = -to_world[:3, 0]
                to_world[:3, 2]  = -to_world[:3, 2]
                to_world[:3, 3]  = -to_world[:3, 3] * render_options['scale']
                to_world[:3, :3] = torch.transpose(to_world[:3, :3], 0, 1)
                to_world[:3, 3]  = torch.matmul(to_world[:3, :3], to_world[:3, 3])
                a_mv = to_world
                # a_mv = lookAt(np.array([0, 0, -2.7]), np.array([0, 0, 0]), np.array([0, 1, 0]))
                # a_mv = lookAt(np.array([-1.5, 1.5, 1.5]), np.array([0, 0, 0]), np.array([0, 1, 0]))
                # print(a_mv)
                # proj_mtx = projection(math.tan(cache['cameras'][sensor_id]['fov']/2.0*math.pi/180), cache['cameras'][sensor_id]['near'], cache['cameras'][sensor_id]['far'])
                proj_mtx = projection(math.tan(cache['cameras'][sensor_id]['fov']/2.0*math.pi/180))
                a_mvp = torch.matmul(proj_mtx.to(configs['device']), a_mv.to(configs['device']))
                if cache['point_light']:
                    a_lightpos = cache['point_light']['position']
                    a_light_power = cache['point_light']['radiance']
                else:
                    a_lightpos = torch.inverse(a_mv)[None, :3, 3]
                    a_light_power = render_options['light_power']
                a_campos = torch.inverse(a_mv)[None, :3, 3]

                name    = cache['meshes'][0]['mat_id']
                pos_idx = cache['meshes'][0]['f'].int()
                vtx_pos = cache['meshes'][0]['v'].float() * render_options['scale']
                uv_idx  = cache['meshes'][0]['fuv'].int()
                vtx_uv  = cache['meshes'][0]['uv'].float()
                # Compute vertex normals
                vts_normals = compute_vertex_normals(vtx_pos, pos_idx.long())
                # Material
                bsdf = cache['textures'][name]['bsdf']
                if len(cache['textures'][name]['d']) <= 3:
                    kd = to_torch_f(cache['textures'][name]['d'])
                    if 'ks' in cache['textures'][name]:
                        ks = cache['textures'][name]['ks']
                    else:
                        ks = to_torch_f((0, 0, 0))
                else:
                    # Flip texture
                    kd = torch.flip(cache['textures'][name]['d'], [0])
                    ks = to_torch_f((0, 0, 0))
                
                npass = render_options['npass']
                with Timer('Backward'):
                    for i in range(npass):
                        # Basic render
                        # tex = kd
                        # image_pass = render(glctx, a_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex, cache['film'], enable_mip=True, max_mip_level=9)
                        
                        # Shading
                        a_mvp = a_mvp[None, ...]
                        params = {'mvp' : a_mvp, 'lightpos' : a_lightpos, 'campos' : a_campos, 'resolution' : cache['film'], 'time' : 0}
                        material = {
                            'kd': Texture2D(kd),
                            'ks': Texture2D(ks),
                            'bsdf': bsdf
                        }
                        nv_mesh =  NvMesh(vtx_pos, pos_idx.long(), vts_normals, pos_idx.long(), vtx_uv, uv_idx.long(), v_weights=None, bone_mtx=None, material=material)
                        nv_mesh_tng = compute_tangents(nv_mesh)
                        
                        image_pass = render_mesh(glctx, nv_mesh_tng.eval(params), a_mvp, a_campos, a_lightpos, a_light_power, cache['film'][1], 
                            num_layers=1, spp=1, background=None, min_roughness=0.08)

                        image_pass.squeeze_(0)
                        image_grad = image_grads[img_index] / npass
                        tmp = (image_grad[..., :3] * image_pass).sum(dim=2)
                        nv_grads = torch.autograd.grad(tmp, nv_params, torch.ones_like(tmp), retain_graph=True)
                        for param_grad, nv_grad in zip(param_grads, nv_grads):
                            param_grad += nv_grad / render_options['scale']

            return param_grads

    def renderGrad(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        # mesh position.x offset
        x = torch.tensor([0.0], requires_grad=True)
        offset = torch.zeros(3)
        offset[0] = x[0]
        offset = to_torch_f(offset)
        offset = offset.repeat(scene['object']['v'].shape[0], 1)
        scene['object']['v'] = scene['object']['v'] + offset

        images = torch.stack(self.renderC(scene, render_options, sensor_ids, integrator_id))
        grads = torch.zeros_like(images).reshape(images.shape[0], -1)
        for i, image in enumerate(list(images)):
            for j in range(image.nelement()):
                grad = torch.autograd.grad(image.reshape(-1)[j], x, retain_graph=True)
                grads[i, j] = grad[0]
                
                if j % 100 == 0 or j == image.nelement() - 1:
                    print(f'\rImage {i}: {j}/{image.nelement() - 1}', end='')

        grads = grads.reshape(images.shape)
        # grads = torch.autograd.functional.jacobian(render, x, vectorize=True)
        # grads.squeeze_(4)
        
        # x = torch.tensor([0.01], requires_grad=True)
        # images = render(x)
        return list(grads)

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

@NvdiffrastConnector.register(HDRFilm)
def process_hdr_film(name, scene):
    film = scene[name]
    cache = scene.cached['nvdiffrast']

    cache['film'] = (film['height'], film['width'])

    return []

@NvdiffrastConnector.register(PerspectiveCamera)
def process_perspective_camera(name, scene):
    sensor = scene[name]
    cache = scene.cached['nvdiffrast']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        pytorch_sensor = {
            'near': sensor['near'],
            'far': sensor['far'],
            'fov': sensor['fov'],
            'to_world': sensor['to_world']
        }
        R = pytorch_sensor['to_world'][:3, :3]
        T = -torch.mm(pytorch_sensor['to_world'][:3, 3][None], R).squeeze()
        
        pytorch_sensor['R'] = R
        pytorch_sensor['T'] = T
        
        cache['cameras'].append(pytorch_sensor)
        cache['name_map'][name] = pytorch_sensor

    pytorch_sensor = cache['name_map'][name]
    
    # Update parameters
    updated = sensor.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == "to_world":
                pytorch_sensor['to_world'] = sensor['to_world']
                R = pytorch_sensor['to_world'][:3, :3]
                T = -torch.mm(pytorch_sensor['to_world'][:3, 3][None], R).squeeze()
                pytorch_sensor['R'] = R
                pytorch_sensor['T'] = T
            sensor.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    torch_params = []
    requiring_grad = sensor.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == "to_world":
                pytorch_sensor['to_world'].requires_grad_()
                # ensure 'to_world' is linked with R,T
                R = pytorch_sensor['to_world'][:3, :3]
                T = -torch.mm(pytorch_sensor['to_world'][:3, 3][None], R).squeeze()
                pytorch_sensor['R'] = R
                pytorch_sensor['T'] = T
                torch_params.append(pytorch_sensor['to_world'])
                
    return torch_params

@NvdiffrastConnector.register(Mesh)
def process_mesh(name, scene):
    mesh = scene[name]
    cache = scene.cached['nvdiffrast']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        # Create its material first
        mat_id = mesh['mat_id']
        if mat_id not in scene:
            raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
        
        verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(configs['device'])), dim=1)
        verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
        if mesh['uv'].nelement() == 0:
            mesh['uv'] = torch.zeros((1, 2)).to(configs['device'])
        if mesh['fuv'].nelement() == 0:
            mesh['fuv'] = torch.zeros_like(mesh['f']).to(configs['device'])
            
        nvdiffrast_mesh = {
            'v': verts,
            'f': mesh['f'],
            'uv': mesh['uv'],
            'fuv': mesh['fuv'],
            'mat_id': mesh['mat_id'],
            'use_face_normal': mesh['use_face_normal']
        }
        
        cache['meshes'].append(nvdiffrast_mesh)
        cache['name_map'][name] = nvdiffrast_mesh

    nvdiffrast_mesh = cache['name_map'][name]
    
    # Update parameters
    updated = mesh.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'v' or param_name == 'to_world':
                verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(configs['device'])), dim=1)
                verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
                if mesh['uv'].nelement() == 0:
                    mesh['uv'] = torch.zeros((1, 2)).to(configs['device'])
                if mesh['fuv'].nelement() == 0:
                    mesh['fuv'] = torch.zeros_like(mesh['f']).to(configs['device'])
                    
                if mesh['can_change_topology']:
                    nvdiffrast_mesh['v'] = verts
                    nvdiffrast_mesh['f'] = mesh['f']
                    nvdiffrast_mesh['uv'] = mesh['uv']
                    nvdiffrast_mesh['fuv'] = mesh['fuv']
                    nvdiffrast_mesh['mat_id'] = mesh['mat_id']
                    nvdiffrast_mesh['use_face_normal'] = mesh['use_face_normal']
                else:
                    nvdiffrast_mesh['v'] = verts
                    nvdiffrast_mesh['f'] = mesh['f']

            mesh.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    nvdiffrast_params = []
    requiring_grad = mesh.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'v':
                nvdiffrast_mesh['v'].requires_grad = True
                nvdiffrast_params.append(nvdiffrast_mesh['v'])

    return nvdiffrast_params

"""
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
"""

@NvdiffrastConnector.register(DiffuseBRDF)
def process_diffuse_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['nvdiffrast']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        nvdiffrast_brdf = {}
        nvdiffrast_brdf['bsdf'] = 'diffuse'
        if brdf['d'].dim() == 1:
            nvdiffrast_brdf['d'] = brdf['d'].reshape(1, 1, 3)
        else:
            nvdiffrast_brdf['d'] = brdf['d']
        
        cache['textures'][name] = nvdiffrast_brdf
        cache['name_map'][name] = nvdiffrast_brdf

    nvdiffrast_brdf = cache['name_map'][name]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                if brdf['d'].dim() == 1:
                    nvdiffrast_brdf['d'] = brdf['d'].reshape(1, 1, 3)
                else: 
                    nvdiffrast_brdf['d'] = brdf['d']
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    nvdiffrast_params = []
    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'd':
                nvdiffrast_brdf['d'].requires_grad = True
                nvdiffrast_params.append(nvdiffrast_brdf['d'])

    return nvdiffrast_params

@NvdiffrastConnector.register(MicrofacetBRDF)
def process_microfacet_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['nvdiffrast']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        nvdiffrast_brdf = {
            'd': brdf['d'],
            's': brdf['s'],
            'r': brdf['r'],
            'bsdf': 'microfacet'
        }
        cache['textures'][name] = nvdiffrast_brdf
        cache['name_map'][name] = nvdiffrast_brdf

    nvdiffrast_brdf = cache['name_map'][name]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                nvdiffrast_brdf['d'] = brdf['d']
            elif param_name == 's':
                nvdiffrast_brdf['s'] = brdf['s']
            elif param_name == 'r':
                nvdiffrast_brdf['r'] = brdf['r']
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    nvdiffrast_params = []
    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'd':
                nvdiffrast_brdf['d'].requires_grad = True
                nvdiffrast_params.append(nvdiffrast_brdf['d'])
            elif param_name == 's':
                nvdiffrast_brdf['s'].requires_grad = True
                nvdiffrast_params.append(nvdiffrast_brdf['s'])
            elif param_name == 'r':
                nvdiffrast_brdf['r'].requires_grad = True
                nvdiffrast_params.append(nvdiffrast_brdf['r'])

    return nvdiffrast_params

"""
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

@NvdiffrastConnector.register(PointLight)
def process_point_light(name, scene):
    light = scene[name]
    cache = scene.cached['nvdiffrast']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        nvdiffrast_point_light = {
            'radiance': light['radiance'],
            'position': light['position']
        }
        
        cache['point_light'] = nvdiffrast_point_light
        cache['name_map'][name] = nvdiffrast_point_light

    nvdiffrast_point_light = cache['name_map'][name]
    
    # Update parameters
    updated = light.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == "position":
                nvdiffrast_point_light['position'] = light['position']
            light.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    nvdiffrast_params = []
    requiring_grad = light.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == "position":
                nvdiffrast_point_light['position'].requires_grad_()
                nvdiffrast_params.append(nvdiffrast_point_light['position'])
                
    return nvdiffrast_params

#----------------------------------------------------------------------------
# Helpers.
def projection(x=0.1, n=1.0, f=1000.0):
    return torch.Tensor([[n/x,    0,            0,              0], 
                        [  0, n/-x,            0,              0], 
                        [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                        [  0,    0,           -1,              0]])

def lookAt(eye, at, up):
    a = eye - at
    b = up
    w = a / np.linalg.norm(a)
    u = np.cross(b, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    translate = np.array([[1, 0, 0, -eye[0]], 
                        [0, 1, 0, -eye[1]], 
                        [0, 0, 1, -eye[2]], 
                        [0, 0, 0, 1]]).astype(np.float32)
    rotate =  np.array([[u[0], u[1], u[2], 0], 
                        [v[0], v[1], v[2], 0], 
                        [w[0], w[1], w[2], 0], 
                        [0, 0, 0, 1]]).astype(np.float32)
    return to_torch_f(np.matmul(rotate, translate))

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

# Image scaling
def scale_img_hwc(x : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def avg_pool_nhwc(x  : torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

# Vector utility functions
NORMAL_THRESHOLD = 0.1

def _dot(x, y):
    return torch.sum(x*y, -1, keepdim=True)

def _reflect(x, n):
    return 2*_dot(x, n)*n - x

def _safe_normalize(x):
    return torch.nn.functional.normalize(x, dim = -1)

def _bend_normal(view_vec, smooth_nrm, geom_nrm, two_sided_shading):
    # Swap normal direction for backfacing surfaces
    if two_sided_shading:
        smooth_nrm = torch.where(_dot(geom_nrm, view_vec) > 0, smooth_nrm, -smooth_nrm)
        geom_nrm   = torch.where(_dot(geom_nrm, view_vec) > 0, geom_nrm, -geom_nrm)

    t = torch.clamp(_dot(view_vec, smooth_nrm) / NORMAL_THRESHOLD, min=0, max=1)
    return torch.lerp(geom_nrm, smooth_nrm, t)

def _perturb_normal(perturbed_nrm, smooth_nrm, smooth_tng, opengl):
    smooth_bitang = _safe_normalize(torch.cross(smooth_tng, smooth_nrm))
    if opengl:
        shading_nrm = smooth_tng * perturbed_nrm[..., 0:1] - smooth_bitang * perturbed_nrm[..., 1:2] + smooth_nrm * torch.clamp(perturbed_nrm[..., 2:3], min=0.0)
    else:
        shading_nrm = smooth_tng * perturbed_nrm[..., 0:1] + smooth_bitang * perturbed_nrm[..., 1:2] + smooth_nrm * torch.clamp(perturbed_nrm[..., 2:3], min=0.0)
    return _safe_normalize(shading_nrm)

def bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
    smooth_nrm = _safe_normalize(smooth_nrm)
    smooth_tng = _safe_normalize(smooth_tng)
    view_vec   = _safe_normalize(view_pos - pos)
    shading_nrm = _perturb_normal(perturbed_nrm, smooth_nrm, smooth_tng, opengl)
    return _bend_normal(view_vec, shading_nrm, geom_nrm, two_sided_shading)

# Shading normal setup (bump mapping + bent normals)
def prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading=True, opengl=True):
    '''Takes care of all corner cases and produces a final normal used for shading:
        - Constructs tangent space
        - Flips normal direction based on geometric normal for two sided Shading
        - Perturbs shading normal by normal map
        - Bends backfacing normals towards the camera to avoid shading artifacts

        All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        pos: World space g-buffer position.
        view_pos: Camera position in world space (typically using broadcasting).
        perturbed_nrm: Trangent-space normal perturbation from normal map lookup.
        smooth_nrm: Interpolated vertex normals.
        smooth_tng: Interpolated vertex tangents.
        geom_nrm: Geometric (face) normals.
        two_sided_shading: Use one/two sided shading
        opengl: Use OpenGL/DirectX normal map conventions 
    Returns:
        Final shading normal
    '''    

    if perturbed_nrm is None:
        perturbed_nrm = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda', requires_grad=False)[None, None, None, ...]
    
    out = bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of prepare_shading_normal contains inf or NaN"
    return out

# Simple lambertian diffuse BSDF
def bsdf_lambert(nrm, wi):
    return torch.clamp(_dot(nrm, wi), min=0.0) / math.pi

# Phong specular, loosely based on mitsuba implementation
def bsdf_phong(nrm, wo, wi, N):
    dp_r = torch.clamp(_dot(_reflect(wo, nrm), wi), min=0.0, max=1.0)
    dp_l = torch.clamp(_dot(nrm, wi), min=0.0, max=1.0)
    return (dp_r ** N) * dp_l * (N + 2) / (2 * math.pi)

# PBR's implementation of GGX specular
SPECULAR_EPSILON = 1e-4

def bsdf_ndf_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=SPECULAR_EPSILON, max=1.0 - SPECULAR_EPSILON)
    d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1
    return alphaSqr / (d * d * math.pi)

def bsdf_lambda_ggx(alphaSqr, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=SPECULAR_EPSILON, max=1.0 - SPECULAR_EPSILON)
    cosThetaSqr = _cosTheta * _cosTheta
    tanThetaSqr = (1.0 - cosThetaSqr) / cosThetaSqr
    res = 0.5 * (torch.sqrt(1 + alphaSqr * tanThetaSqr) - 1.0)
    return res

def bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO):
    lambdaI = bsdf_lambda_ggx(alphaSqr, cosThetaI)
    lambdaO = bsdf_lambda_ggx(alphaSqr, cosThetaO)
    return 1 / (1 + lambdaI + lambdaO)

def bsdf_fresnel_shlick(f0, f90, cosTheta):
    _cosTheta = torch.clamp(cosTheta, min=SPECULAR_EPSILON, max=1.0 - SPECULAR_EPSILON)
    return f0 + (f90 - f0) * (1.0 - _cosTheta) ** 5.0

def bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08):
    _alpha = torch.clamp(alpha, min=min_roughness*min_roughness, max=1.0)
    alphaSqr = _alpha * _alpha

    h = _safe_normalize(wo + wi)
    woDotN = _dot(wo, nrm)
    wiDotN = _dot(wi, nrm)
    woDotH = _dot(wo, h)
    nDotH  = _dot(nrm, h)

    D = bsdf_ndf_ggx(alphaSqr, nDotH)
    G = bsdf_masking_smith_ggx_correlated(alphaSqr, woDotN, wiDotN)
    F = bsdf_fresnel_shlick(col, 1, woDotH)

    w = F * D * G * 0.25 / torch.clamp(woDotN, min=SPECULAR_EPSILON)

    frontfacing = (woDotN > SPECULAR_EPSILON) & (wiDotN > SPECULAR_EPSILON)
    return torch.where(frontfacing, w, torch.zeros_like(w))

def microfacet_eval(kd, ks, nrm, wo, wi, _roughness):
    
    cos_theta_nv = _dot(wi, nrm)    # view dir
    cos_theta_nl = _dot(wo, nrm)    # light dir
    
    diffuse = kd / torch.pi

    H = _safe_normalize(wi + wo)
    cos_theta_nh = _dot(nrm, H)
    cos_theta_vh = _dot(H, wi)

    F0 = ks
    roughness = _roughness
    alpha = roughness ** 2
    k = (roughness + 1) ** 2 / 8

    # GGX NDF term
    tmp = alpha / (cos_theta_nh * cos_theta_nh * (alpha ** 2 - 1) + 1)
    ggx = tmp * tmp / torch.pi

    # Fresnel term
    coeff = cos_theta_vh * (-5.55473 * cos_theta_vh - 6.8316)
    fresnel = F0 + (1 - F0) * torch.pow(2, coeff)

    # Geometry term
    smithG1 = cos_theta_nv / (cos_theta_nv * (1 - k) + k)
    smithG2 = cos_theta_nl / (cos_theta_nl * (1 - k) + k)
    smithG = smithG1 * smithG2

    numerator = ggx * smithG * fresnel
    denominator = 4 * cos_theta_nl * cos_theta_nv
    specular = numerator / (denominator + 1e-6)

    value = (diffuse + specular) * cos_theta_nl
    
    return value

def bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08):
    wo = _safe_normalize(view_pos - pos)
    wi = _safe_normalize(light_pos - pos)

    spec_str  = arm[..., 0:1] # x component
    roughness = arm[..., 1:2] # y component
    metallic  = arm[..., 2:3] # z component
    ks = (0.04 * (1.0 - metallic) + kd * metallic) * (1 - spec_str)
    kd = kd * (1.0 - metallic)

    diffuse = kd * bsdf_lambert(nrm, wi)
    specular = bsdf_pbr_specular(ks, nrm, wo, wi, roughness*roughness, min_roughness=min_roughness)
    return diffuse + specular

def bsdf_microfacet(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08):
    wo = _safe_normalize(view_pos - pos)
    wi = _safe_normalize(light_pos - pos)

    spec_str  = arm[..., 0:1] # x component
    roughness = arm[..., 1:2] # y component
    metallic  = arm[..., 2:3] # z component
    ks = spec_str
    kd = kd * (1.0 - metallic)
    
    return microfacet_eval(kd, ks, nrm, wo, wi, roughness) / _dot(pos - light_pos, pos - light_pos)

# BSDF functions
def lambert(nrm, wi):
    '''Lambertian bsdf. 
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        nrm: World space shading normal.
        wi: World space light vector.

    Returns:
        Shaded diffuse value with shape [minibatch_size, height, width, 1]
    '''

    out = bsdf_lambert(nrm, wi)
 
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of lambert contains inf or NaN"
    return out

def pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08):
    '''Physically-based bsdf, both diffuse & specular lobes
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        kd: Diffuse albedo.
        arm: Specular parameters (attenuation, linear roughness, metalness).
        pos: World space position.
        nrm: World space shading normal.
        view_pos: Camera position in world space, typically using broadcasting.
        light_pos: Light position in world space, typically using broadcasting.
        min_roughness: Scalar roughness clamping threshold

    Returns:
        Shaded color.
    '''    

    out = bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=min_roughness)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_bsdf contains inf or NaN"
    return out

def microfacet_bsdf(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08):
    '''Microfacet bsdf, both diffuse & specular lobes
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        kd: Diffuse albedo.
        arm: Specular parameters (attenuation, linear roughness, metalness).
        pos: World space position.
        nrm: World space shading normal.
        view_pos: Camera position in world space, typically using broadcasting.
        light_pos: Light position in world space, typically using broadcasting.
        min_roughness: Scalar roughness clamping threshold

    Returns:
        Shaded color.
    '''    

    out = bsdf_microfacet(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=min_roughness)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of micro_bsdf contains inf or NaN"
    return out

# Transform points function
def xfm_points(points, matrix):
    '''Transform points. 

    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    
    out = torch.matmul(torch.nn.functional.pad(points, pad=(0,1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# Basic render
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution)

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color

#----------------------------------------------------------------------------
# Simple texture class. A texture can be either 
# - A 3D tensor (using auto mipmaps)
# - A list of 3D tensors (full custom mip hierarchy)
class Texture2D:
     # Initializes a texture from image data.
     # Input can be constant value (1D array) or texture (3D array) or mip hierarchy (list of 3d arrays)
    def __init__(self, init):
        if isinstance(init, np.ndarray):
            init = torch.tensor(init, dtype=torch.float32, device='cuda')
        elif isinstance(init, list) and len(init) == 1:
            init = init[0]

        if isinstance(init, list) or len(init.shape) == 4:
            self.data = init
        elif len(init.shape) == 3:
            self.data = init[None, ...]
        else:
            self.data = init[None, None, None, :] # Convert constant to 1x1 tensor

    # Filtered (trilinear) sample texture at a given location
    def sample(self, texc, texc_deriv, filter_mode='linear-mipmap-linear', data_fmt=torch.float32):
        if isinstance(self.data, list):
            out = dr.texture(self.data[0], texc, texc_deriv, mip=self.data[1:], filter_mode=filter_mode)
        else:
            out = dr.texture(self.data, texc, texc_deriv, filter_mode=filter_mode)
        return out.to(data_fmt)

    def getRes(self):
        return self.getMips()[0].shape[1:3]

    def getMips(self):
        if isinstance(self.data, list):
            return self.data
        else:
            return [self.data]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self, min=None, max=None):
        with torch.no_grad():
            for mip in self.getMips():
                mip.clamp_(min=min, max=max)

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_rgb_(self, minR=None, maxR=None, minG=None, maxG=None, minB=None, maxB=None):
        with torch.no_grad():
            for mip in self.getMips():
                mip[...,0].clamp_(min=minR, max=maxR)
                mip[...,1].clamp_(min=minG, max=maxG)
                mip[...,2].clamp_(min=minB, max=maxB)

#----------------------------------------------------------------------------
# Base mesh class
class NvMesh:
    def __init__(self, v_pos=None, t_pos_idx=None, v_nrm=None, t_nrm_idx=None, v_tex=None, t_tex_idx=None, v_tng=None, t_tng_idx=None, 
    v_weights=None, bone_mtx=None, material=None, base=None):
        self.v_pos = v_pos
        self.v_weights = v_weights
        self.v_nrm = v_nrm
        self.v_tex = v_tex
        self.v_tng = v_tng
        self.t_pos_idx = t_pos_idx
        self.t_nrm_idx = t_nrm_idx
        self.t_tex_idx = t_tex_idx
        self.t_tng_idx = t_tng_idx
        self.material = material
        self.bone_mtx = bone_mtx

        if base is not None:
            self.copy_none(base)

    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        if self.v_weights is None:
            self.v_weights = other.v_weights
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
        if self.v_tex is None:
            self.v_tex = other.v_tex
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        if self.material is None:
            self.material = other.material
        if self.bone_mtx is None:
            self.bone_mtx = other.bone_mtx

    def get_frames(self):
        return self.bone_mtx.shape[0] if self.bone_mtx is not None else 1

    def clone(self):
        out = NvMesh(base=self)
        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone()
        if out.v_weights is not None:
            out.v_weights = out.v_weights.clone()
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone()
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone()
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone()
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone()
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone()
        if out.bone_mtx is not None:
            out.bone_mtx = out.bone_mtx.clone()
        return out

    def eval(self, params={}):
        return self

# Compute tangent space from texture map coordinates
def compute_tangents(mesh):
    class mesh_op_compute_tangents:
        def __init__(self, input):
            self.input = input

        def eval(self, params={}):
            imesh = self.input.eval(params)

            vn_idx = [None] * 3
            pos = [None] * 3
            tex = [None] * 3
            for i in range(0,3):
                pos[i] = imesh.v_pos[imesh.t_pos_idx[:, i]]
                tex[i] = imesh.v_tex[imesh.t_tex_idx[:, i]]
                vn_idx[i] = imesh.t_nrm_idx[:, i]

            tangents = torch.zeros_like(imesh.v_nrm)
            tansum   = torch.zeros_like(imesh.v_nrm)

            # Compute tangent space for each triangle
            uve1 = tex[1] - tex[0]
            uve2 = tex[2] - tex[0]
            pe1  = pos[1] - pos[0]
            pe2  = pos[2] - pos[0]
            
            nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
            denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
            
            # Avoid division by zero for degenerated texture coordinates
            tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

            # Update all 3 vertices
            for i in range(0,3):
                idx = vn_idx[i][:, None].repeat(1,3)
                tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang
                tansum.scatter_add_(0, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
            tangents = tangents / tansum

            # Normalize and make sure tangent is perpendicular to normal
            tangents = _safe_normalize(tangents)
            tangents = _safe_normalize(tangents - _dot(tangents, imesh.v_nrm) * imesh.v_nrm)

            self.v_tng = tangents

            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(tangents))

            return NvMesh(v_tng=self.v_tng, t_tng_idx=imesh.t_nrm_idx, base=imesh)

    return mesh_op_compute_tangents(mesh)

#----------------------------------------------------------------------------
#  pixel shader
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        light_pos,
        light_power,
        material,
        min_roughness
    ):

    # Texture lookups
    kd = material['kd'].sample(gb_texc, gb_texc_deriv)
    ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3] # skip alpha
    perturbed_nrm = None
    if 'normal' in material:
        perturbed_nrm = material['normal'].sample(gb_texc, gb_texc_deriv)

    gb_normal = prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)

    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1]) 
    kd = kd[..., 0:3]

    # Evaluate BSDF
    assert 'bsdf' in material, "Material must specify a BSDF type"
    if material['bsdf'] == 'pbr':
        shaded_col = pbr_bsdf(kd, ks, gb_pos, gb_normal, view_pos, light_pos, min_roughness) * light_power
    if material['bsdf'] == 'microfacet':
        shaded_col = microfacet_bsdf(kd, ks, gb_pos, gb_normal, view_pos, light_pos, min_roughness) * light_power
    elif material['bsdf'] == 'diffuse':
        shaded_col = kd * lambert(gb_normal, _safe_normalize(light_pos - gb_pos)) * light_power
    elif material['bsdf'] == 'normal':
        shaded_col = (gb_normal + 1.0)*0.5
    elif material['bsdf'] == 'tangent':
        shaded_col = (gb_tangent + 1.0)*0.5
    else:
        assert False, "Invalid BSDF '%s'" % material['bsdf']

    out = torch.cat((shaded_col, alpha), dim=-1)

    return out

#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        light_pos,
        light_power,
        resolution,
        min_roughness,
        spp,
        msaa
    ):

    full_res = resolution*spp

    ## Rasterize
    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = scale_img_nhwc(rast, [resolution, resolution], mag='nearest', min='nearest')
        rast_out_deriv_s = scale_img_nhwc(rast_deriv, [resolution, resolution], mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ## Interpolate attributes
    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
    face_normals = _safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents

    # Texure coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)

    ## Shade
    color = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, 
        view_pos, light_pos, light_power, mesh.material, min_roughness)

    ## Prepare output
    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        color = scale_img_nhwc(color, [full_res, full_res], mag='nearest', min='nearest')

    # Return color & raster output for peeling
    return color

#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        view_pos,
        light_pos,
        light_power,
        resolution,
        spp                       = 1,
        num_layers                = 1,
        msaa                      = False,
        background                = None,
        antialias                 = True,
        min_roughness             = 0.08
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    full_res = resolution*spp

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    light_pos   = prepare_input_vector(light_pos)
    light_power = prepare_input_vector(light_power)
    view_pos    = prepare_input_vector(view_pos)

    # clip space transform
    v_pos_clip = xfm_points(mesh.v_pos[None, ...], mtx_in)

    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), [resolution*spp, resolution*spp]) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(rast, db, mesh, view_pos, light_pos, light_power, resolution, min_roughness, spp, msaa), rast)]

    # Clear to background layer
    if background is not None:
        assert background.shape[1] == resolution and background.shape[2] == resolution
        if spp > 1:
            background = scale_img_nhwc(background, [full_res, full_res], mag='nearest', min='nearest')
        accum_col = background
    else:
        accum_col = torch.zeros(size=(1, full_res, full_res, 3), dtype=torch.float32, device='cuda')

    # Composite BACK-TO-FRONT
    for color, rast in reversed(layers):
        alpha     = (rast[..., -1:] > 0) * color[..., 3:4]
        accum_col = torch.lerp(accum_col, color[..., 0:3], alpha)
        if antialias:
            accum_col = dr.antialias(accum_col.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int()) # TODO: need to support bfloat16

    # Downscale to framebuffer resolution. Use avg pooling 
    out = avg_pool_nhwc(accum_col, spp) if spp > 1 else accum_col

    return out
