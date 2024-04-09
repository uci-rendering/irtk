from ..connector import Connector
from ..scene import *
from ..config import *
from ..io import *
from ..utils import Timer

import drjit as dr
import mitsuba as mi
import torch

from collections import OrderedDict

mi.set_variant('cuda_ad_rgb')
# mi.set_log_level(mi.LogLevel.Debug)
# mi.register_bsdf("MitsubaMicrofacetBSDF", lambda props: MitsubaMicrofacetBSDF(props))

class MitsubaConnector(Connector, connector_name='mitsuba'):

    debug = False

    def __init__(self):
        super().__init__()
        
        self.default_render_options = {
            'spp': 64,
            'seed': 0
        }

    def update_scene_objects(self, scene, render_options):
        if 'mitsuba' in scene.cached:
            cache = scene.cached['mitsuba']
        else:
            cache = {}
            scene.cached['mitsuba'] = cache
            cache['name_map'] = {}
            cache['integrators'] = []
            cache['sensors'] = []
            cache['film'] = None
            cache['initialized'] = False
            
        for k in self.default_render_options:
            if k not in render_options:
                render_options[k] = self.default_render_options[k]
        
        mi_params = []
        for name in scene.components:
            component = scene[name]
            mi_params += self.extensions[type(component)](name, scene)

        if not cache['initialized']:
            assert cache['film']
            assert len(cache['integrators']) > 0
            assert len(cache['sensors']) > 0

            scene_dict = {'type': 'scene'}
            for name in cache['name_map']:
                scene_dict[name] = cache['name_map'][name]
            cache['scene'] = mi.load_dict(scene_dict)

            for i in range(len(cache['sensors'])):
                cache['sensors'][i]['film'] = cache['film']
                cache['sensors'][i] = mi.load_dict(cache['sensors'][i])

            cache['initialized'] = True

        return cache, mi_params

    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        with Timer(f"-- Prepare Scene", prt=self.debug, record=False):
            cache, _ = self.update_scene_objects(scene, render_options)

            mi_scene = cache['scene']
            mi_sensors = cache['sensors']
            mi_integrator = cache['integrators'][integrator_id]

        with Timer('-- Backend Forward', prt=self.debug, record=False):
            images = []
            seed = render_options['seed']
            spp = render_options['spp']
            
            for sensor_id in sensor_ids:
                image = mi_integrator.render(mi_scene, sensor=mi_sensors[sensor_id], seed=seed, spp=spp).torch()
                image = to_torch_f(image)
                images.append(image)

        return images
        
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        with Timer(f"-- Prepare Scene", prt=self.debug, record=False):
            cache, mi_params = self.update_scene_objects(scene, render_options)
        
            mi_scene = cache['scene']
            mi_sensors = cache['sensors']
            mi_integrator = cache['integrators'][integrator_id]

            param_grads = [torch.zeros_like(scene[param_name]) for param_name in scene.requiring_grad]

        with Timer('-- Backend Backward', prt=self.debug, record=False):
            for i, sensor_id in enumerate(sensor_ids):
                seed = render_options['seed']
                spp = render_options['spp']
                
                image_grad = mi.TensorXf(image_grads[i])
                mi_integrator.render_backward(mi_scene, mi_params, image_grad, sensor=mi_sensors[sensor_id], seed=seed, spp=spp)
                for param_grad, mi_param in zip(param_grads, mi_params):
                    grad = to_torch_f(dr.grad(mi_param).torch())
                    grad = torch.nan_to_num(grad).reshape(param_grad.shape)
                    param_grad += grad
                        
            return param_grads

@MitsubaConnector.register(Integrator)
def process_integrator(name, scene):
    integrator = scene[name]
    cache = scene.cached['mitsuba']

    if not cache['initialized']:
        mi_integrator = mi.load_dict({
            'type': integrator['type'],
            **integrator['config']
        })
        cache['integrators'].append(mi_integrator)
        cache['name_map'][name] = mi_integrator

    return []

@MitsubaConnector.register(HDRFilm)
def process_hdr_film(name, scene):
    film = scene[name]
    cache = scene.cached['mitsuba']

    if not cache['initialized']:
        mi_film = mi.load_dict({
            'type': 'hdrfilm',
            'width': film['width'],
            'height': film['height'],
            'sample_border': True,
            'pixel_format': 'rgb',
            'component_format': 'float32',
            'filter': {
                'type': 'box'
            }
        })

        cache['film'] = mi_film

    return []

@MitsubaConnector.register(PerspectiveCamera)
def process_perspective_camera(name, scene):
    sensor = scene[name]
    cache = scene.cached['mitsuba']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        mi_sensor_dict = {
            'type': 'perspective',
            'near_clip': sensor['near'],
            'far_clip': sensor['far'],
            'fov': sensor['fov'],
            'to_world': mi.ScalarTransform4f(to_numpy(sensor['to_world']))
        }
        cache['sensors'].append(mi_sensor_dict)

    return []

@MitsubaConnector.register(Mesh)
def process_mesh(name, scene):
    mesh = scene[name]
    cache = scene.cached['mitsuba']

    # Since we use mi.Mesh which is not well supported, we need to use 
    # some unusual tricks. 
    def helper():
        '''
        Mitsuba doesn't provide a to_world matrix for differentiation. So
        we manually multiply the to_world matrix and track the gradient. 
        Also, when passing texture coordinates, Mitsuba requires the 
        size of textures coordinates matches the size of vertex positions,
        so we need seperate the mesh into triangle faces and recompute all 
        the mesh properties. Under this scenario, the vertex normals computed
        by Mitsuba will be the same as the face normals, which is not desired
        and we need to compute the vertex normals ourselves. 
        '''
        mi_v = tensor_f_to_mi(mesh['v'], mi.Point3f)
        mi_f = tensor_i_to_mi(mesh['f'])
        mi_uv = tensor_f_to_mi(mesh['uv'], mi.Point2f)
        mi_fuv = tensor_i_to_mi(mesh['fuv'])
        mi_to_world = tensor_f_to_mi(mesh['to_world'], mi.Matrix4f)

        if mesh['v'].requires_grad:
            dr.enable_grad(mi_v)
        if mesh['to_world'].requires_grad:
            dr.enable_grad(mi_to_world)

        mesh_info = {}
        mesh_info['v'] = mi_v
        mesh_info['f'] = mi_f
        mesh_info['to_world'] = mi_to_world
        
        if mesh['can_change_topology']:
            mi_v_new = mi.Transform4f(mi_to_world) @ mi_v
            mi_f_new = mi_f

            # Let Mitsuba computes the vertex normals
            if not mesh['use_face_normal']:
                mesh_info['vn_new'] = dr.zeros(mi.Float, dr.shape(mi_v)[1])
        else:
            # Turn the mesh into separate faces
            mi_v_new = dr.gather(mi.Point3f, mi_v, mi_f)
            mi_v_new = mi.Transform4f(mi_to_world) @ mi_v_new
            mi_f_new = dr.arange(mi.UInt, 0, dr.shape(mi_f)[0])
            mesh_info['uv_new'] = dr.gather(mi.Point2f, mi_uv, mi_fuv)

            # Compute vertex normals
            if not mesh['use_face_normal']:
                # Todo: this should be updated to match Mitsuba's implementation
                idx = dr.arange(mi.UInt, 0, dr.shape(mi_f)[0], 3)
                mi_v0 = dr.gather(mi.Point3f, mi_v_new, idx + 0)
                mi_v1 = dr.gather(mi.Point3f, mi_v_new, idx + 1)
                mi_v2 = dr.gather(mi.Point3f, mi_v_new, idx + 2)

                mi_fn = dr.normalize(dr.cross(mi_v2 - mi_v1, mi_v0 - mi_v1))
                mi_vn = dr.zeros(mi.Normal3f, dr.shape(mi_v)[1])
                dr.scatter_reduce(dr.ReduceOp.Add, mi_vn, mi_fn, dr.gather(mi.UInt, mi_f, idx + 0))
                dr.scatter_reduce(dr.ReduceOp.Add, mi_vn, mi_fn, dr.gather(mi.UInt, mi_f, idx + 1))
                dr.scatter_reduce(dr.ReduceOp.Add, mi_vn, mi_fn, dr.gather(mi.UInt, mi_f, idx + 2))

                mi_vn = dr.normalize(mi_vn)
                mi_vn_new = dr.gather(mi.Normal3f, mi_vn, mi_f)
                mesh_info['vn_new'] = mi_vn_new
        
        mesh_info['v_new'] = mi_v_new
        mesh_info['f_new'] = mi_f_new
        
        return mesh_info

    # Create the object if it has not been created
    if name not in cache['name_map']:
        props = mi.Properties()

        # Create its material first
        if 'mat_id' in mesh:
            mat_id = mesh['mat_id']
            if mat_id not in scene:
                raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
            bsdf = scene[mat_id]
            MitsubaConnector.extensions[type(bsdf)](mat_id, scene)
            mi_bsdf = cache['name_map'][mat_id] # its material
            props["bsdf"] = mi_bsdf

        # Create area light is used as an emitter
        # TODO: In mitsuba 3.5.0, there is a bug that prevent creating an area 
        #       light this way, but it is fixed in: 
        #       https://github.com/mitsuba-renderer/mitsuba3/pull/1096
        #       You will need to build the master branch of mitsuba until they
        #       release a new version. 
        if 'radiance' in mesh:
            radiance = convert_color(mesh['radiance'], return_dict=True)
            mi_area_light = mi.load_dict({
                'type': 'area',
                'radiance': radiance
            })
            props["emitter"] = mi_area_light

        mi_mesh = mi.Mesh(name, 0, 0, props=props) # placeholder mesh
        mi_mesh_params = mi.traverse(mi_mesh)

        mesh_info = helper()
        
        mi_mesh_params['vertex_positions'] = dr.ravel(mesh_info['v_new'])
        mi_mesh_params['faces'] = mesh_info['f_new']

        # If vertex_positions is updated, mitsuba will recompute the vertex normals.
        # So we need to update them first before loading the vertex normals we computed,
        # or the normals will be overrided. 
        mi_mesh_params.update()

        if 'uv_new' in mesh_info: 
            mi_mesh_params['vertex_texcoords'] = dr.ravel(mesh_info['uv_new'])

        if 'vn_new' in mesh_info:
            mi_mesh_params['vertex_normals'] = dr.ravel(mesh_info['vn_new'])

        mi_mesh_params.update()
        cache['name_map'][name] = mi_mesh

    # mi.Mesh is a bit buggy and we cannot update the mesh using mi_mesh directly.
    # We instead use the scene parameters to update the mesh.

    updated = mesh.get_updated()
    requiring_grad = mesh.get_requiring_grad()
    mi_diff_params, add_param = gen_add_param()

    if 'scene' in cache and (updated or requiring_grad):
        mesh_info = helper()

        mi_params = mi.traverse(cache['scene'])

        mesh_update_needed = 'v' in updated or 'to_world' in updated
        mesh_update_needed |= 'v' in requiring_grad or 'to_world' in requiring_grad

        if mesh_update_needed:
            mi_params[f'{name}.vertex_positions'] = dr.ravel(mesh_info['v_new'])
            mi_params[f'{name}.faces'] = mesh_info['f_new']
            # Update in advance to avoid overriding the vertex normals
            mi_params.update()
            if 'vn_new' in mesh_info:
                mi_params[f'{name}.vertex_normals'] = dr.ravel(mesh_info['vn_new'])
                mi_params.update()

        # Update parameters
        for param_name in updated:
            if param_name == 'radiance':
                radiance, radiance_t = convert_color(mesh['radiance'])
                mi_params[f'{name}.emitter.radiance.{radiance_t}'] = radiance
            mesh.mark_updated(param_name, False)
        mi_params.update()
        
        # Enable grad for parameters requiring grad
        for param_name in requiring_grad:
            if param_name == 'v':
                add_param(mesh_info['v'])
            elif param_name == 'to_world':
                add_param(mesh_info['to_world'])
            elif param_name == 'radiance':
                radiance, radiance_t = convert_color(mesh['radiance'])
                add_param(mi_params[f'{name}.emitter.radiance.{radiance_t}'])

    return mi_diff_params

@MitsubaConnector.register(DiffuseBRDF)
def process_diffuse_brdf(name, scene):
    bsdf = scene[name]
    cache = scene.cached['mitsuba']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        reflectance = convert_color(bsdf['d'], return_dict=True)

        mi_bsdf = mi.load_dict({
            'type': 'diffuse',
            'reflectance': reflectance
        })
        cache['name_map'][name] = mi_bsdf

    mi_bsdf = cache['name_map'][name]
    mi_bsdf_params = mi.traverse(mi_bsdf)

    # Update parameters
    updated = bsdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                kd, kd_t = convert_color(bsdf['d'])
                mi_bsdf_params[f'reflectance.{kd_t}'] = kd
            bsdf.mark_updated(param_name, False)
        mi_bsdf_params.update()

    # Enable grad for parameters requiring grad
    mi_diff_params, add_param = gen_add_param()

    requiring_grad = bsdf.get_requiring_grad()
    for param_name in requiring_grad:
        if param_name == 'd':
            kd, kd_t = convert_color(bsdf['d'])
            add_param(mi_bsdf_params[f'reflectance.{kd_t}'])

    return mi_diff_params

@MitsubaConnector.register(EnvironmentLight)
def process_environment_light(name, scene):
    emitter = scene[name]
    cache = scene.cached['mitsuba']

    def helper():
        '''
        Mitsuba duplicates the first column of the envmap and append it 
        after the last column. So if the original shape of the envmap is
        (h, w, c), dr.shape(params['envmap.data']) == (h, w + 1, c). We 
        manually copy the column here to ensure gradient can be correctly
        propogated. 
        '''
        assert emitter['radiance'].dim() == 3
        r = mi.TensorXf(emitter['radiance'])
        
        if emitter['radiance'].requires_grad:
            dr.enable_grad(r)

        h, w, c = dr.shape(r)
        r_new = dr.zeros(mi.TensorXf, shape=(h, w + 1, c))

        hh, ww, cc = dr.meshgrid(dr.arange(mi.UInt, h), dr.arange(mi.UInt, w), dr.arange(mi.UInt, c), indexing='ij')
        idx = hh * (w + 1) * c + ww * c + cc
        dr.scatter(r_new.array, r.array, idx)

        hh, cc = dr.meshgrid(dr.arange(mi.UInt, h), dr.arange(mi.UInt, c), indexing='ij')
        idx = hh * (w + 1) * c + w * c + cc
        dr.scatter(r_new.array, r[:, 0].array, idx)

        emitter_info = {}
        emitter_info['r'] = r
        emitter_info['r_new'] = r_new
        return emitter_info
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        mi_emitter = mi.load_dict({
            'type': 'envmap',
            'bitmap': mi.Bitmap(mi.TensorXf(emitter['radiance'])),
            'to_world': mi.ScalarTransform4f(to_numpy(emitter['to_world'])),
        })
        cache['name_map'][name] = mi_emitter

    mi_emitter = cache['name_map'][name]
    mi_emitter_params = mi.traverse(mi_emitter)

    updated = emitter.get_updated()
    requiring_grad = emitter.get_requiring_grad()
    mi_diff_params, add_param = gen_add_param()

    if updated or requiring_grad:
        emitter_info = helper()

        if 'radiance' in updated or 'radiance' in requiring_grad:
            mi_emitter_params['data'] = emitter_info['r_new']
            mi_emitter_params.update()

        # Update parameters
        for param_name in updated:
            emitter.mark_updated(param_name, False)
        
        # Enable grad for parameters requiring grad
        for param_name in requiring_grad:
            if param_name == 'radiance':
                add_param(emitter_info['r'])

    return mi_diff_params

#----------------------------------------------------------------------------
# Helpers.

def tensor_f_to_mi(tensor_f, mi_type=None):
    array_f = mi.TensorXf(tensor_f).array
    return dr.unravel(mi_type, array_f) if mi_type else array_f

def tensor_i_to_mi(tensor_i, mi_type=None):
    array_i = mi.TensorXi(tensor_i).array
    return dr.unravel(mi_type, array_i) if mi_type else array_i

def gen_add_param():
    mi_diff_params = []
    def add_param(p):
        dr.enable_grad(p)
        mi_diff_params.append(p)
    return mi_diff_params, add_param

def convert_color(color, return_dict=False):
    if color.dim() == 1:
        if return_dict:
            return {
                'type': 'rgb',
                'value': to_numpy(color)
            }
        else: return tensor_f_to_mi(color), 'value'
    else:
        assert color.dim() == 3
        color = mi.TensorXf(color)
        if return_dict:
            return {
                'type': 'bitmap',
                'bitmap': mi.Bitmap(color)
            }
        else: return color, 'data'