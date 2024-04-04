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
    mesh_alt = mesh.separate_faces()
    cache = scene.cached['mitsuba']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        # Create its material first
        mat_id = mesh_alt['mat_id']
        if mat_id not in scene:
            raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
        brdf = scene[mat_id]
        MitsubaConnector.extensions[type(brdf)](mat_id, scene)

        mi_bsdf = cache['name_map'][mat_id] # its material

        props = mi.Properties()
        props["bsdf"] = mi_bsdf
        props["face_normals"] = mesh_alt['use_face_normal']

        mi_mesh = mi.Mesh(name, 0, 0, props=props) # placeholder mesh
        mi_mesh_params = mi.traverse(mi_mesh)
        
        mi_v = tensor_f_to_mi(mesh_alt['v'], mi.Point3f)
        mi_transform = mi.Transform4f(tensor_f_to_mi(mesh_alt['to_world'], mi.Matrix4f))
        mi_mesh_params['vertex_positions'] = dr.ravel(mi_transform @ mi_v)
        mi_mesh_params['faces'] = tensor_i_to_mi(mesh_alt['f'])

        if 'uv' in mesh_alt: 
            mi_mesh_params['vertex_texcoords'] = tensor_f_to_mi(mesh_alt['uv'])

        if not mesh_alt['use_face_normal']:
            n = compute_vertex_normals(mesh_alt['v'], mesh_alt['f'])
            mi_n = tensor_f_to_mi(n, mi.Normal3f)
            mi_mesh_params['vertex_normals'] = dr.ravel(mi_transform @ mi_n)

        mi_mesh_params.update()

        cache['name_map'][name] = mi_mesh

    mi_mesh = cache['name_map'][name]
    mi_mesh_params = mi.traverse(mi_mesh)

    # Update parameters
    updated = mesh.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'v' or param_name == 'to_world':
                mi_v = tensor_f_to_mi(mesh_alt['v'], mi.Point3f)
                mi_to_world = tensor_f_to_mi(mesh_alt['to_world'], mi.Matrix4f)
                mi_transform = mi.Transform4f(mi_to_world)
                mi_mesh_params['vertex_positions'] = dr.ravel(mi_transform @ mi_v)
            elif param_name == 'f':
                mi_mesh_params['faces'] = tensor_i_to_mi(mesh_alt['f'])

        mi_mesh_params.update()

    # Enable grad for parameters requiring grad
    mi_params, add_param = gen_add_param()

    requiring_grad = mesh.get_requiring_grad()
    if len(requiring_grad) > 0:
        need_update = ('v' in requiring_grad) or ('to_world' in requiring_grad)

        if need_update:
            mi_v = tensor_f_to_mi(mesh_alt['v'], mi.Point3f)
            mi_to_world = tensor_f_to_mi(mesh_alt['to_world'], mi.Matrix4f)

        for param_name in requiring_grad:
            if param_name == 'v':
                add_param(mi_v)
            elif param_name == 'to_world':
                add_param(mi_to_world)

        # Need to update vertex_position to enable gradient
        if need_update:
            mi_transform = mi.Transform4f(mi_to_world)
            mi_mesh_params['vertex_positions'] = dr.ravel(mi_transform @ mi_v)
      
    return mi_params

@MitsubaConnector.register(DiffuseBRDF)
def process_diffuse_brdf(name, scene):
    bsdf = scene[name]
    cache = scene.cached['mitsuba']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        kd = convert_color(bsdf['d'], 3, bitmap=True)

        mi_bsdf = mi.load_dict({
            'type': 'diffuse',
            'reflectance': {
                'type': 'bitmap',
                'bitmap': kd
            }
        })
        cache['name_map'][name] = mi_bsdf

    mi_bsdf = cache['name_map'][name]
    mi_bsdf_params = mi.traverse(mi_bsdf)

    # Update parameters
    updated = bsdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                kd = convert_color(bsdf['d'], 3)
                mi_bsdf_params['reflectance.data'] = kd

        mi_bsdf_params.update()

    # Enable grad for parameters requiring grad
    mi_params, add_param = gen_add_param()

    requiring_grad = bsdf.get_requiring_grad()
    for param_name in requiring_grad:
        if param_name == 'd':
            add_param(mi_bsdf_params['reflectance.data'])

    return mi_params

@MitsubaConnector.register(EnvironmentLight)
def process_environment_light(name, scene):
    emitter = scene[name]
    cache = scene.cached['mitsuba']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        radiance = convert_color(emitter['radiance'], 3, bitmap=True)
        mi_emitter = {
            'type': 'envmap',
            'bitmap': radiance,
            'to_world': mi.ScalarTransform4f(to_numpy(emitter['to_world'])),
        }
        cache['name_map'][name] = mi_emitter

    return []

#----------------------------------------------------------------------------
# Helpers.

def tensor_f_to_mi(tensor_f, mi_type=None):
    array_f = mi.TensorXf(tensor_f).array
    return dr.unravel(mi_type, array_f) if mi_type else array_f

def tensor_i_to_mi(tensor_i, mi_type=None):
    array_i = mi.TensorXi(tensor_i).array
    return dr.unravel(mi_type, array_i) if mi_type else array_i

def gen_add_param():
    mi_params = []
    def add_param(p):
        dr.enable_grad(p)
        mi_params.append(p)
    return mi_params, add_param

def convert_color(color, c, bitmap=False):
    if color.shape == ():
        color = color.tile(c)
    if color.shape == (c,):
        color = color.reshape(1, 1, c)
        color = color.repeat(2, 2, 1) # Bitmap resolution must be at least 2 x 2
    assert color.dim() == 3
    color = mi.TensorXf(color)
    return mi.Bitmap(color) if bitmap else color

def compute_vertex_normals(verts, faces):
    """Computes the packed version of vertex normals from the packed verts
        and faces. This assumes verts are shared between faces. The normal for
        a vertex is computed as the sum of the normals of all the faces it is
        part of weighed by the face areas.
    """
    faces = faces.long()
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

def compute_texture_coordinates(verts, faces, uv, fuv, vflip=True):
    # Cut faces
    verts_new = verts[faces.long().flatten()]
    uvs_new = uv[fuv.long().flatten()]
    if vflip: uvs_new[:, 1] = 1 - uvs_new[:, 1]
    # Calculate the corresponding indices
    faces_new = torch.arange(verts_new.shape[0]).reshape(-1, 3)
    faces_new = to_torch_i(faces_new)

    return verts_new, faces_new, uvs_new