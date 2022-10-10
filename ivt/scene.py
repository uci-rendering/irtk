import torch
from .parameter import Parameter, NaiveParameter
from .io import read_obj

class Scene:
    def __init__(self, device='cuda', ftype=torch.float32, itype=torch.long):
        # Scene data 
        self.integrators = [] 
        self.film = None 
        self.sensors = []
        self.meshes = []
        self.bsdfs = []
        self.emitters = []

        self._alias = {}
        self._diff_param_names = [] # names of all differentiable parameters

        self.cached = {} # Cached scenes

        self.device = device
        self.ftype = ftype
        self.itype = itype

    def add_alias(self, param_name, alias):
        self._alias[alias] = param_name
    
    def __getitem__(self, param_name):
        if param_name in self._alias:
            param_name = self._alias[param_name]
        group, idx, prop = split_param_name(param_name)
        return getattr(self, group)[idx][prop]

    def __setitem__(self, param_name, param):
        if param_name in self._alias:
            param_name = self._alias[param_name]
        group, idx, prop = split_param_name(param_name)
        getattr(self, group)[idx][prop] = param

    def __make_param(self, param_data, dtype):
        if issubclass(type(param_data), Parameter):
            param = param_data
        else:
            param = NaiveParameter(param_data, dtype, self.device)
        return param
        
    def __make_iparam(self, param_data):
        return self.__make_param(param_data, self.itype)
    
    def __make_fparam(self, param_data):
        return self.__make_param(param_data, self.ftype)

    def __make_id(self, group_name):
        return f'{group_name}[{len(getattr(self, group_name))}]'

    def __mark_diff(self, id, props):
        for prop in props:
            param_name = id + f'.{prop}'
            self._diff_param_names.append(param_name)
        
    def add_integrator(self, integrator_type, integrator_params={}):
        id = self.__make_id('integrators')
        integrator = {
            'id': id,
            'type': integrator_type,
            'params': integrator_params
        }
        self.integrators.append(integrator)

    def add_hdr_film(self, resolution, rfilter='tent', crop=(0, 0, 1, 1)):
        self.film = {
            'type': 'hdrfilm',
            'resolution': resolution,
            'rfilter': rfilter,
            'crop': crop
        }

    def add_perspective_camera(self, fov, origin=(1, 0, 0), target=(0, 0, 0), up=(0, 1, 0), use_to_world=False, to_world=torch.eye(4)):
        id = self.__make_id('sensors')
        sensor = {
            'id': id,
            'type': 'perspective',
            'fov': self.__make_fparam(fov)
        }
        if use_to_world:
            sensor['to_world'] = self.__make_fparam(to_world)
            self.__mark_diff(id, ['to_world'])
        else:
            sensor['origin'] = self.__make_fparam(origin)
            sensor['target'] = self.__make_fparam(target)
            sensor['up'] = self.__make_fparam(up)

        self.sensors.append(sensor)

    def add_mesh(self, vertex_positions, vertex_indices, bsdf_id, uv_positions=[], uv_indices=[], to_world=torch.eye(4), use_face_normal=False):
        id = self.__make_id('meshes')
        mesh = {
            'id': id,
            'vertex_positions': self.__make_fparam(vertex_positions),
            'vertex_indices': self.__make_iparam(vertex_indices),
            'uv_positions': self.__make_fparam(uv_positions),
            'uv_indices': self.__make_iparam(uv_indices),
            'to_world': self.__make_fparam(to_world),
            'bsdf_id': bsdf_id,
            'use_face_normal': use_face_normal
        }
        self.__mark_diff(id, ['vertex_positions', 'to_world'])
        self.meshes.append(mesh)

    def add_obj(self, obj_path, bsdf_id, to_world=torch.eye(4), use_face_normal=False):
        v, tc, _, f, ftc, _ = read_obj(obj_path)
        self.add_mesh(v, f, bsdf_id, tc, ftc, to_world, use_face_normal)

    def add_diffuse_bsdf(self, reflectance, to_world=torch.eye(3)):
        id = self.__make_id('bsdfs')
        bsdf = {
            'id': id,
            'type': 'diffuse',
            'reflectance': self.__make_fparam(reflectance),
            'to_world': self.__make_fparam(to_world)
        }
        self.__mark_diff(id, ['reflectance', 'to_world'])
        self.bsdfs.append(bsdf)

    def add_microfacet_bsdf(self, diffuse_reflectance, specular_reflectance, roughness, to_world=torch.eye(3)):
        id = self.__make_id('bsdfs')
        bsdf = {
            'id': id,
            'type': 'microfacet',
            'diffuse_reflectance': self.__make_fparam(diffuse_reflectance),
            'specular_reflectance': self.__make_fparam(specular_reflectance),
            'roughness': self.__make_fparam(roughness),
            'to_world': self.__make_fparam(to_world)
        }
        self.__mark_diff(id, ['diffuse_reflectance', 'specular_reflectance', 'roughness', 'to_world'])
        self.bsdfs.append(bsdf)
    
    def add_area_light(self, mesh_id, radiance):
        id = self.__make_id('emitters')
        emitter = {
            'id': id, 
            'type': 'area',
            'mesh_id': mesh_id,
            'radiance': self.__make_fparam(radiance)
        }
        self.__mark_diff(id, ['radiance'])
        self.emitters.append(emitter)

    def add_env_light(self, env_map, to_world=torch.eye(4)):
        id = self.__make_id('emitters')
        emitter = {
            'id': id,
            'type': 'env',
            'env_map': self.__make_fparam(env_map),
            'to_world': self.__make_fparam(to_world)
        }
        self.__mark_diff(id, ['env_map', 'to_world'])
        self.emitters.append(emitter)
            
    def get_requiring_grad(self):
        return [param_name for param_name in self._diff_param_names if self[param_name].requires_grad]

    def clear_cache(self):
        self.cached = {}

def split_param_name(param_name):
    group_name, idx, prop = param_name.replace('[', '.').replace(']', '').split('.')
    idx = int(idx)
    return group_name, idx, prop