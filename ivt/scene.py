from collections import OrderedDict
import torch
import numpy as np
from abc import ABC, abstractmethod



class Parameter(ABC):
    @abstractmethod
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device
        self._requires_grad = False
        self._updated = False

    @property
    @abstractmethod
    def data(self):
        pass
    
    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, b_requires_grad):
        self._requires_grad = b_requires_grad

    def to_tensor(self, array):
        if torch.is_tensor(array):
            array = array.to(self.dtype).to(self.device)
        else:
            array = torch.tensor(array, dtype=self.dtype, device=self.device)
        return array

class DefaultParameter(Parameter):
    def __init__(self, dtype, device):
        super().__init__(dtype, device)
        self._raw_data = []

    @property
    def raw_data(self):
        return self._raw_data 

    @raw_data.setter
    def raw_data(self, raw_data):
        self._raw_data = self.to_tensor(raw_data)

    @property
    def data(self):
        return self._raw_data
    
    @Parameter.requires_grad.setter
    def requires_grad(self, requires_grad):
        self._requires_grad = requires_grad
        self._raw_data.requires_grad = requires_grad

class Scene:
    def __init__(self, device='cuda', ftype=torch.float32, itype=torch.long):
        # Scene data 
        self.integrators = [] 
        self.film = None 
        self.sensors = []
        self.meshes = []
        self.bsdfs = []
        self.emitters = []

        # Cached scenes
        self.cached = {}
        
        # A map from all parameter names to their corresponding parameters.
        self.param_map = OrderedDict()

        self.device = device
        self.ftype = ftype
        self.itype = itype
        
    def add_iparam(self, param_name, param_data):
        if issubclass(type(param_data), Parameter):
            param = param_data
        else:
            param = DefaultParameter(self.itype, self.device)
            param.raw_data = param_data
        self.param_map[param_name] = param 
        return param
    
    def add_fparam(self, param_name, param_data):
        if issubclass(type(param_data), Parameter):
            param = param_data
        else:
            param = DefaultParameter(self.ftype, self.device)
            param.raw_data = param_data
        self.param_map[param_name] = param 
        return param
        
    def add_integrator(self, integrator_type, integrator_params={}):
        integrator = {
            'type': integrator_type,
            'params': integrator_params
        }
        self.integrators.append(integrator)

    def add_hdr_film(self, resolution, rfilter='tent', crop=(0, 0, 1, 1)):
        self.film = {
            'type': 'hdrfilm',
            'resolution': resolution,
            'rfilter': 'tent',
            'crop': crop
        }

    def add_perspective_camera(self, fov, origin=(1, 0, 0), target=(0, 0, 0), up=(0, 1, 0), use_to_world=False, to_world=torch.eye(4)):
        id = f'sensors[{len(self.sensors)}]'
        sensor = {
            'type': 'perspective',
            'fov': self.add_fparam(id + '.fov', fov)
        }
        if use_to_world:
            sensor['to_world'] = self.add_fparam(id + '.to_world', to_world)
        else:
            sensor['origin'] = self.add_fparam(id + '.origin', origin)
            sensor['target'] = self.add_fparam(id + '.target', target)
            sensor['up'] = self.add_fparam(id + '.up', up)

        self.sensors.append(sensor)

    def add_mesh(self, vertex_positions, vertex_indices, bsdf_id, uv_positions=[], uv_indices=[], to_world=torch.eye(4), use_face_normal=False):
        id = f'meshes[{len(self.meshes)}]'
        mesh = {
            'id': id,
            'vertex_positions': self.add_fparam(id + '.vertex_positions', vertex_positions),
            'vertex_indices': self.add_iparam(id + '.vertex_indices', vertex_indices),
            'uv_positions': self.add_fparam(id + '.uv_positions', uv_positions),
            'uv_indices': self.add_iparam(id + '.uv_indices', uv_indices),
            'to_world': self.add_fparam(id + '.to_world', to_world),
            'bsdf_id': bsdf_id,
            'use_face_normal': use_face_normal
        }
        self.meshes.append(mesh)

    def add_diffuse_bsdf(self, reflectance, to_world=torch.eye(3)):
        id = f'bsdfs[{len(self.bsdfs)}]'
        bsdf = {
            'id': id,
            'type': 'diffuse',
            'reflectance': self.add_fparam(id + '.reflectance', reflectance),
            'to_world': self.add_fparam(id + '.to_world', to_world)
        }
        self.bsdfs.append(bsdf)

    def add_microfacet_bsdf(self, diffuse_reflectance, specular_reflectance, roughness, to_world=torch.eye(3)):
        id = f'bsdfs[{len(self.bsdfs)}]'
        bsdf = {
            'id': id,
            'type': 'microfacet',
            'diffuse_reflectance': self.add_fparam(id + '.diffuse_reflectance', diffuse_reflectance),
            'specular_reflectance': self.add_fparam(id + '.specular_reflectance', specular_reflectance),
            'roughness': self.add_fparam(id + '.roughness', roughness),
            'to_world': self.add_fparam(id + '.to_world', to_world)
        }
        self.bsdfs.append(bsdf)
        
    def add_null_bsdf(self):
        bsdf = {
            'type': 'null',
        }
        self.bsdfs.append(bsdf)
    
    def add_area_light(self, mesh_id, radiance):
        id = f'emitters[{len(self.emitters)}]'
        emitter = {
            'id': id,
            'type': 'area',
            'mesh_id': mesh_id,
            'radiance': self.add_fparam(id + '.radiance', radiance)
        }
        self.emitters.append(emitter)

    def add_env_light(self, env_map, to_world=torch.eye(4)):
        id = f'emitters[{len(self.emitters)}]'
        emitter = {
            'id': id,
            'type': 'env',
            'env_map': self.add_fparam(id + '.env_map', env_map),
            'to_world': self.add_fparam(id + '.to_world', to_world)
        }
        self.emitters.append(emitter)
            
    def get_requiring_grad(self):
        return [param_name for param_name in self.param_map if self.param_map[param_name].requires_grad]

    def get_updated(self):
        return [param_name for param_name in self.param_map if self.param_map[param_name].updated]

    def __repr__(self):
        s = '\n'.join([f'{param_name}: {self.param_map[param_name]}' for param_name in self.param_map])

        return s
    
def split_param_name(param_name):
    group, idx, prop = param_name.replace('[', '.').replace(']', '').split('.')
    idx = int(idx)
    return group, idx, prop