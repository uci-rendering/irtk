from collections import OrderedDict
import torch
import numpy as np

class Parameter:
    """
    A class to store parameters using different backends such as torch and numpy.
    It tracks requires_grad even if the backend doesn't support autodiff. 
    """
    
    def __init__(self, data, backend='torch', dtype=torch.float32, device='cpu', is_float=True):
        
        self.data = data
        self.backend = backend
        self.dtype = dtype
        self.device = device
        self.is_float = is_float
        self.requires_grad = False
        self.updated = False
        
        self.set(data)

    def configure(self):
        assert self.backend in ['torch', 'numpy']
        # if torch.is_tensor(self.data):
            # self.data = self.data.to(self.dtype).to(self.device)
        # else:
            # self.data = torch.tensor(self.data, dtype=self.dtype, device=self.device)

        # if self.data.requires_grad:
        #     self.requires_grad = True
        # else:
        #     self.data.requires_grad = self.requires_grad

        # self.data = torch.tensor(self.data, dtype=torch.float, device=self.device)
        if self.backend == 'torch':
            if torch.is_tensor(self.data):
                self.data = self.data.to(self.dtype).to(self.device)
            else:
                self.data = torch.tensor(self.data, dtype=self.dtype, device=self.device)

            if self.data.requires_grad:
                self.requires_grad = True
            else:
                self.data.requires_grad = self.requires_grad
                
        elif self.backend == 'numpy':
            # if torch.is_tensor(self.data):
            #     self.data = self.data.detach().cpu()
            # self.data = np.array(self.data, dtype=self.dtype)
            self.device = 'cpu'
    
    def numpy(self):
        return self.data.detach().cpu().numpy().astype(dtype=self.dtype)

    def set(self, data):
        self.data = data
        self.updated = True
        self.configure()

    def set_requires_grad(self, b=True):
        self.requires_grad = b
        if self.backend == 'torch':
            self.data.requires_grad = b

    def tolist(self):
        if self.backend == 'torch' or self.backend == 'numpy':
            return self.data.tolist()
        else:
            assert False

    def item(self):
        if self.backend == 'torch' or self.backend == 'numpy':
            return self.data.item()
        else:
            assert False
            
    def __repr__(self):
        return repr(self.data)

class Scene:

    def __init__(self, backend='torch', device='cpu', ftype=None, itype=None):
        assert backend in ['torch', 'numpy']
        
        # Scene data 
        self.integrators = [] 
        self.film = None 
        self.sensors = []
        self.meshes = []
        self.bsdfs = []
        self.emitters = []
        self.phases= []
        self.mediums = []

        # Cached scenes
        self.cached = {}
        
        # A map from all parameter names to their corresponding parameters.
        self.param_map = OrderedDict()

        self.backend = backend
        self.device = device
        
        if backend == 'numpy': device = 'cpu'
        
        if ftype is None:
            if backend == 'torch':
                self.ftype = torch.float32
            elif backend == 'numpy':
                self.ftype = np.float64
        
        if itype is None:
            if backend == 'torch':
                self.itype = torch.long
            elif backend == 'numpy':
                self.itype = np.int64
            
    def add_iparam(self, param_name, array):
        param = Parameter(array, self.backend, self.itype, self.device, is_float=False)
        self.param_map[param_name] = param 
        return param
    
    def add_fparam(self, param_name, array):
        param = Parameter(array, self.backend, self.ftype, self.device)
        self.param_map[param_name] = param 
        return param
        
    def add_integrator(self, integrator_type, integrator_params={}):
        integrator = {
            'type': integrator_type,
            'props': integrator_params
        }
        # integrator = integrator_type
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

    def add_mesh(self, vertex_positions, vertex_indices, bsdf_id, med_int_id = None, med_ext_id = None,
                 uv_positions=[], uv_indices=[], to_world=torch.eye(4), use_face_normal=False):
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
        if med_int_id is not None:
            mesh.update({'med_int_id' : med_int_id})
        if med_ext_id is not None:
            mesh.update({'med_ext_id' : med_ext_id})

        self.meshes.append(mesh)

    def add_diffuse_bsdf(self, reflectance):
        id = f'bsdfs[{len(self.bsdfs)}]'
        bsdf = {
            'id': id,
            'type': 'diffuse',
            'reflectance': self.add_fparam(id + '.reflectance', reflectance)
        }
        self.bsdfs.append(bsdf)

    def add_microfacet_bsdf(self, diffuse_reflectance, specular_reflectance, roughness):
        id = f'bsdfs[{len(self.bsdfs)}]'
        bsdf = {
            'id': id,
            'type': 'microfacet',
            'diffuse_reflectance': self.add_fparam(id + '.diffuse_reflectance', diffuse_reflectance),
            'specular_reflectance': self.add_fparam(id + '.specular_reflectance', specular_reflectance),
            'roughness': self.add_fparam(id + '.roughness', roughness),
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
    
    def add_isotropic_phase(self):
        i = len(self.phases)
        id = f'phases[{i}]'
        phase = {
            'id': id,
            'type': 'isotropic'
        }
        self.phases.append(phase)
        return i
    
    def add_volume(self, id, volume, name):
        assert name in ['sigmaT', 'albedo']
        if type(volume) == dict:
            ret =  {
                'type': 'gridvolume',
                'data': self.add_fparam(id + '.' + name, volume['data']),
                'res': volume['res'],
                'nchannel': volume['nchannel'],
                'min': volume['min'],
                'max': volume['max']
            }
            if 'to_world' in volume:
                ret.update({'to_world': self.add_fparam(
                    id + '.to_world', volume['to_world'])})
            return ret
        elif type(volume) == list:
            return {
                'type': 'constvolume',
                'data': self.add_fparam(id + '.' + name, np.array(volume)),
            }                   
        elif type(volume) == float:
            return {
                'type': 'constvolume',
                'data': self.add_fparam(id + '.' + name, np.array([volume]*3)),
            }
        else:
            raise ValueError('Unknown volume type')
    
    def add_homogeneous_medium(self, sigmaT, albedo, phase_id=None):
        id = f'mediums[{len(self.mediums)}]'
        medium = {
            'id': id,
            'type': 'homogeneous',
            'sigmaT': self.add_fparam(id + '.sigmaT', sigmaT),
            # 'albedo': self.add_fparam(id + '.albedo', albedo),
            'albedo': self.add_volume(id, albedo, 'albedo'),
            'phase_id': phase_id if phase_id is not None else self.add_isotropic_phase()
        }
        self.mediums.append(medium)
    
    def add_heterogeneous_medium(self, sigmaT, albedo, scale= 1., phase_id=None):
        id = f'mediums[{len(self.mediums)}]'
        medium = {
            'id': id,
            'type': 'heterogeneous',
            'sigmaT': self.add_volume(id, sigmaT, 'sigmaT'),
            'albedo': self.add_volume(id, albedo, 'albedo'),
            'scale': self.add_fparam(id + '.scale', scale),
            'phase_id': phase_id if phase_id is not None else self.add_isotropic_phase()
        }
        self.mediums.append(medium)

    def add_env_light(self, env_map, to_world=torch.eye(4)):
        id = f'emitters[{len(self.emitters)}]'
        emitter = {
            'id': id,
            'type': 'env',
            'env_map': self.add_fparam(id + '.env_map', env_map),
            'to_world': self.add_fparam(id + '.to_world', to_world)
        }
        self.emitters.append(emitter)
        
    def configure(self):
        for param_name in self.param_map:
            param = self.param_map[param_name]
            param.backend = self.backend
            param.dtype = self.ftype if param.is_float else self.itype
            param.device = self.device
            param.configure()
            
    def get_requiring_grad(self):
        return [param_name for param_name in self.param_map if self.param_map[param_name].requires_grad]

    def get_updated(self):
        return [param_name for param_name in self.param_map if self.param_map[param_name].updated]

    def __repr__(self):
        s = '\n'.join([f'{param_name}: {self.param_map[param_name]}' for param_name in self.param_map])

        return s

    def validate(self):
        pass
    
def split_param_name(param_name):
    group, idx, prop = param_name.replace('[', '.').replace(']', '').split('.')
    idx = int(idx)
    return group, idx, prop