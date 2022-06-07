from collections import OrderedDict
import torch
import numpy as np

class Parameter:
    
    def __init__(self, data, backend='torch', dtype=torch.float32, device='cpu', is_float=True):
        
        self.data = data
        self.backend = backend
        self.dtype = dtype
        self.device = device
        self.is_float = is_float
        self.requires_grad = False
        
        if torch.is_tensor(data):
            self.requires_grad = data.requires_grad
        
        self.configure()
            
    def configure(self):
        assert self.backend in ['torch', 'numpy']
        
        if self.backend == 'torch':
            if torch.is_tensor(self.data):
                self.data = self.data.to(self.dtype).to(self.device)
                self.data.requires_grad = self.requires_grad
            else:
                self.data = torch.tensor(self.data, dtype=self.dtype, device=self.device)
                
        elif self.backend == 'numpy':
            if torch.is_tensor(self.data):
                self.data = self.data.detach().cpu()
            self.data = np.array(self.data, dtype=self.dtype)
            self.device = 'cpu'
            
    def set(self, data):
        self.data = data
        self.configure()
            
    def __repr__(self):
        return repr(self.data)

class Scene:

    def __init__(self, backend='torch', device='cpu', ftype=None, itype=None):
        assert backend in ['torch', 'numpy']
        
        self.integrator = None 
        self.render_options = None 
        self.film = None 
        self.sensors = []
        self.meshes = []
        self.bsdfs = []
        self.emitters = []
        
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
        
    def add_integrator(self, type):
        self.integrator = {
            'type': type
        }

    def add_render_options(self, options):
        self.render_options = options

    def add_hdr_film(self, resolution, rfilter='tent', crop=(0, 0, 1, 1)):
        self.film = {
            'type': 'hdr',
            'resolution': resolution,
            'rfilter': 'tent',
            'crop': crop
        }

    def add_perspective_camera(self, fov, origin, target, up):
        id = f'sensors[{len(self.sensors)}]'
        sensor = {
            'type': type,
            'fov': self.add_fparam(id + '.fov', fov),
            'origin': self.add_fparam(id + '.origin', origin),
            'target': self.add_fparam(id + '.target', target),
            'up': self.add_fparam(id + '.up', up)
        }
        self.sensors.append(sensor)

    def add_mesh(self, vertex_positions, vertex_indices, bsdf_id, vertex_normals=[], uv_positions=[], uv_indices=[], to_world=torch.eye(4)):
        id = f'meshes[{len(self.meshes)}]'
        mesh = {
            'vertex_positions': self.add_fparam(id + '.vertex_positions', vertex_positions),
            'vertex_indices': self.add_iparam(id + '.vertex_indices', vertex_indices),
            'vertex_normals': self.add_fparam(id + '.vertex_normals', vertex_normals),
            'uv_positions': self.add_fparam(id + '.uv_positions', uv_positions),
            'uv_indices': self.add_iparam(id + '.uv_indices', uv_indices),
            'to_world': self.add_fparam(id + '.to_world', to_world),
            'bsdf_id': bsdf_id,
        }
        self.meshes.append(mesh)

    def add_diffuse_bsdf(self, reflectance):
        id = f'bsdfs[{len(self.bsdfs)}]'
        bsdf = {
            'type': 'diffuse',
            'reflectance': self.add_fparam(id + '.reflectance', reflectance)
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
            'type': 'area',
            'mesh_id': mesh_id,
            'radiance': self.add_fparam(id + '.radiance', radiance)
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

    def __repr__(self):
        s = '\n'.join([f'{param_name}: {self.param_map[param_name]}' for param_name in self.param_map])

        return s

    def validate(self):
        pass
    