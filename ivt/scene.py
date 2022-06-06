import torch

class Scene:

    def __init__(self, ftype=torch.float32, itype=torch.long, device='cuda'):
        self.integrator = None 
        self.render_options = None 
        self.film = None 

        self.params = {}
        self.params['sensors'] = []
        self.params['meshes'] = []
        self.params['bsdfs'] = []
        self.params['emitters'] = []

        self.ftype = ftype
        self.itype = itype
        self.device = device

    def to_ftensor(self, array):
        if array is None: return None 
        
        if torch.is_tensor(array):
            return array.to(self.ftype).to(self.device)
        else:
            return torch.tensor(array, dtype=self.ftype, device=self.device)
        
    def to_itensor(self, array):
        if array is None: return None 
        
        if torch.is_tensor(array):
            return array.to(self.itype).to(self.device)
        else:
            return torch.tensor(array, dtype=self.itype, device=self.device)
        
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
        sensor = {
            'type': type,
            'fov': self.to_ftensor(fov),
            'origin': self.to_ftensor(origin),
            'target': self.to_ftensor(target),
            'up': self.to_ftensor(up)
        }
        self.params['sensors'].append(sensor)

    def add_mesh(self, vertex_positions, vertex_indices, bsdf_id, vertex_normals=None, uv_positions=None, uv_indices=None, to_world=torch.eye(4)):
        mesh = {
            'vertex_positions': self.to_ftensor(vertex_positions),
            'vertex_indices': self.to_itensor(vertex_indices),
            'vertex_normals': self.to_ftensor(vertex_normals),
            'uv_positions': self.to_ftensor(uv_positions),
            'uv_indices': self.to_ftensor(uv_indices),
            'to_world': self.to_ftensor(to_world),
            'bsdf_id': bsdf_id,
        }
        self.params['meshes'].append(mesh)

    def add_diffuse(self, reflectance):
        bsdf = {
            'type': 'diffuse',
            'reflectance': self.to_ftensor(reflectance)
        }
        self.params['bsdfs'].append(bsdf)
    
    def add_area_light(self, mesh_id, radiance):
        emitter = {
            'type': 'area',
            'mesh_id': mesh_id,
            'radiance': self.to_ftensor(radiance)
        }
        self.params['emitters'].append(emitter)

    def requires_grad(self, param_type, id, prop, req=True):
        data = self.params[param_type][id][prop]
        assert torch.is_tensor(data)
        data.requires_grad = req

    def __repr__(self):
        s = f'integrator: {self.integrator}\n'
        s += f'render_options: {self.render_options}\n'
        s += f'flim: {self.film}\n'
        s += f'params: {self.params}'

        return s

    def validate(self):
        pass
    