from .parameter import ParamGroup
from .transform import lookat
from .io import read_image, read_obj, to_torch_f, to_torch_i

from collections import OrderedDict

import torch

class Scene:

    def __init__(self) -> None:
        self.components = OrderedDict()
        self.requiring_grad = ()
        self.cached = {}

    def set(self, name, component):
        self.components[name] = component

    def __getitem__(self, name):
        item = self.components
        for c in name.split('.'):
            item = item[c]
        return item
    
    def __setitem__(self, name, param_value):
        component_name, param_name = name.rsplit('.', 1)
        self[component_name][param_name] = param_value
    
    def __contains__(self, name):
        item = self.components
        for c in name.split('.'):
            if c not in item: return False
            item = item[c]
        return True

    def configure(self):
        requiring_grad = []
        for cname in self.components:
            requiring_grad += [f'{cname}.{pname}' for pname in self.components[cname].get_requiring_grad()]
        self.requiring_grad = tuple(requiring_grad)

    def __str__(self):
        lines = []

        for name in self.components:
            lines.append(f"{name}:")
            lines.append(str(self.components[name]))
            lines.append("\n")

        return '\n'.join(lines)
    
    def clear_cache(self):
        self.cached = {}
        # Detach the tensors requiring grad
        for param_name in self.requiring_grad:
            self[param_name] = self[param_name].detach()

class Integrator(ParamGroup):

    def __init__(self, type, config):
        super().__init__()
        
        self.add_param('type', type, help_msg='integrator type')
        self.add_param('config', config, help_msg='integrator config')

class HDRFilm(ParamGroup):

    def __init__(self, width, height):
        super().__init__()
        
        self.add_param('width', width, help_msg='film width')
        self.add_param('height', height, help_msg='film height')

class PerspectiveCamera(ParamGroup):
    
    def __init__(self, fov, to_world, near=1e-6, far=1e7):
        super().__init__()

        self.add_param('fov', fov, help_msg='sensor fov')
        self.add_param('near', near, help_msg='sensor near clip')
        self.add_param('far', far, help_msg='sensor far clip')
        self.add_param('to_world', to_torch_f(to_world), is_tensor=True, is_diff=True, help_msg='sensor to_world matrix')

    @classmethod
    def from_lookat(cls, fov, origin, target, up, near=1e-6, far=1e7):
        sensor = cls(fov, torch.eye(4), near, far)
        origin = to_torch_f(origin)
        target = to_torch_f(target)
        up = to_torch_f(up)
        sensor['to_world'] = lookat(origin, target, up)
        return sensor
        
class Mesh(ParamGroup):

    def __init__(self, v, f, uv, fuv, mat_id, to_world=torch.eye(4), use_face_normal=True, radiance=torch.zeros(3)):
        super().__init__()
        
        self.add_param('v', to_torch_f(v), is_tensor=True, is_diff=True, help_msg='mesh vertex positions')
        self.add_param('f', to_torch_i(f), is_tensor=True, help_msg='mesh face indices')
        self.add_param('uv', to_torch_f(uv), is_tensor=True, help_msg='mesh uv coordinates')
        self.add_param('fuv', to_torch_i(fuv), is_tensor=True, help_msg='mesh uv face indices')
        self.add_param('mat_id', mat_id, help_msg='name of the material of the mesh')
        self.add_param('to_world', to_torch_f(to_world), is_tensor=True, help_msg='mesh to world matrix')
        self.add_param('use_face_normal', use_face_normal, help_msg='whether to use face normal')

        radiance = to_torch_f(radiance)
        is_emitter = radiance.sum() > 0
        self.add_param('is_emitter', is_emitter, help_msg='whether it is used as an emitter')
        self.add_param('radiance', radiance, is_tensor=True, is_diff=True, help_msg='radiance if it is used as an emitter')

    @classmethod
    def from_file(cls, filename, mat_id, to_world=torch.eye(4), use_face_normal=True, radiance=torch.zeros(3)):
        v, tc, n, f, ftc, fn = read_obj(filename)
        return cls(v, f, tc, ftc, mat_id, to_world, use_face_normal, radiance)
    
class DiffuseBRDF(ParamGroup):

    def __init__(self, d):
        super().__init__()
        
        self.add_param('d', to_torch_f(d), is_tensor=True, is_diff=True, help_msg='diffuse reflectance')

    @classmethod
    def from_file(cls, filename, is_srgb=None):
        texture = read_image(filename, is_srgb)
        return cls(texture)
    
class MicrofacetBRDF(ParamGroup):

    def __init__(self, d, s, r):
        super().__init__()
        
        self.add_param('d', to_torch_f(d), is_tensor=True, is_diff=True, help_msg='diffuse reflectance')
        self.add_param('s', to_torch_f(s), is_tensor=True, is_diff=True, help_msg='specular reflectance')
        self.add_param('r', to_torch_f(r), is_tensor=True, is_diff=True, help_msg='roughness')

    @classmethod
    def from_file(cls, d_filename, s_filename, r_filename, d_is_srgb=None, s_is_srgb=None, r_is_srgb=None):
        d_texture = read_image(d_filename, d_is_srgb)
        s_texture = read_image(s_filename, s_is_srgb)
        r_texture = read_image(r_filename, r_is_srgb)[..., 0:1]

        return cls(d_texture, s_texture, r_texture)
    
class EnvironmentLight(ParamGroup):

    def __init__(self, radiance):
        super().__init__()
        
        self.add_param('radiance', to_torch_f(radiance), is_tensor=True, is_diff=True, help_msg='environment light radiance')

    @classmethod
    def from_file(cls, radiance_filename, radiance_is_srgb=None):
        radiance_texture = read_image(radiance_filename, radiance_is_srgb)

        return cls(radiance_texture)