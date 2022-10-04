import torch
from abc import ABC, abstractmethod
from pathlib import Path
import sys
from .io import *
from .parameter import *
from .optimizers import LargeStepsOptimizer

class Model(ABC):
    def __init__(self, scene, dtype=torch.float32, device='cuda'):
        self.scene = scene
        self._dtype = dtype
        self._device = device

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def set_data(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_results(self):
        pass

    @abstractmethod
    def write_results(self, result_path):
        pass

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def get_regularization(self):
        return 0

class MultiOpt(Model):
    def __init__(self, scene, models):
        super().__init__(scene)

        self._models = models

    def zero_grad(self):
        for model in self._models:
            model.zero_grad()

    def set_data(self):
        for model in self._models:
            model.set_data()
        
    def step(self):
        for model in self._models:
            model.step()

    def get_results(self):
        results = []
        for model in self._models:
            results.append(model.get_results())
        return results

    def write_results(self, result_path):
        for model in self._models:
            model.write_results(result_path)

    def get_regularization(self):
        reg = 0
        for model in self._models:
            reg += model.get_regularization()
        return reg

class ModelFactory:

    @staticmethod
    def from_config(scene, config):
        name = config['model_name']
        args = config['args']
        model_class = getattr(sys.modules[__name__], name)
        model = model_class(scene, **args)
        return model 

    @staticmethod
    def from_configs(scene, configs):
        models = []
        for config in configs:
            models.append(ModelFactory.from_config(scene, config))
        model = MultiOpt(scene, models)
        return model

class LargeStepsShapeOpt(Model):

    def __init__(self, scene, mesh_id, init_mesh_path='', result_name='', optimizer_kwargs={}, dtype=torch.float32, device='cuda'):
        super().__init__(scene, dtype, device)

        mesh = self.scene.meshes[mesh_id]
        self._v = mesh['vertex_positions']
        self._f = mesh['vertex_indices']
        self._tc = mesh['uv_positions']
        self._ftc = mesh['uv_indices']

        assert isinstance(self._v, NaiveParameter), f'Vertex of mesh[{mesh_id}] must be an instance of NaiveParameter!'

        if init_mesh_path:
            v, tc, _, f, ftc, _ = read_obj(init_mesh_path)
            self._v.set(v)
            self._f.set(f)
            self._tc.set(tc)
            self._ftc.set(ftc)

        self._result_name = result_name if result_name else mesh['id']

        self._v.requires_grad = True

        self.optimizer = LargeStepsOptimizer(self._v.data, self._f.data, **optimizer_kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_data(self):
        return
        
    def step(self):
        self.optimizer.step()

    def get_results(self):
        results = {
            'v': self._v.data.detach(),
            'f': self._f.data,
            'tc': self._tc.data,
            'ftc': self._ftc.data,
        }
        return results

    def write_results(self, result_path):
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        write_obj(result_path / f'{self._result_name}.obj', results['v'], results['f'], results['tc'], results['ftc'])

class VanillaTextureOpt(Model):

    def __init__(self, scene, t_param_name, result_name='', t_kwargs={}, optimizer_kwargs={}, dtype=torch.float32, device='cuda'):
        super().__init__(scene, dtype, device)

        if 't_res' not in t_kwargs:
            t_kwargs['t_res'] = self.scene[t_param_name].data.shape
        if 'v_min' not in t_kwargs:
            t_kwargs['v_min'] = 0
        if 'v_max' not in t_kwargs:
            t_kwargs['v_max'] = 1

        self._t = self.scene[t_param_name]
        self._raw_t = torch.zeros(t_kwargs['t_res'], dtype=self.dtype, device=self.device, requires_grad=True)
        self._v_min = t_kwargs['v_min']
        self._v_span = t_kwargs['v_max'] - t_kwargs['v_min']

        self._result_name = result_name if result_name else t_param_name

        self.optimizer = torch.optim.Adam([self._raw_t], **optimizer_kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_data(self):
        self._t.set(torch.sigmoid(self._raw_t) * self._v_span + self._v_min)
        
    def step(self):
        self.optimizer.step()

    def get_results(self):
        results = {
            't': self._t.data.detach(),
        }
        return results

    def write_results(self, result_path):
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        write_exr(result_path / f'{self._result_name}.exr', results['t'])

class ConvexHullTextureOpt(Model):

    def __init__(self, scene, bsdf_id, N=50, t_size=512, t_kwargs={}, optimizer_kwargs={}, dtype=torch.float32, device='cuda'):
        super().__init__(scene, dtype, device)

        bsdf = self.scene.bsdfs[bsdf_id]
        self._bsdf_type = bsdf['type']

        if self._bsdf_type == 'microfacet':
            self._d = bsdf['diffuse_reflectance']
            self._s = bsdf['specular_reflectance']
            self._r = bsdf['roughness']
            self._t_kwargs = {
                "d": {
                    "name": f"{bsdf['id']}.diffuse_reflectance",
                    "v_min": 0.0,
                    "v_max": 1.0,
                },
                "s": {
                    "name": f"{bsdf['id']}.specular_reflectance",
                    "v_min": 0.0,
                    "v_max": 1.0,
                },
                "r": {
                    "name": f"{bsdf['id']}.roughness",
                    "v_min": 0.0,
                    "v_max": 1.0,
                }
            }

            for c in t_kwargs:
                for k in t_kwargs[c]:
                    self._t_kwargs[c][k] = t_kwargs[c][k]

            for c in self._t_kwargs:
                self._t_kwargs[c]['v_span'] = self._t_kwargs[c]['v_max'] - self._t_kwargs[c]['v_min']

            self.weight_map = torch.normal(0, 1e-4, (t_size, t_size, N), dtype=self.dtype, device=self.device)
            self.weight_map.requires_grad = True

            self.brdf_points = torch.rand((N, 7), dtype=self.dtype, device=self.device) * 2 - 1
            self.brdf_points.requires_grad = True
        else:
            assert False

        self.optimizer = torch.optim.Adam([self.weight_map, self.brdf_points], **optimizer_kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.__eval()

    def set_data(self):
        w = torch.softmax(self.weight_map, dim=2)
        b = torch.sigmoid(self.brdf_points)
        t = torch.matmul(w, b)

        if self._bsdf_type == 'microfacet':
            self._d.set(t[:, :, 0:3] * self._t_kwargs['d']['v_span'] + self._t_kwargs['d']['v_min'])
            self._s.set(t[:, :, 3:6] * self._t_kwargs['s']['v_span'] + self._t_kwargs['s']['v_min'])
            self._r.set(t[:, :, 6:7] * self._t_kwargs['r']['v_span'] + self._t_kwargs['r']['v_min'])

    def step(self):
        self.optimizer.step()

    def get_results(self):
        if self._bsdf_type == 'microfacet':
            results = {
                'd': self._d.data.detach(),
                's': self._s.data.detach(),
                'r': self._r.data.detach(),
            }
        return results

    def write_results(self, result_path):
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        if self._bsdf_type == 'microfacet':
            write_exr(result_path / f"{self._t_kwargs['d']['name']}.exr", results['d'])
            write_exr(result_path / f"{self._t_kwargs['s']['name']}.exr", results['s'])
            write_exr(result_path / f"{self._t_kwargs['r']['name']}.exr", results['r'])