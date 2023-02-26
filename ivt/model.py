import torch
from abc import ABC, abstractmethod
from pathlib import Path
import sys
from .io import *
from .parameter import *
from .loss import mesh_laplacian_smoothing
from .optimizers import LargeStepsOptimizer

import gin 

class Model(ABC):
    def __init__(self, scene):
        self.scene = scene
        self._ftype = scene.ftype
        self._device = scene.device

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
    def ftype(self):
        return self._ftype

    @property
    def device(self):
        return self._device

    def get_regularization(self):
        return torch.tensor(0.0, device=self._device, dtype=self._ftype)

@gin.configurable
class MultiOpt(Model):

    def __init__(self, scene, model_classes):
        super().__init__(scene)

        self._models = [model_class(scene) for model_class in model_classes]

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

@gin.configurable
class LargeStepsShapeOpt(Model):

    def __init__(self, scene, mesh_id, optimizer_kwargs={}, init_mesh_path='', result_name='', w_smooth=0.0):
        super().__init__(scene)

        mesh = self.scene.meshes[mesh_id]
        self._v = mesh['vertex_positions']
        self._f = mesh['vertex_indices']
        self._tc = mesh['uv_positions']
        self._ftc = mesh['uv_indices']
        self._w_smooth = w_smooth

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

    def get_regularization(self):
        return self._w_smooth * mesh_laplacian_smoothing(self._v.data, self._f.data)

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

@gin.configurable
class VanillaTextureOpt(Model):

    def __init__(self, scene, t_param_name, t_res, optimizer_class, result_name='', v_min=0, v_max=1):
        super().__init__(scene)

        self._t = self.scene[t_param_name]
        self._raw_t = torch.zeros(t_res, dtype=self.ftype, device=self.device, requires_grad=True)
        self._v_min = v_min
        self._v_span = v_max - v_min

        self._result_name = result_name if result_name else t_param_name

        self.optimizer = optimizer_class([self._raw_t])

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