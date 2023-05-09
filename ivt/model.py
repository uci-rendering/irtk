import torch
from abc import ABC, abstractmethod
from pathlib import Path
from .io import *
from .optimizers import LargeStepsOptimizer
from .scene import *
from .config import *

import gin 

class Model(ABC):
    def __init__(self, scene):
        self.scene = scene

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

    def get_regularization(self):
        return torch.tensor(0.0, device=device, dtype=ftype)

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

    def __init__(self, scene, mesh_id, optimizer_kwargs={}, init_mesh_path='', result_name='', v_mask=None):
        super().__init__(scene)

        mesh = self.scene[mesh_id]
        if init_mesh_path:
            mesh = Mesh.from_file(init_mesh_path, mesh['mat_id'])
            scene.set(mesh_id, mesh)
        self.mesh = mesh
        self._v = mesh['v']
        self._f = mesh['f']
        self._v.requires_grad = True

        self._result_name = result_name if result_name else mesh_id

        self._v_mask = v_mask

        self.optimizer = LargeStepsOptimizer(self._v, self._f, **optimizer_kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_data(self):
        self.mesh.mark_updated('v')
        
    def step(self):
        if self._v_mask is not None:
            self._v.grad[~self._v_mask] = 0

        self.optimizer.step()

    def get_results(self):
        results = {
            'v': self._v.detach(),
            'f': self._f,
        }
        return results

    def write_results(self, result_path):
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        write_obj(result_path / f'{self._result_name}.obj', results['v'], results['f'])