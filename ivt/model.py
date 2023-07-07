import torch
from abc import ABC, abstractmethod
from pathlib import Path
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
        reg = torch.tensor(0.0, device=device, dtype=ftype)
        for model in self._models:
            reg += model.get_regularization()
        return reg