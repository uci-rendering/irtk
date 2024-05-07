import torch
from abc import ABC, abstractmethod
from pathlib import Path
from .io import to_torch_f

import gin 

class Model(ABC):
    def __init__(self, scene):
        self.scene = scene
    
    def initialize(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    @abstractmethod
    def set_data(self):
        pass

    def get_regularization(self):
        return to_torch_f([0])

    @abstractmethod
    def step(self):
        pass

    def schedule_lr(self, curr_iter):
        pass
    
    @abstractmethod
    def get_results(self):
        pass

    @abstractmethod
    def write_results(self, result_path):
        pass

    def load_states(self, state_path):
        pass

    def save_states(self, state_path):
        pass
    

@gin.configurable
class MultiOpt(Model):

    def __init__(self, scene, model_classes):
        super().__init__(scene)

        self._models = [model_class(scene) for model_class in model_classes]

    def initialize(self):
        for model in self._models:
            model.initialize()

    def zero_grad(self):
        for model in self._models:
            model.zero_grad()

    def set_data(self):
        for model in self._models:
            model.set_data()
        
    def step(self):
        for model in self._models:
            model.step()

    def schedule_lr(self, curr_iter):
        for model in self._models:
            model.schedule_lr(curr_iter)

    def get_results(self):
        results = []
        for model in self._models:
            results.append(model.get_results())
        return results

    def write_results(self, result_path):
        for model in self._models:
            model.write_results(result_path)

    def get_regularization(self):
        reg = to_torch_f([0])
        for model in self._models:
            reg += model.get_regularization()
        return reg
    
    def load_states(self, state_path):
        for model in self._models:
            model.load_states(state_path)

    def save_states(self, state_path):
        for model in self._models:
            model.save_states(state_path)