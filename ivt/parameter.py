import torch
from abc import ABC, abstractmethod

class Parameter(ABC):
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
    def __init__(self, raw_data, dtype, device):
        super().__init__(dtype, device)
        self._raw_data = raw_data

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