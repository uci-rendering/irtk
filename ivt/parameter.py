import torch
from abc import ABC, abstractmethod

class Parameter(ABC):
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device
        self._requires_grad = False
        self.updated = False

    @property
    @abstractmethod
    def data(self):
        pass
    
    @property
    @abstractmethod
    def raw_data(self):
        pass
    
    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    @abstractmethod
    def requires_grad(self, requires_grad):
        pass

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self

    def to_tensor(self, array):
        if torch.is_tensor(array):
            array = array.to(self.dtype).to(self.device)
        else:
            array = torch.tensor(array, dtype=self.dtype, device=self.device)
        return array

class NaiveParameter(Parameter):
    def __init__(self, raw_data, dtype, device):
        super().__init__(dtype, device)
        self.set(raw_data)

    def set(self, raw_data):
        self._raw_data = self.to_tensor(raw_data)
        self.updated = True
        self.requires_grad = self._raw_data.requires_grad

    @property
    def data(self):
        return self._raw_data

    @property
    def raw_data(self):
        return [self._raw_data]

    @Parameter.requires_grad.setter
    def requires_grad(self, requires_grad):
        self._requires_grad = requires_grad
        self._raw_data.requires_grad = requires_grad

class FixedRangeTexture(Parameter):
    def __init__(self, t_res, v_min=0.0, v_max=1.0, dtype=torch.float32, device='cuda'):
        super().__init__(dtype, device)
        self._raw_data = torch.zeros(t_res, dtype=dtype, device=device)
        self._v_min = v_min
        self._v_span = v_max - v_min

    @property
    def data(self):
        return torch.sigmoid(self._raw_data) * self._v_span + self._v_min

    @property
    def raw_data(self):
        return [self._raw_data]
    
    @Parameter.requires_grad.setter
    def requires_grad(self, requires_grad):
        self._requires_grad = requires_grad
        self._raw_data.requires_grad = requires_grad