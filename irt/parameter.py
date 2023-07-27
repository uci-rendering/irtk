from .io import to_torch
import torch
from collections import OrderedDict

class ParamGroup:

    def __init__(self):
        self.params = OrderedDict()

    def __getitem__(self, param_name):
        return self.params[param_name]['value']
    
    def __setitem__(self, param_name, param_value):
        if self.params[param_name]['is_tensor'] and not torch.is_tensor(param_value):
            dtype = self[param_name].dtype
            requires_grad = self[param_name].requires_grad
            param_value = to_torch(param_value, dtype).requires_grad_(requires_grad)
        self.params[param_name]['value'] = param_value
        self.mark_updated(param_name)

    def add_param(self, name, value, is_tensor=False, is_diff=False, help_msg=""):
        self.params[name] = {
            'value': value,
            'is_tensor': is_tensor,
            'is_diff': is_diff,
            'help_msg': help_msg,
            'updated': False
        }

    def get_requiring_grad(self):
        return [name for name in self.params if self.params[name]['is_diff'] and self[name].requires_grad]
    
    def get_updated(self):
        return [name for name in self.params if self.params[name]['updated']]
    
    def mark_updated(self, param_name):
        self.params[param_name]['updated'] = True
    
    def __str__(self):
        lines = []

        lines.append('----')

        for name in self.params:
            lines.append(f"{name}:")
            lines.append(f"\tdescription: {self.params[name]['help_msg']}")

            is_diff = self.params[name]['is_diff']
            lines.append(f"\tis differentiable: {is_diff}")
            if is_diff:
                lines.append(f"\trequires grad: {self[name].requires_grad}")

            lines.append(f"\tvalue: \t{self[name]}")

        lines.append('----')

        return '\n'.join(lines)

