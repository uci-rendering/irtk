from .io import to_torch
import torch
from collections import OrderedDict
from typing import Any, List, Union

class ParamGroup:
    """A class to manage a group of parameters.

    This class provides functionality to add, access, and update parameters,
    as well as track which parameters require gradients or have been updated.
    """

    def __init__(self):
        """Initialize an empty ParamGroup."""
        self.params: OrderedDict = OrderedDict()
    
    def __contains__(self, param_name: str) -> bool:
        """Check if a parameter exists in the group.

        Args:
            param_name: The name of the parameter to check.

        Returns:
            True if the parameter exists, False otherwise.
        """
        return param_name in self.params

    def __getitem__(self, param_name: str) -> Any:
        """Get the value of a parameter.

        Args:
            param_name: The name of the parameter to retrieve.

        Returns:
            The value of the parameter.
        """
        return self.params[param_name]['value']
    
    def __setitem__(self, param_name: str, param_value: Any) -> None:
        """Set the value of a parameter.

        If the parameter is a tensor, this method ensures the new value
        is also a tensor with the same dtype and requires_grad status.

        Args:
            param_name: The name of the parameter to set.
            param_value: The new value for the parameter.
        """
        if self.params[param_name]['is_tensor'] and not torch.is_tensor(param_value):
            dtype = self[param_name].dtype
            requires_grad = self[param_name].requires_grad
            param_value = to_torch(param_value, dtype).requires_grad_(requires_grad)
        self.params[param_name]['value'] = param_value
        self.mark_updated(param_name)

    def add_param(self, name: str, value: Any, is_tensor: bool = False, is_diff: bool = False, help_msg: str = "") -> None:
        """Add a new parameter to the group.

        Args:
            name: The name of the parameter.
            value: The value of the parameter.
            is_tensor: Whether the parameter is a tensor.
            is_diff: Whether the parameter is differentiable.
            help_msg: A description of the parameter.
        """
        self.params[name] = {
            'value': value,
            'is_tensor': is_tensor,
            'is_diff': is_diff,
            'help_msg': help_msg,
            'updated': False
        }

    def get_requiring_grad(self) -> List[str]:
        """Get names of parameters that require gradients.

        Returns:
            A list of parameter names that are differentiable and require gradients.
        """
        return [name for name in self.params if self.params[name]['is_diff'] and self[name].requires_grad]
    
    def get_updated(self) -> List[str]:
        """Get names of parameters that have been updated.

        Returns:
            A list of parameter names that have been marked as updated.
        """
        return [name for name in self.params if self.params[name]['updated']]
    
    def mark_updated(self, param_name: str, updated: bool = True) -> None:
        """Mark a parameter as updated or not.

        Args:
            param_name: The name of the parameter to mark.
            updated: Whether to mark the parameter as updated (True) or not (False).
        """
        self.params[param_name]['updated'] = updated
    
    def __str__(self) -> str:
        """Generate a string representation of the ParamGroup.

        Returns:
            A formatted string containing information about all parameters in the group.
        """
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

