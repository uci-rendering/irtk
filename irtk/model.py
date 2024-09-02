import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any
from .io import to_torch_f

import gin 

class Model(ABC):
    """
    Abstract base class for models. A model optimizes certain scene parameters
    and can be combined with other models to form an inverse rendering pipeline.
    Encapsulating an inverse rendering algorithm in a model promotes reusability.
    For example, you can combine existing models with your own to quickly form a
    new inverse rendering pipeline.

    Attributes:
        scene: The scene object associated with the model.
    """

    def __init__(self, scene: Any) -> None:
        """
        Initialize the Model.

        Args:
            scene: The scene object associated with the model.
        """
        self.scene = scene
    
    def initialize(self) -> None:
        """Initialize the model. Can be overridden by subclasses."""
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        """Zero out the gradients. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def set_data(self) -> None:
        """Set the data for the model. Must be implemented by subclasses."""
        pass

    def get_regularization(self) -> torch.Tensor:
        """
        Get the regularization term.

        Returns:
            A torch tensor containing the regularization value.
        """
        return to_torch_f([0])

    @abstractmethod
    def step(self) -> None:
        """Perform a step in the model. Must be implemented by subclasses."""
        pass

    def schedule_lr(self, curr_iter: int) -> None:
        """
        Schedule the learning rate.

        Args:
            curr_iter: The current iteration number.
        """
        pass
    
    @abstractmethod
    def get_results(self) -> Any:
        """Get the results of the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def write_results(self, result_path: str) -> None:
        """
        Write the results to a file.

        Args:
            result_path: The path to write the results to.
        """
        pass

    def load_states(self, state_path: str) -> None:
        """
        Load the model states from a file.

        Args:
            state_path: The path to load the states from.
        """
        pass

    def save_states(self, state_path: str) -> None:
        """
        Save the model states to a file.

        Args:
            state_path: The path to save the states to.
        """
        pass
    

@gin.configurable
class MultiOpt(Model):
    """
    A model that combines multiple other models.

    Attributes:
        _models: A list of model instances.
    """

    def __init__(self, scene: Any, model_classes: List[type]) -> None:
        """
        Initialize the MultiOpt model.

        Args:
            scene: The scene object associated with the model.
            model_classes: A list of model classes to instantiate.
        """
        super().__init__(scene)
        self._models = [model_class(scene) for model_class in model_classes]

    def initialize(self) -> None:
        """Initialize all the models."""
        for model in self._models:
            model.initialize()

    def zero_grad(self) -> None:
        """Zero out the gradients for all models."""
        for model in self._models:
            model.zero_grad()

    def set_data(self) -> None:
        """Set the data for all models."""
        for model in self._models:
            model.set_data()
        
    def step(self) -> None:
        """Perform a step for all models."""
        for model in self._models:
            model.step()

    def schedule_lr(self, curr_iter: int) -> None:
        """
        Schedule the learning rate for all models.

        Args:
            curr_iter: The current iteration number.
        """
        for model in self._models:
            model.schedule_lr(curr_iter)

    def get_results(self) -> List[Any]:
        """
        Get the results from all models.

        Returns:
            A list of results from all models.
        """
        results = []
        for model in self._models:
            results.append(model.get_results())
        return results

    def write_results(self, result_path: str) -> None:
        """
        Write the results from all models.

        Args:
            result_path: The path to write the results to.
        """
        for model in self._models:
            model.write_results(result_path)

    def get_regularization(self) -> torch.Tensor:
        """
        Get the combined regularization from all models.

        Returns:
            A torch tensor containing the combined regularization value.
        """
        reg = to_torch_f([0])
        for model in self._models:
            reg += model.get_regularization()
        return reg
    
    def load_states(self, state_path: str) -> None:
        """
        Load the states for all models.

        Args:
            state_path: The path to load the states from.
        """
        for model in self._models:
            model.load_states(state_path)

    def save_states(self, state_path: str) -> None:
        """
        Save the states for all models.

        Args:
            state_path: The path to save the states to.
        """
        for model in self._models:
            model.save_states(state_path)