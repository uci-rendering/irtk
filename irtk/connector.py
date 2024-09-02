from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
from .scene import Scene
import torch

_connector_table: Dict[str, Type["Connector"]] = {}


class Connector(ABC):

    def __init_subclass__(cls, connector_name: str, **kwargs: Any) -> None:
        """Registers a subclass of Connector.

        Args:
            connector_name: The name of the connector.
            **kwargs: Additional keyword arguments.
        """
        super().__init_subclass__(**kwargs)
        _connector_table[connector_name] = cls
        cls.extensions: Dict[str, Callable] = {}

    @abstractmethod
    def renderC(
        self,
        scene: Scene,
        render_options: Dict[str, Any],
        sensor_ids: List[int] = [0],
        integrator_id: int = 0,
    ) -> torch.Tensor:
        """Renders the scene with the given sensor IDs and integrator IDs.

        Args:
            scene: The scene to render.
            render_options: The render options.
            sensor_ids: The sensor IDs.
            integrator_id: The integrator ID.

        Returns:
            A torch.Tensor containing the rendered images with shape (num_sensors, h, w, c).
        """
        pass

    @abstractmethod
    def renderD(
        self,
        image_grads: torch.Tensor,
        scene: Scene,
        render_options: Dict[str, Any],
        sensor_ids: List[int] = [0],
        integrator_id: int = 0,
    ) -> List[torch.Tensor]:
        """Given the gradients of the rendered images, compute the gradients of the scene parameters.

        Args:
            image_grads: The image gradients.
            scene: The scene passed to renderC.
            render_options: The render options.
            sensor_ids: The sensor IDs.
            integrator_id: The integrator ID.

        Returns:
            A list of torch.Tensors, where each tensor corresponds to the gradient of a parameter in scene.requiring_grad, in the same order.
        """
        pass

    @classmethod
    def register(cls, class_name: str) -> Callable:
        """Registers an extension function.

        Args:
            class_name: The name of the class.

        Returns:
            A wrapper function.
        """

        def wrapper(func: Callable) -> Callable:
            cls.extensions[class_name] = func
            return func

        return wrapper


def is_connector_available(connector_name: str) -> bool:
    """Checks if a connector is available.

    Args:
        connector_name: The name of the connector.

    Returns:
        True if the connector is available, False otherwise.
    """
    return connector_name in _connector_table


def get_connector_list() -> List[str]:
    """Gets the list of available connectors.

    Returns:
        A list of available connector names.
    """
    return list(_connector_table.keys())


def get_connector(connector_name: str) -> Connector:
    """Gets a connector instance.

    Args:
        connector_name: The name of the connector.

    Returns:
        An instance of the connector.
    """
    return _connector_table[connector_name]()
