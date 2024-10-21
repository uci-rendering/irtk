from typing import List, Dict, Union, Any
from .connector import get_connector, Connector
from .scene import Scene
from .config import *
from .utils import Timer
import torch 
import numpy as np

import gin
import threading

class RenderFunction(torch.autograd.Function):
    """
    Custom autograd function for rendering.

    This class defines the forward and backward passes for the rendering process,
    allowing for automatic differentiation through the rendering operation.
    """

    @staticmethod
    def forward(ctx: Any, connector: Connector, scene: Scene, render_options: Dict[str, Any],
                sensor_ids: Union[List[int], torch.Tensor], integrator_id: int, *params: Any) -> torch.Tensor:
        """
        Performs the forward pass of the rendering operation.

        Args:
            ctx: Context object to save information for the backward pass.
            connector: The connector object used for rendering.
            scene: The scene to be rendered.
            render_options: Options for the rendering process.
            sensor_ids: List of sensor IDs to render from.
            integrator_id: ID of the integrator to use.
            *params: Additional parameters required for rendering.

        Returns:
            torch.Tensor: The rendered image(s) as a tensor.
        """
        images = connector.renderC(scene, render_options, sensor_ids=sensor_ids, integrator_id=integrator_id)
        images = torch.stack(images, dim=0)

        ctx.connector = connector
        ctx.scene = scene
        ctx.render_options = render_options
        ctx.sensor_ids = sensor_ids
        ctx.integrator_id = integrator_id
        ctx.num_no_grads = 5  # number of inputs that don't require grad

        ctx.params = params
       
        images = torch.nan_to_num(images)
        
        return images.cpu()

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple:
        """
        Performs the backward pass of the rendering operation.

        Args:
            ctx: Context object with saved information from the forward pass.
            grad_out: Gradient of the loss with respect to the output of the forward pass.

        Returns:
            tuple: Gradients with respect to the inputs of the forward pass.
        """
        image_grads = [image_grad.to(configs['device']) for image_grad in grad_out]
        param_grads = ctx.connector.renderD(image_grads, ctx.scene, ctx.render_options, ctx.sensor_ids, ctx.integrator_id)
        return tuple([None] * ctx.num_no_grads + param_grads)

@gin.configurable
class Renderer(torch.nn.Module):
    """
    A PyTorch module for rendering.

    This class wraps the rendering process in a PyTorch module, allowing it to be
    used seamlessly with other PyTorch operations.
    """

    def __init__(self, connector_name: str, render_options: Dict[str, Any] = {}):
        """
        Initializes the Renderer.

        Args:
            connector_name: Name of the connector to use for rendering.
            render_options: Options for the rendering process.
        """
        super().__init__()
        self.connector = get_connector(connector_name)
        self.render_options = render_options

    def forward(self, scene: Any, sensor_ids: Union[List[int], torch.Tensor] = [0], integrator_id: int = 0) -> torch.Tensor:
        """
        Performs the forward pass of the rendering.

        Args:
            scene: The scene to be rendered.
            sensor_ids: List of sensor IDs to render from.
            integrator_id: ID of the integrator to use.

        Returns:
            torch.Tensor: The rendered image(s) as a tensor.
        """
        if torch.is_tensor(sensor_ids):
            sensor_ids = sensor_ids.flatten().tolist()

        params = [scene[param_name] for param_name in scene.requiring_grad]
        
        images = RenderFunction.apply(self.connector, scene, self.render_options, sensor_ids, integrator_id, *params).to(configs['device'])
        return images