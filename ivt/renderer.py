from .connector import get_connector
from .config import *
import torch 
import numpy as np

import gin

class RenderFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, connector, scene, render_options, sensor_ids, integrator_id, *params):
        images = connector.renderC(scene, render_options, sensor_ids=sensor_ids, integrator_id=integrator_id)
        images = torch.stack(images, dim=0)

        ctx.connector = connector
        ctx.scene = scene
        ctx.render_options = render_options
        ctx.sensor_ids = sensor_ids
        ctx.integrator_id = integrator_id
        ctx.num_no_grads = 5 # number of inputs that don't require grad

        ctx.params = params
       
        assert(images.sum().isfinite())
        return images

    @staticmethod
    def backward(ctx, grad_out):
        image_grads = [image_grad for image_grad in grad_out]
        param_grads = ctx.connector.renderD(image_grads, ctx.scene, ctx.render_options, ctx.sensor_ids, ctx.integrator_id)
        return tuple([None] * ctx.num_no_grads + param_grads)

@gin.configurable
class Renderer(torch.nn.Module):

    def __init__(self, connector_name, render_options=None):
        super().__init__()
        self.connector = get_connector(connector_name)
        self.render_options = render_options

    def forward(self, scene, sensor_ids=[0], integrator_id=0):
        assert self.render_options is not None, "Please set render options first."
        if torch.is_tensor(sensor_ids):
            sensor_ids = sensor_ids.flatten().tolist()

        params = [scene[param_name] for param_name in scene.requiring_grad]
        
        images = RenderFunction.apply(self.connector, scene, self.render_options, sensor_ids, integrator_id, *params)
        return images

    def set_render_options(self, render_options):
        self.render_options = render_options