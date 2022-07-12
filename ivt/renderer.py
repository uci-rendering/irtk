from ivt.connector import ConnectorManager
import torch 
import numpy as np
from copy import deepcopy

class RenderFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, scene, connector, sensor_ids, device, dtype, *params):
        images = connector.renderC(scene, sensor_ids=sensor_ids)
        if isinstance(images[0], np.ndarray):
            images = [torch.from_numpy(image) for image in images]
        images = [image.to(device).to(dtype) for image in images]
        images = torch.stack(images, dim=0)

        ctx.scene = scene 
        ctx.connector = connector
        ctx.params = params
        ctx.sensor_ids = sensor_ids

        return images

    @staticmethod
    def backward(ctx, grad_out):
        image_grads = [image_grad for image_grad in grad_out]
        param_grads = ctx.connector.renderD(image_grads, ctx.scene, ctx.sensor_ids)
        return tuple([None] * 5 + param_grads)

class Renderer(torch.nn.Module):

    def __init__(self, connector_name, device, dtype):
        super().__init__()
        cm = ConnectorManager()
        self.connector = cm.get_connector(connector_name)
        self.device = device
        self.dtype = dtype

    def forward(self, scene, params=[], sensor_ids=[0]):
        images = RenderFunction.apply(scene, self.connector, sensor_ids, self.device, self.dtype, *params)
        return images