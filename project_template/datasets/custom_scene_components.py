from ivt import PSDRJITConnector
from ivt.parameter import ParamGroup
from ivt.transform import lookat
from ivt.io import to_torch_f

import drjit
import psdr_jit
from drjit.scalar import Array3f
from drjit.cuda import Array3f as Vector3fC, Array3i as Vector3iC
from drjit.cuda.ad import Array3f as Vector3fD, Float32 as FloatD, Matrix4f as Matrix4fD, Matrix3f as Matrix3fD
from drjit.cuda.ad import Float32 as FloatD
import torch

class MyPerspectiveCamera(ParamGroup):
    
    def __init__(self, fx, fy, cx, cy, to_world=torch.eye(4), near=1e-6, far=1e7):
        super().__init__()

        self.add_param('fx', fx, help_msg='sensor focal length in x axis')
        self.add_param('fy', fy, help_msg='sensor focal length in y axis')
        self.add_param('cx', cx, help_msg='sensor principal point offset in x axis')
        self.add_param('cy', cy, help_msg='sensor principal point offset in y axis')
        self.add_param('near', near, help_msg='sensor near clip')
        self.add_param('far', far, help_msg='sensor far clip')
        self.add_param('to_world', to_torch_f(to_world), is_tensor=True, is_diff=True, help_msg='sensor to_world matrix')

    @classmethod
    def from_lookat(cls, fx, fy, cx, cy, origin, target, up, near=1e-6, far=1e7):
        sensor = cls(fx, fy, cx, cy)
        origin = to_torch_f(origin)
        target = to_torch_f(target)
        up = to_torch_f(up)
        sensor['to_world'] = lookat(origin, target, up)
        return sensor

@PSDRJITConnector.register(MyPerspectiveCamera)
def process_perspective_camera(name, scene):
    sensor = scene[name]
    cache = scene.cached['psdr_jit']
    psdr_scene = cache['scene']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        psdr_sensor = psdr_jit.PerspectiveCamera(sensor['fx'], sensor['fy'], sensor['cx'], sensor['cy'], sensor['near'], sensor['far'])
        psdr_sensor.to_world = Matrix4fD(sensor['to_world'].reshape(1, 4, 4))
        psdr_scene.add_Sensor(psdr_sensor)
        cache['name_map'][name] = f"Sensor[{psdr_scene.num_sensors - 1}]"

    psdr_sensor = psdr_scene.param_map[cache['name_map'][name]]
    
    # Update parameters
    updated = sensor.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == "to_world":
                psdr_sensor.to_world = Matrix4fD(sensor['to_world'].reshape(1, 4, 4))
            sensor.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    drjit_params = []
    requiring_grad = sensor.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == "to_world":
                drjit_param = psdr_sensor.to_world
                drjit.enable_grad(drjit_param)
                drjit_params.append(drjit_param)

    return drjit_params