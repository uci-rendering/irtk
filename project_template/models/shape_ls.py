from ivt.model import Model
from ivt.io import *
from ivt.config import *
from pathlib import Path
from .utils_largestep import LargeStepsOptimizer

import gin

@gin.configurable
class ShapeLS(Model):

    def __init__(
            self, 
            scene, 
            mesh_id, 
            init_mesh_path='', 
            v_mask=None,
            optimizer_kwargs={}, 
            ):
        
        super().__init__(scene)

        self.mesh = scene[mesh_id]
        self.mesh['can_change_topology'] = False

        if isinstance(init_mesh_path, str):
            v, f, uv, fuv = read_mesh(init_mesh_path)
            self.mesh['v'] = v
            self.mesh['f'] = f
            self.mesh['uv'] = uv
            self.mesh['fuv'] = fuv

        self.mesh['v'].requires_grad = True

        self.v_mask = v_mask

        self.optimizer = LargeStepsOptimizer(self.mesh['v'], self.mesh['f'], **optimizer_kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_data(self):
        return
        
    def step(self):
        if self.v_mask is not None:
            self.mesh['v'].grad[~self.v_mask] = 0

        self.optimizer.step()

    def get_results(self):
        results = {
            'v': self.mesh['v'].detach(),
            'f': self.mesh['f'],
            'uv': self.mesh['uv'],
            'fuv': self.mesh['fuv'],
        }
        return results

    def write_results(self, result_path):
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        write_mesh(result_path / 'mesh.obj', results['v'], results['f'], results['uv'], results['fuv'])