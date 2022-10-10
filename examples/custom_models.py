from ivt.model import Model
import torch
from ivt.io import *
from pathlib import Path

import gin

@gin.configurable
class ConvexHullTextureOpt(Model):

    def __init__(
        self, 
        scene, 
        bsdf_id, 
        optimizer_class,
        N=50, 
        t_size=512, 
        d_name='diffuse',
        s_name='specular',
        r_name='roughness',
        d_v_min=0.0,
        d_v_max=1.0, 
        s_v_min=0.0,
        s_v_max=1.0, 
        r_v_min=0.0,
        r_v_max=1.0,
        ):
        
        super().__init__(scene)

        bsdf = self.scene.bsdfs[bsdf_id]
        self._bsdf_type = bsdf['type']

        self._d = bsdf['diffuse_reflectance']
        self._s = bsdf['specular_reflectance']
        self._r = bsdf['roughness']

        self._d_name = d_name
        self._s_name = s_name
        self._r_name = r_name

        self._d_v_min = d_v_min
        self._s_v_min = s_v_min
        self._r_v_min = r_v_min

        self._d_v_span = d_v_max - d_v_min
        self._s_v_span = s_v_max - d_v_min
        self._r_v_span = r_v_max - d_v_min

        self.weight_map = torch.normal(0, 1e-4, (t_size, t_size, N), dtype=self.ftype, device=self.device)
        self.weight_map.requires_grad = True

        self.brdf_points = torch.rand((N, 7), dtype=self.ftype, device=self.device) * 2 - 1
        self.brdf_points.requires_grad = True

        self.optimizer = optimizer_class([self.weight_map, self.brdf_points])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_data(self):
        w = torch.softmax(self.weight_map, dim=2)
        b = torch.sigmoid(self.brdf_points)
        t = torch.matmul(w, b)

        self._d.set(t[:, :, 0:3] * self._d_v_span + self._d_v_min)
        self._s.set(t[:, :, 3:6] * self._s_v_span + self._s_v_min)
        self._r.set(t[:, :, 6:7] * self._r_v_span + self._r_v_min)

    def step(self):
        self.optimizer.step()

    def get_results(self):
        if self._bsdf_type == 'microfacet':
            results = {
                'd': self._d.data.detach(),
                's': self._s.data.detach(),
                'r': self._r.data.detach(),
            }
        return results

    def write_results(self, result_path):
        result_path = Path(result_path)
        result_path.mkdir(parents=True, exist_ok=True)

        results = self.get_results()
        if self._bsdf_type == 'microfacet':
            write_exr(result_path / f"{self._d_name}.exr", results['d'])
            write_exr(result_path / f"{self._s_name}.exr", results['s'])
            write_exr(result_path / f"{self._r_name}.exr", results['r'])