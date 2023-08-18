import trimesh
from chamferdist import ChamferDistance
import numpy as np
import torch

def chamfer_distance(mesh_a_path, mesh_b_path, num_samples, mode='bidirectional', return_extra=False):
    assert mode in ['forward', 'backward', 'bidirectional']

    mesh_a = trimesh.load_mesh(mesh_a_path)
    mesh_b = trimesh.load_mesh(mesh_b_path)
    samples_a, _ = trimesh.sample.sample_surface(mesh_a, num_samples)
    samples_b, _ = trimesh.sample.sample_surface(mesh_b, num_samples)
    samples_a = torch.from_numpy(samples_a.view(np.ndarray)).unsqueeze(0).cuda()
    samples_b = torch.from_numpy(samples_b.view(np.ndarray)).unsqueeze(0).cuda()

    reverse = False
    bidirectional = False

    if mode == 'backward':
        reverse = True
    elif mode == 'bidirectional':
        bidirectional = True

    cd = ChamferDistance()
    dist = cd(samples_a, samples_b, 
                  reverse=reverse, 
                  bidirectional=bidirectional, 
                  point_reduction=None, 
                  batch_reduction=None).squeeze(0)
    mean_dist = dist.mean().item()

    if return_extra:
        extra = {
            'dist': dist, 
            'samples_a': samples_a,
            'samples_b': samples_b,
        }
        return mean_dist, extra

    return mean_dist