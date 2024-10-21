import trimesh
import numpy as np
import torch
from .config import *
from typing import Literal, Dict, Union

def chamfer_distance(
    mesh_a_: Union[str, Dict], 
    mesh_b_: Union[str, Dict], 
    num_samples: int, 
    mode: Literal['forward', 'backward', 'bidirectional'] = 'bidirectional'
) -> float:
    """Computes the Chamfer Distance (CD) between two meshes.

    Args:
        mesh_a_: One mesh specified by file name or a dict containing 'v' and 'f'. 
        mesh_b_: Another mesh specified by file name or a dict containing 'v' and 'f'. 
        num_samples: Number of samples to compute Chamfer Distance.
        mode: The sample mode. Supports 'forward', 'backward', and 'bidirectional'.

    Returns:
        The Chamfer Distance between the two meshes.
    """
    try:
        from chamferdist import ChamferDistance
    except ImportError:
        raise ImportError("The package 'chamferdist' is not installed. Please install it using 'pip install chamferdist'.")

    assert mode in ['forward', 'backward', 'bidirectional']

    if type(mesh_a_) == str:
        mesh_a = trimesh.load_mesh(mesh_a_)
    else:
        mesh_a = trimesh.Trimesh(mesh_a_['v'], mesh_a_['f'])
    if type(mesh_b_) == str:
        mesh_b = trimesh.load_mesh(mesh_b_)
    else:
        mesh_b = trimesh.Trimesh(mesh_b_['v'], mesh_b_['f'])
        
    samples_a, _ = trimesh.sample.sample_surface(mesh_a, num_samples)
    samples_b, _ = trimesh.sample.sample_surface(mesh_b, num_samples)
    samples_a = torch.from_numpy(samples_a.view(np.ndarray)).unsqueeze(0).to(configs['device'])
    samples_b = torch.from_numpy(samples_b.view(np.ndarray)).unsqueeze(0).to(configs['device'])

    reverse = False
    bidirectional = False

    if mode == 'backward':
        reverse = True
    elif mode == 'bidirectional':
        bidirectional = True

    cd = ChamferDistance()
    dist = cd(samples_a, samples_b, 
              reverse=reverse, 
              bidirectional=bidirectional).item()
    mean_dist = dist / num_samples

    return mean_dist