import torch 
import math
from typing import Literal, Union

def sample_sphere(
    batch: int,
    radius: float,
    method: Literal['uniform', 'fibonacci'] = 'uniform',
    axis: int = 1,
    phi_min: float = 0,
    phi_max: float = math.pi,
    theta_min: float = 0,
    theta_max: float = 2 * math.pi
) -> torch.Tensor:
    """
    Sample points on a sphere.

    Args:
        batch: Number of points to sample.
        radius: Radius of the sphere.
        method: Sampling method, either 'uniform' or 'fibonacci'.
        axis: Axis to use as up direction (0 for x, 1 for y, 2 for z).
        phi_min: Minimum polar angle in radians.
        phi_max: Maximum polar angle in radians.
        theta_min: Minimum azimuthal angle in radians.
        theta_max: Maximum azimuthal angle in radians.

    Returns:
        torch.Tensor: Sampled points on the sphere with shape (batch, 3).

    Raises:
        ValueError: If an invalid sampling method is provided.
    """
    if method == 'uniform':
        phi_range = phi_max - phi_min
        theta_range = theta_max - theta_min
        phi = torch.rand(batch, 1) * phi_range + phi_min
        theta = torch.rand(batch, 1) * theta_range + theta_min   
        sinPhi = torch.sin(phi)
        cosPhi = torch.cos(phi) 
        sinTheta = torch.sin(theta) 
        cosTheta = torch.cos(theta) 
        samples = torch.cat([
            sinPhi * cosTheta, 
            cosPhi,
            sinPhi * sinTheta
        ], dim=1) * radius

    elif method == 'fibonacci':
        # From http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
        golden_ratio = (1 + 5**0.5) / 2
        i = torch.arange(batch).unsqueeze(-1)
        theta = 2 * torch.pi * i / golden_ratio
        cos_phi_range = math.cos(phi_min) - math.cos(phi_max)
        phi = torch.acos(math.cos(phi_min) - cos_phi_range * (i + 0.5) / batch)
        sinPhi = torch.sin(phi)  
        cosPhi = torch.cos(phi)
        sinTheta = torch.sin(theta) 
        cosTheta = torch.cos(theta) 
        samples = torch.cat([
            sinPhi * cosTheta, 
            cosPhi,
            sinPhi * sinTheta
        ], dim=1) * radius

    else:
        raise ValueError(f"Invalid sampling method: {method}. Supported methods are 'uniform' and 'fibonacci'.")
    
    # Switch axis 
    index = torch.LongTensor([0, 1, 2])
    index[axis] = 1 
    index[1] = axis 
    points = samples[:, index]

    return points

def sample_hemisphere(
    batch: int,
    radius: float,
    method: Literal['uniform', 'fibonacci'] = 'uniform',
    axis: int = 1
) -> torch.Tensor:
    """
    Sample points on a hemisphere.

    Args:
        batch: Number of points to sample.
        radius: Radius of the hemisphere.
        method: Sampling method, either 'uniform' or 'fibonacci'.
        axis: Axis to use as up direction (0 for x, 1 for y, 2 for z).

    Returns:
        torch.Tensor: Sampled points on the hemisphere with shape (batch, 3).
    """
    return sample_sphere(batch, radius, method, axis, phi_max=0.5 * math.pi)