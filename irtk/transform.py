from typing import Union, List, Tuple
from .io import to_torch_f
import torch 
import torch.nn.functional as F

def lookat(origin: Union[List[float], torch.Tensor], target: Union[List[float], torch.Tensor], up: Union[List[float], torch.Tensor]) -> torch.Tensor:
    """
    Compute a 'look-at' transformation matrix.

    Args:
        origin (Union[List[float], torch.Tensor]): The position of the camera.
        target (Union[List[float], torch.Tensor]): The point the camera is looking at.
        up (Union[List[float], torch.Tensor]): The up vector of the camera.

    Returns:
        torch.Tensor: A 4x4 transformation matrix.
    """
    origin = to_torch_f(origin)
    target = to_torch_f(target)
    up = to_torch_f(up)

    dir = F.normalize(target - origin, dim=0)
    left = F.normalize(torch.cross(up, dir), dim=0)
    new_up = F.normalize(torch.cross(dir, left), dim=0)

    to_world = to_torch_f(torch.eye(4))
    to_world[:3, 0] = left
    to_world[:3, 1] = new_up
    to_world[:3, 2] = dir
    to_world[:3, 3] = origin

    return to_world

def perspective(fov: float, aspect_ratio: float, near: float = 1e-6, far: float = 1e7) -> torch.Tensor:
    """
    Compute a perspective projection matrix.

    Args:
        fov (float): The field of view in degrees.
        aspect_ratio (float): The aspect ratio of the viewport.
        near (float, optional): The distance to the near clipping plane. Defaults to 1e-6.
        far (float, optional): The distance to the far clipping plane. Defaults to 1e7.

    Returns:
        torch.Tensor: A 4x4 perspective projection matrix.
    """
    recip = 1 / (far - near)
    tan = torch.tan(torch.deg2rad(to_torch_f(fov * 0.5)))
    cot = 1 / tan

    mat = torch.diag(to_torch_f([cot, cot, far * recip, 0]))
    mat[2, 3] = -near * far * recip
    mat[3, 2] = 1

    mat = scale([-0.5, -0.5 * aspect_ratio, 1]) @ translate([-1, -1 / aspect_ratio, 0]) @ mat

    return mat

def perspective_full(fx: float, fy: float, cx: float, cy: float, aspect_ratio: float, near: float = 1e-6, far: float = 1e7) -> torch.Tensor:
    """
    Compute a full perspective projection matrix.

    Args:
        fx (float): The focal length in x direction.
        fy (float): The focal length in y direction.
        cx (float): The x-coordinate of the principal point.
        cy (float): The y-coordinate of the principal point.
        aspect_ratio (float): The aspect ratio of the viewport.
        near (float, optional): The distance to the near clipping plane. Defaults to 1e-6.
        far (float, optional): The distance to the far clipping plane. Defaults to 1e7.

    Returns:
        torch.Tensor: A 4x4 full perspective projection matrix.
    """
    recip = 1 / (far - near)
    mat = torch.diag(to_torch_f([1, 1, far * recip, 0]))
    mat[2, 3] = -near * far * recip
    mat[3, 2] = 1

    mat = translate([1 - 2 * cx, 1 - 2 * cy, 0]) @ scale([2 * fx, 2 * fy, 1]) @ mat

    mat = scale([-0.5, -0.5 * aspect_ratio, 1]) @ translate([-1, -1 / aspect_ratio, 0]) @ mat
    return mat

def batched_transform_pos(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Apply a batched transformation to position vectors.

    Args:
        mat (torch.Tensor): A 4x4 transformation matrix.
        vec (torch.Tensor): A batch of 3D position vectors.

    Returns:
        torch.Tensor: The transformed position vectors.
    """
    mat = mat.view(1, 4, 4)
    vec_shape = vec.shape
    vec = vec.view(-1, 3, 1)
    tmp = (mat @ torch.cat([vec, torch.ones_like(vec)[:, 0:1]], dim=1)).reshape(-1, 4)
    return (tmp[:, 0:3] / tmp[:, 3:]).reshape(vec_shape)

def batched_transform_dir(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """
    Apply a batched transformation to direction vectors.

    Args:
        mat (torch.Tensor): A 4x4 transformation matrix.
        vec (torch.Tensor): A batch of 3D direction vectors.

    Returns:
        torch.Tensor: The transformed direction vectors.
    """
    mat = mat.view(1, 4, 4)
    vec_shape = vec.shape
    vec = vec.view(-1, 3, 1)
    tmp = (mat @ torch.cat([vec, torch.zeros_like(vec)[:, 0:1]], dim=1)).reshape(-1, 4)
    return tmp[:, 0:3].reshape(vec_shape)

def translate(t_vec: Union[List[float], torch.Tensor]) -> torch.Tensor:
    """
    Create a translation matrix.

    Args:
        t_vec (Union[List[float], torch.Tensor]): The translation vector.

    Returns:
        torch.Tensor: A 4x4 translation matrix.
    """
    t_vec = to_torch_f(t_vec)

    to_world = to_torch_f(torch.eye(4))
    to_world[:3, 3] = t_vec

    return to_world

def rotate(axis: Union[List[float], torch.Tensor], angle: float, use_degree: bool = True) -> torch.Tensor:
    """
    Create a rotation matrix.

    Args:
        axis (Union[List[float], torch.Tensor]): The axis of rotation.
        angle (float): The angle of rotation.
        use_degree (bool, optional): If True, the angle is in degrees. If False, it's in radians. Defaults to True.

    Returns:
        torch.Tensor: A 4x4 rotation matrix.
    """
    axis = to_torch_f(axis)
    angle = to_torch_f(angle)

    to_world = to_torch_f(torch.eye(4))
    axis = F.normalize(axis, dim=0).reshape(3, 1)

    if use_degree:
        angle = torch.deg2rad(angle)

    sin_theta = torch.sin(angle)
    cos_theta = torch.cos(angle)

    cpm = to_torch_f(torch.zeros((3, 3)))
    cpm[0, 1] = -axis[2]
    cpm[0, 2] =  axis[1]
    cpm[1, 0] =  axis[2]
    cpm[1, 2] = -axis[0]
    cpm[2, 0] = -axis[1]
    cpm[2, 1] =  axis[0]

    R = cos_theta * to_torch_f(torch.eye(3))
    R += sin_theta * cpm
    R += (1 - cos_theta) * (axis @ axis.T)

    to_world[:3, :3] = R

    return to_world

def scale(size: Union[float, List[float], torch.Tensor]) -> torch.Tensor:
    """
    Create a scaling matrix.

    Args:
        size (Union[float, List[float], torch.Tensor]): The scaling factor(s).

    Returns:
        torch.Tensor: A 4x4 scaling matrix.
    """
    size = to_torch_f(size)

    to_world = to_torch_f(torch.eye(4))

    if size.size() == () or size.size(dim=0) == 1:
        to_world[:3, :3] = to_torch_f(torch.eye(3)) * size
    elif size.size(dim=0) == 3:
        to_world[:3, :3] = torch.diag(size)
    else:
        print(f"unrecognized shape for size: {size.shape}")
        exit()

    return to_world

# texture map transform (2d)
def translate2D(t_vec: Union[List[float], torch.Tensor]) -> torch.Tensor:
    """
    Create a 2D translation matrix.

    Args:
        t_vec (Union[List[float], torch.Tensor]): The 2D translation vector.

    Returns:
        torch.Tensor: A 3x3 2D translation matrix.
    """
    t_vec = to_torch_f(t_vec)

    to_world = to_torch_f(torch.eye(3))
    to_world[:2, 2] = t_vec

    return to_world

def rotate2D(angle: float, use_degree: bool = True) -> torch.Tensor:
    """
    Create a 2D rotation matrix.

    Args:
        angle (float): The angle of rotation.
        use_degree (bool, optional): If True, the angle is in degrees. If False, it's in radians. Defaults to True.

    Returns:
        torch.Tensor: A 3x3 2D rotation matrix.
    """
    angle = to_torch_f(angle)

    to_world = to_torch_f(torch.eye(3))

    if use_degree:
        angle = torch.deg2rad(angle)

    sin_theta = torch.sin(angle)
    cos_theta = torch.cos(angle)

    R = cos_theta * to_torch_f(torch.eye(2))

    R[0 ,1] = -sin_theta
    R[1 ,0] = sin_theta

    to_world[:2, :2] = R

    return to_world

def scale2D(size: Union[float, List[float], torch.Tensor]) -> torch.Tensor:
    """
    Create a 2D scaling matrix.

    Args:
        size (Union[float, List[float], torch.Tensor]): The scaling factor(s).

    Returns:
        torch.Tensor: A 3x3 2D scaling matrix.
    """
    size = to_torch_f(size)
    to_world = to_torch_f(torch.eye(3))

    if size.size(dim=0) == 1:
        to_world[:2, :2] = torch.diag(size) * to_torch_f(torch.eye(2))
    elif size.size(dim=0) == 2:
        to_world[:2, :2] = torch.diag(size)
    else:
        print(f"unrecognized shape for size: {size.shape}")
        exit()

    return to_world