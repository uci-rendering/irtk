from .io import to_torch_f
import torch 
import torch.nn.functional as F

def lookat(origin, target, up):
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

def perspective(fov, aspect_ratio, near=1e-6, far=1e7):
    recip = 1 / (far - near)
    tan = torch.tan(torch.deg2rad(to_torch_f(fov * 0.5)))
    cot = 1 / tan

    mat = torch.diag(to_torch_f([cot, cot, far * recip, 0]))
    mat[2, 3] = -near * far * recip
    mat[3, 2] = 1

    mat = scale([-0.5, -0.5 * aspect_ratio, 1]) @ translate([-1, -1 / aspect_ratio, 0]) @ mat

    return mat

def batched_transform_pos(mat, vec):
    mat = mat.view(1, 4, 4)
    vec = vec.view(-1, 3, 1)
    tmp = (mat @ torch.cat([vec, torch.ones_like(vec)[:, 0:1]], dim=1)).reshape(-1, 4)
    return tmp[:, 0:3] / tmp[:, 3:]

def batched_transform_dir(mat, vec):
    mat = mat.view(1, 4, 4)
    vec = vec.view(-1, 3, 1)
    tmp = (mat @ torch.cat([vec, torch.zeros_like(vec)[:, 0:1]], dim=1)).reshape(-1, 4)
    return tmp[:, 0:3]

def translate(t_vec):
    t_vec = to_torch_f(t_vec)

    to_world = to_torch_f(torch.eye(4))
    to_world[:3, 3] = t_vec

    return to_world

def rotate(axis, angle, use_degree=True):
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

def scale(size):
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
def translate2D(t_vec):
    t_vec = to_torch_f(t_vec)

    to_world = to_torch_f(torch.eye(3))
    to_world[:2, 2] = t_vec

    return to_world

def rotate2D(angle, use_degree=True):
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

def scale2D(size):
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