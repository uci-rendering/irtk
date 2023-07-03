from .config import *
import imageio.v3 as iio
import numpy as np
import torch
from pathlib import Path
import gpytoolbox

def read_mesh(mesh_path):
    v, f, uv, fuv = gpytoolbox.read_mesh(mesh_path, return_UV=True)
    if uv.size == 0:
        uv = np.zeros((0, 2))
        fuv = np.zeros((0, 3))
    return v, f, uv, fuv

def write_mesh(mesh_path, v, f, uv=None, fuv=None):
    v = to_numpy(v)
    f = to_numpy(f)
    if uv is not None: 
        if uv.size == 0:
            uv = None
        else:
            uv = to_numpy(uv)
    if fuv is not None: 
        if fuv.size == 0:
            fuv = None
        else:
            fuv = to_numpy(fuv)
    gpytoolbox.write_mesh(str(mesh_path), v, f, uv, fuv)

def linear_to_srgb(l):
    s = np.zeros_like(l)
    m = l <= 0.00313066844250063
    s[m] = l[m] * 12.92
    s[~m] = 1.055*(l[~m]**(1.0/2.4))-0.055
    return s

def srgb_to_linear(s):
    l = np.zeros_like(s)
    m = s <= 0.0404482362771082
    l[m] = s[m] / 12.92
    l[~m] = ((s[~m]+0.055)/1.055) ** 2.4
    return l

def to_srgb(image):
    image = to_numpy(image)
    if image.shape[2] == 4:
        image_alpha = image[:, :, 3:4]
        image = linear_to_srgb(image[:, :, 0:3])
        image = np.concatenate([image, image_alpha], axis=2)
    else:
        image = linear_to_srgb(image)
    return np.clip(image, 0, 1)

def to_linear(image):
    image = to_numpy(image)
    if image.shape[2] == 4:
        image_alpha = image[:, :, 3:4]
        image = srgb_to_linear(image[:, :, 0:3])
        image = np.concatenate([image, image_alpha], axis=2)
    else:
        image = srgb_to_linear(image)
    return image

def to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    else:
        return np.array(data)
    
def to_torch(data, dtype):
    if torch.is_tensor(data):
        return data.to(dtype).to(device).contiguous()
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype).to(device).contiguous()
    else:
        return torch.tensor(data, dtype=dtype, device=device).contiguous()

def to_torch_f(data):
    return to_torch(data, ftype)

def to_torch_i(data):
    return to_torch(data, itype)

def read_image(image_path, is_srgb=None, remove_alpha=True):
    image_path = Path(image_path)
    image = iio.imread(image_path)
    image = np.atleast_3d(image)
    if remove_alpha and image.shape[2] == 4:
        image = image[:, :, 0:3]

    if image.dtype == np.uint8 or image.dtype == np.int16:
        image = image.astype("float32") / 255.0
    elif image.dtype == np.uint16 or image.dtype == np.int32:
        image = image.astype("float32") / 65535.0

    if is_srgb is None:
        if image_path.suffix in ['.exr', '.hdr', '.rgbe']:
            is_srgb = False
        else:
            is_srgb = True

    if is_srgb:
        image = to_linear(image)

    return image

def write_image(image_path, image, is_srgb=None):
    image_path = Path(image_path)
    image = to_numpy(image)
    image = np.atleast_3d(image)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if is_srgb is None:
        if image_path.suffix in ['.exr', '.hdr', '.rgbe']:
            is_srgb = False
        else:
            is_srgb = True

    if is_srgb:
        image = to_srgb(image)

    if image_path.suffix == '.exr':
        image = image.astype(np.float32)
    else:
        image = (image * 255).astype(np.uint8)

    iio.imwrite(image_path, image)
