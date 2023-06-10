from .config import *
import imageio.v3 as iio
import numpy as np
import torch
from pathlib import Path
import trimesh
import igl

def read_obj(obj_path):
    obj_path = str(obj_path)

    v, tc, n, f, ftc, fn = igl.read_obj(obj_path)
    if f.ndim == 1:
        f = np.expand_dims(f, axis=0)
    if f.shape[1] == 4:
        f = np.concatenate([f[:, :3], f[:, (0, 2, 3)]], axis=0)
    return v, tc, n, f, ftc, fn


def write_obj(obj_path, v, f, tc=None, ftc=None):
    obj_file = open(obj_path, 'w')

    def f2s(f):
        return [str(e) for e in f]

    v = np.atleast_2d(to_numpy(v).astype(float))
    f = np.atleast_2d(to_numpy(f).astype(int) + 1)

    if tc is None and ftc is None:
        for v_ in v:
            obj_file.write(f"v {' '.join(f2s(v_))}\n")
        for f_ in f:
            obj_file.write(f"f {' '.join(f2s(f_))}\n")
    else:
        tc = to_numpy(tc).astype(float)
        ftc = to_numpy(ftc).astype(int)

        if tc.size > 0 and ftc.size == f.size:
            ftc += 1
            for v_ in v:
                obj_file.write(f"v {' '.join(f2s(v_))}\n")
            for tc_ in tc:
                obj_file.write(f"vt {' '.join(f2s(tc_))}\n")
            for f_, ftc_ in zip(f, ftc):
                obj_file.write(
                    f"f {f_[0]}/{ftc_[0]} {f_[1]}/{ftc_[1]} {f_[2]}/{ftc_[2]}\n")
        else:
            for v_ in v:
                obj_file.write(f"v {' '.join(f2s(v_))}\n")
            for f_ in f:
                obj_file.write(f"f {' '.join(f2s(f_))}\n")

    obj_file.close()

def read_mesh(mesh_path):
    mesh = trimesh.load(mesh_path, maintain_order=True)
    v = mesh.vertices
    f = mesh.faces
    if mesh.visual.kind == 'texture':
        uv = mesh.visual.uv
    else:
        uv = np.ones((v.shape[0], 2)) * 0.5

    return v, f, uv

def write_mesh(mesh_path, v, f, uv=None):
    v = to_numpy(v).astype(float)
    f = to_numpy(f).astype(int)

    if uv is None:
        visual = None
    else:
        uv = to_numpy(uv).astype(float)
        visual = trimesh.visual.texture.TextureVisuals(uv=uv)
    
    mesh = trimesh.Trimesh(v, f, visual=visual)
    mesh.export(mesh_path, write_texture=False)

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
        return data.to(dtype).to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype).to(device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)

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
