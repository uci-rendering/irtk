import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import igl
import imageio.v3 as iio
import imageio
import numpy as np
import torch
from skimage.transform import resize
import xatlas
import pymeshfix
from gpytoolbox import remesh_botsch

def read_obj(obj_path):
    obj_path = str(obj_path)
    
    v, tc, n, f, ftc, fn = igl.read_obj(obj_path)
    
    return v, tc, n, f, ftc, fn

def write_obj(obj_path, v, f, tc=None, ftc=None):
    obj_file = open(obj_path, 'w')

    def f2s(f):
        return [str(e) for e in f]

    v = to_numpy(v).astype(float)
    f = to_numpy(f).astype(int) + 1

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
                obj_file.write(f"f {f_[0]}/{ftc_[0]} {f_[1]}/{ftc_[1]} {f_[2]}/{ftc_[2]}\n")
        else:
            for v_ in v:
                obj_file.write(f"v {' '.join(f2s(v_))}\n")
            for f_ in f:
                obj_file.write(f"f {' '.join(f2s(f_))}\n")

    obj_file.close()

def unwrap_uv(v, f):
    vmapping, f, tc = xatlas.parametrize(v, f)
    return v[vmapping], f, tc, f

def fix_mesh(v, f):
    v, f = pymeshfix.clean_from_arrays(v, f)
    return v, f

def remesh(v, f, iter=10, edge_length=0.1):
    v, f = remesh_botsch(v, f, i=iter, h=edge_length)
    return v, f

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
    return np.clip(linear_to_srgb(to_numpy(image)), 0, 1)

def to_linear(image):
    return srgb_to_linear(to_numpy(image))

def to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    else:
        return np.array(data)

def read_png(png_path, is_srgb=True):
    image = iio.imread(png_path, extension='.png')
    if image.dtype == np.uint8 or image.dtype == np.int16:
        image = image.astype("float32") / 255.0
    elif image.dtype == np.uint16 or image.dtype == np.int32:
        image = image.astype("float32") / 65535.0

    if len(image.shape) == 4:
        image = image[0]

    # Only read the RGB channels
    if len(image.shape) == 3:
        image = image[:, :, :3]

    if is_srgb:
        return to_linear(image)
    else:
        return image

def write_png(png_path, image):
    image = to_srgb(to_numpy(image))
    image = (image * 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    iio.imwrite(png_path, image, extension='.png')

def read_exr(exr_path):
    image = iio.imread(exr_path, extension='.exr')
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    return image

def write_exr(exr_path, image):
    image = to_numpy(image).astype(np.float32)
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    try:
        iio.imwrite(exr_path, image)
    except OSError:
        imageio.plugins.freeimage.download()
        iio.imwrite(exr_path, image, extension='.exr')

def resize_image(image, height, width):
    return resize(image, (height, width))