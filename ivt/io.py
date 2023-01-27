import os
from struct import unpack, unpack_from

from skimage.transform import resize
import igl
import imageio.v3 as iio
import imageio
import numpy as np
import torch
from skimage.transform import resize
import xatlas
import pymeshfix
from gpytoolbox import remesh_botsch
from pathlib import Path

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

def read_image(image_path, is_srgb=None):
    image_path = Path(image_path)
    image = iio.imread(image_path)
    image = np.atleast_3d(image)
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

from PIL import Image as im
def write_jpg(jpg_path, image):
    image = to_srgb(to_numpy(image))
    image = (image * 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    rgb_im = im.fromarray(image).convert('RGB')
    rgb_im.save(jpg_path, format='JPEG', quality=95)


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

class FileStream:
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'rb')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def read(self, count: int, dtype=np.byte):
        data = self.file.read(count * np.dtype(dtype).itemsize)
        return np.frombuffer(data, dtype=dtype)


def read_volume(path):
    with FileStream(path) as s:
        header = s.read(3, dtype=np.byte)
        assert header.tobytes() == b'VOL'
        version = s.read(1, dtype=np.uint8)[0]
        assert version == 3
        data_type = s.read(1, dtype=np.int32)[0]
        assert data_type == 1
        res = s.read(3, dtype=np.int32)
        nchannel = s.read(1, dtype=np.int32)[0]
        bbox_min = s.read(3, dtype=np.float32)
        bbox_max = s.read(3, dtype=np.float32)
        assert np.all(bbox_max > bbox_min)
        data = s.read(np.prod(res) * nchannel, dtype=np.float32)
        return {
            'header': header,
            'version': version,
            'data_type': data_type,
            'res': res,
            'nchannel': nchannel,
            'min': bbox_min,
            'max': bbox_max,
            'data': data
        }


def read_volume2(path):
    # https://johanneskopf.de/publications/solid/textures/file_format.txt
    with FileStream(path) as s:
        buf = s.read(4096)
        header, version, tex_name, wrap, vol_size, nchannels, bytes_per_channel = unpack_from('4si256s?iii', buf)
        tex_name = tex_name[:tex_name.find(b'\0')]
        assert header == b'VOLU'
        res = np.array([vol_size, vol_size, vol_size])
        data = s.read(np.prod(res) * nchannels)
        data = np.frombuffer(data, dtype=np.uint8) / 255.0
        return {
            'header': header,
            'version': version,
            'data_type': 2,
            'res': res,
            'nchannel': nchannels,
            'min': np.array([-1., -1., -1.]),
            'max': np.array([1., 1., 1.]),
            'data': data
        }
