import os

from skimage.transform import resize
import igl
import imageio.v3 as iio
import imageio
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from numba import jit, njit

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
                obj_file.write(
                    f"f {f_[0]}/{ftc_[0]} {f_[1]}/{ftc_[1]} {f_[2]}/{ftc_[2]}\n")
        else:
            for v_ in v:
                obj_file.write(f"v {' '.join(f2s(v_))}\n")
            for f_ in f:
                obj_file.write(f"f {' '.join(f2s(f_))}\n")

    obj_file.close()


def linear_to_srgb(l):
    if l <= 0.00313066844250063:
        return l * 12.92
    else:
        return 1.055*(l**(1.0/2.4))-0.055


def srgb_to_linear(s):
    if s <= 0.0404482362771082:
        return s / 12.92
    else:
        return ((s+0.055)/1.055) ** 2.4

# def to_srgb(image):
#     return np.clip(np.vectorize(linear_to_srgb)(to_numpy(image)), 0, 1)

# def to_linear(image):
#     return np.vectorize(srgb_to_linear)(to_numpy(image))

# @jit(nopython=True, fastmath=True)
def to_srgb(l):
    return np.where(l <= 0.00313066844250063, l * 12.92, 1.055*(l**(1.0/2.4))-0.055)

# @jit(nopython=True, fastmath=True)
def to_linear(l):
    return np.where(l <= 0.0404482362771082, l / 12.92, ((l+0.055)/1.055) ** 2.4)

def to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    else:
        return data


def read_png(png_path):
    image = iio.imread(png_path, extension='.png')
    if image.ndim == 4:
        image = image[0]
    image = image.astype(float) / 255
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


def read_texture(tex_path, res):
    image = iio.imread(tex_path)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    return resize(image, (res, res))


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

