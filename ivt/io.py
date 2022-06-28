import igl
import imageio.v3 as iio
import imageio
import numpy as np
import torch

def read_obj(obj_path):
    obj_path = str(obj_path)
    
    v, tc, n, f, ftc, fn = igl.read_obj(obj_path)
    
    return v, f

def write_obj(obj_path, v, f):
    v = to_numpy(v)
    f = to_numpy(f)
    obj_path = str(obj_path)
    igl.write_obj(obj_path, v, f)

def srgb_encoding(v):
	if (v <= 0.0031308):
		return (v * 12.92)
	else:
		return (1.055*(v**(1.0/2.4))-0.055)

def to_srgb(image):
    return np.clip(np.vectorize(srgb_encoding)(image), 0, 1)

def to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    else:
        return data

def write_png(png_path, image):
    image = to_srgb(to_numpy(image))
    image = (image * 255).astype(np.uint8)
    iio.imwrite(png_path, image, extension='.png')

def write_exr(exr_path, image):
    image = to_numpy(image).astype(np.float32)
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    try:
        iio.imwrite(exr_path, image)
    except OSError:
        imageio.plugins.freeimage.download()
        iio.imwrite(exr_path, image, extension='.exr')
    