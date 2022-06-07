from distutils import extension
import igl
import imageio.v3 as iio
import numpy as np

def read_obj(obj_path):
    obj_path = str(obj_path)
    
    v, tc, n, f, ftc, fn = igl.read_obj(obj_path)
    
    return v, f

def srgb_encoding(v):
	if (v <= 0.0031308):
		return (v * 12.92)
	else:
		return (1.055*(v**(1.0/2.4))-0.055)

def to_srgb(image):
    return np.clip(np.vectorize(srgb_encoding)(image), 0, 1)

def write_png(png_path, image):
    image = to_srgb(image)
    image = (image * 255).astype(np.uint8)
    iio.imwrite(png_path, image, extension='.png')