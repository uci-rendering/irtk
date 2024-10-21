from .config import configs, TensorLike
import imageio
import imageio.v3 as iio
import numpy as np
import torch
from pathlib import Path
import gpytoolbox
from typing import Tuple

# Download necessary imageio plugins. If they already exists they won't be 
# downloaded again. 
imageio.plugins.freeimage.download()

def read_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reads a mesh from a file.

    Args:
        mesh_path: The path to the mesh file.

    Returns:
        A tuple containing vertices, faces, UV coordinates, and face UV coordinates.
        If the file is not an .obj or doesn't contain UV coordinates, the UV coordinates
        will be set to 0 for all vertices, and face UV indices will match face indices.
        Currently, UV coordinates can only be read from .obj files.
    """
    mesh_path = Path(mesh_path)
    support_uv = mesh_path.suffix == '.obj'
    if support_uv:
        v, f, uv, fuv = gpytoolbox.read_mesh(str(mesh_path), return_UV=True)
    else:
        v, f = gpytoolbox.read_mesh(str(mesh_path))
        uv, fuv = None, None

    if not support_uv or uv.size == 0:
        uv = np.zeros((v.shape[0], 2))
        fuv = f.copy()
    return v, f, uv, fuv

def write_mesh(mesh_path: str, v: np.ndarray, f: np.ndarray, uv: np.ndarray = None, fuv: np.ndarray = None) -> None:
    """Writes a mesh to a file.

    Args:
        mesh_path: The path to the mesh file.
        v: The vertices of the mesh, shape (num_v, 3).
        f: The faces of the mesh, shape (num_f, 3).
        uv: The UV coordinates of the mesh, shape (num_uv, 2). If not present, it will not be written.
        fuv: The face UV coordinates of the mesh, shape (num_f, 3). If not present, it will not be written.
    """
    v = to_numpy(v)
    f = to_numpy(f)
    if uv is not None: 
        uv = to_numpy(uv)
        if uv.size == 0:
            uv = None
    if fuv is not None: 
        fuv = to_numpy(fuv)
        if fuv.size == 0:
            fuv = None
    gpytoolbox.write_mesh(str(mesh_path), v, f, uv, fuv)

def linear_to_srgb(l: np.ndarray) -> np.ndarray:
    """Converts a linear color space image to sRGB.

    Args:
        l: The linear color space image.

    Returns:
        The sRGB image.
    """
    s = np.zeros_like(l)
    m = l <= 0.00313066844250063
    s[m] = l[m] * 12.92
    s[~m] = 1.055*(l[~m]**(1.0/2.4))-0.055
    return s

def srgb_to_linear(s: np.ndarray) -> np.ndarray:
    """Converts an sRGB image to linear color space.

    Args:
        s: The sRGB image.

    Returns:
        The linear color space image.
    """
    l = np.zeros_like(s)
    m = s <= 0.0404482362771082
    l[m] = s[m] / 12.92
    l[~m] = ((s[~m]+0.055)/1.055) ** 2.4
    return l

def to_srgb(image: np.ndarray) -> np.ndarray:
    """Converts an image to sRGB color space.

    Args:
        image: The input image.

    Returns:
        The sRGB image.
    """
    image = to_numpy(image)
    if image.shape[2] == 4:
        image_alpha = image[:, :, 3:4]
        image = linear_to_srgb(image[:, :, 0:3])
        image = np.concatenate([image, image_alpha], axis=2)
    else:
        image = linear_to_srgb(image)
    return np.clip(image, 0, 1)

def to_linear(image: np.ndarray) -> np.ndarray:
    """Converts an image to linear color space.

    Args:
        image: The input image.

    Returns:
        The linear color space image.
    """
    image = to_numpy(image)
    if image.shape[2] == 4:
        image_alpha = image[:, :, 3:4]
        image = srgb_to_linear(image[:, :, 0:3])
        image = np.concatenate([image, image_alpha], axis=2)
    else:
        image = srgb_to_linear(image)
    return image

def to_numpy(data: torch.Tensor) -> np.ndarray:
    """Converts a torch tensor to a numpy array.

    Args:
        data: The input torch tensor.

    Returns:
        The numpy array.
    """
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    else:
        return np.array(data)
    
def to_torch(data: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Converts a numpy array to a torch tensor.

    Args:
        data: The input numpy array.
        dtype: The desired torch data type.

    Returns:
        The torch tensor.
    """
    if torch.is_tensor(data):
        return data.to(dtype).to(configs['device']).contiguous()
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype).to(configs['device']).contiguous()
    else:
        return torch.tensor(data, dtype=dtype, device=configs['device']).contiguous()

def to_torch_f(data: np.ndarray) -> torch.Tensor:
    """Converts a numpy array to a torch float tensor.

    Args:
        data: The input numpy array.

    Returns:
        The torch float tensor.
    """
    return to_torch(data, configs['ftype'])

def to_torch_i(data: np.ndarray) -> torch.Tensor:
    """Converts a numpy array to a torch integer tensor.

    Args:
        data: The input numpy array.

    Returns:
        The torch integer tensor.
    """
    return to_torch(data, configs['itype'])

def read_image(image_path: str, is_srgb: bool = None, remove_alpha: bool = True) -> np.ndarray:
    """Reads an image from a file.

    Args:
        image_path: The path to the image file.
        is_srgb: Whether the image is in sRGB format.
        remove_alpha: Whether to remove the alpha channel.

    Returns:
        The image as a numpy array.
    """
    image_path = Path(image_path)
    
    image_ext = image_path.suffix
    iio_plugins = {
        '.exr': 'EXR-FI',
        '.hdr': 'HDR-FI',
        '.png': 'PNG-FI',
    }
    
    image = iio.imread(image_path, plugin=iio_plugins.get(image_ext))
    image = np.atleast_3d(image)
    
    if remove_alpha and image.shape[2] == 4:
        image = image[:, :, 0:3]

    if image.dtype == np.uint8 or image.dtype == np.int16:
        image = image.astype("float32") / 255.0
    elif image.dtype == np.uint16 or image.dtype == np.int32:
        image = image.astype("float32") / 65535.0

    if is_srgb is None:
        if image_ext in ['.exr', '.hdr', '.rgbe']:
            is_srgb = False
        else:
            is_srgb = True

    if is_srgb:
        image = to_linear(image)

    return image

def write_image(image_path: str, image: np.ndarray, is_srgb: bool = None) -> None:
    """Writes an image to a file.

    Args:
        image_path: The path to the image file.
        image: The image as a numpy array.
        is_srgb: Whether the image is in sRGB format.
    """
    image_path = Path(image_path)
    
    image_ext = image_path.suffix
    iio_plugins = {
        '.exr': 'EXR-FI',
        '.hdr': 'HDR-FI',
        '.png': 'PNG-FI',
    }
    iio_flags = {
        '.exr': imageio.plugins.freeimage.IO_FLAGS.EXR_NONE,
    }
    hdr_formats = ['.exr', '.hdr', '.rgbe']
    
    image = to_numpy(image)
    image = np.atleast_3d(image)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
        
    if image_ext in hdr_formats:
        is_srgb = False if is_srgb is None else is_srgb
    else:
        is_srgb = True if is_srgb is None else is_srgb
    if is_srgb:
        image = to_srgb(image)
        
    if image_ext in hdr_formats:
        image = image.astype(np.float32)
    else:
        image = (image * 255).astype(np.uint8)
    
    flags = iio_flags.get(image_ext)
    if flags is None: flags = 0
    
    iio.imwrite(image_path, image, 
                flags=flags,
                plugin=iio_plugins.get(image_ext))

def exr2png(image_path: str, verbose: bool = False) -> None:
    """Converts EXR images to PNG format recursively.

    Args:
        image_path: The path to the image file or directory. If a directory is provided,
                    it will recursively convert all .exr files in the directory and its subdirectories.
        verbose: Whether to print the file paths being processed.
    """
    image_path = Path(image_path)
    if image_path.is_dir():
        for p in image_path.glob('**/*.exr'):
            if verbose: print(p)
            im = read_image(p)
            write_image(p.with_suffix('.png'), im)
    elif image_path.suffix == '.exr':
        if verbose: print(image_path)
        im = read_image(image_path)
        write_image(image_path.with_suffix('.png'), im)

def write_video(video_path: str, frames: list, fps: int = 20, kwargs: dict = {}) -> None:
    """Writes a video from a sequence of frames.

    Args:
        video_path: The path to the video file.
        frames: A list of frames.
        fps: The frames per second.
        kwargs: Additional keyword arguments for the video writer.
    """
    video_path = Path(video_path)
    video_path.parent.mkdir(exist_ok=True, parents=True)
    writer = imageio.get_writer(video_path, fps=fps, **kwargs)
    for frame in frames:
        frame = (to_srgb(to_numpy(frame)) * 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()