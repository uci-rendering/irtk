from .io import to_numpy

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"[{self.label}] Elapsed time: {elapsed_time} seconds")

def apply_pmkmp_cm(image, vmin=0, vmax=1):
    """
    Apply the pmkmp color map to an image.
    The image should be an np.array of shape [H, W, 3].
    """
    gray_cm = plt.get_cmap('gray')

    cubicL_path = Path(__file__).parent / 'data' / 'cubicL.txt' 
    cubicL = np.loadtxt(cubicL_path)
    pmkmp_cm = LinearSegmentedColormap.from_list("cubicL", cubicL, N=256)

    norm = Normalize(vmin, vmax)
    image = norm(to_numpy(image))

    gray_image = gray_cm(image)[:, :, 0, 0]
    pmkmp_image = pmkmp_cm(gray_image)

    return pmkmp_image

def get_pmkmp_color_bar(cb_path='colorbar.png'):
    """
    Save an example pmkmp color bar to cb_path.
    """
    cubicL_path = Path(__file__).parent / 'data' / 'cubicL.txt' 
    cubicL = np.loadtxt(cubicL_path)
    cm = LinearSegmentedColormap.from_list("cubicL", cubicL, N=256)
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
    cb = ColorbarBase(ax, orientation='horizontal', 
                                cmap=cm)
    plt.savefig(cb_path, bbox_inches='tight')