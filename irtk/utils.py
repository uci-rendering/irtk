from .io import to_numpy

import time
from pathlib import Path
from collections import OrderedDict
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, Normalize

class Timer:
    '''
    Timer to record and display operation elapsed time. 
    
    Typical usage is adding `with Timer():` before the code block.
    '''
    
    timers = OrderedDict()
    timers['Forward'] = 0
    timers['Backward'] = 0
    
    def __init__(self, label: str, prt: bool = True, record: bool = True):
        '''
        Initialize the timer.
        
        Args:
            label: A user defined label for the timer.
            prt: Whether to print the time.
            record: Whether to store the time in the `timers` dict for later retrieval. 
                Time pieces with the same label are added together.
        '''
        self.label = label
        self.prt = prt
        self.record = record
        if self.record and not label in self.timers:
            self.timers[label] = 0

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        if self.prt:
            print(f"[{self.label}] Elapsed time: {elapsed_time} seconds")
        if self.record:
            self.timers[self.label] += elapsed_time
    
    @classmethod
    def reset_timers(cls):
        '''
        Clear all recorded times in the `timers` dict.
        '''
        cls.timers = OrderedDict()
        cls.timers['Forward'] = 0
        cls.timers['Backward'] = 0

class Logger(object):
    '''
    Logger to redirect stdout to both the console and a file.
    
    Typical usage: `sys.stdout = Logger('a.log')`.
    '''
    
    def __init__(self, filename: str = "exp.log"):
        '''
        Initialize the logger.
        
        Args:
            filename: The file name of the log file.
        '''
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
 
    def write(self, message):
        if message != '\n':
            self.log.write(f'[{datetime.now()}] {message}')
        else:
            self.log.write('\n')
        self.terminal.write(message)
        self.log.flush()
    
    def flush(self):
        pass

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
