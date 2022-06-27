from common import *

from pathlib import Path
import igl
import torch
import numpy as np

tests = []
def add_test(func):
    def wrapper():
        print(f'Test ({func.__name__}) starts.\n')
        func()
        print(f'\nTest ({func.__name__}) ends.')
    tests.append(wrapper)

@add_test
def different_backends():
    scene = simple_scene()
    
    print('backend: torch + cpu')
    print(scene)
    print()
    
    scene.device = 'cuda'
    scene.configure()
    print('backend: torch + cuda')
    print(scene)
    print()
    
    scene.backend = 'numpy'
    scene.ftype = np.float32
    scene.itype = np.int64
    scene.device = 'cpu'
    scene.configure()
    print('backend: numpy')
    print(scene)

if __name__ == '__main__':
    for test in tests:
        test()