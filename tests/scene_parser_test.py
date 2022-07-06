import ivt.scene_parser
from ivt.scene_parser import SceneParserManager
from ivt.connector import ConnectorManager
from ivt.io import write_png
from common import *

from pathlib import Path
import torch
import numpy as np
import os

tests = []
def add_test(func):
    def wrapper():
        print(f'Test ({func.__name__}) starts.\n')
        func()
        print(f'\nTest ({func.__name__}) ends.\n')
    tests.append(wrapper)
    
@add_test # comment this to skip the test
def write():
    spm = SceneParserManager()
    for spn in spm.get_availability_list():
        print(f'Writing scene file with scene parser [{spn}]...')
    
        output_path = Path('tmp_output', 'scene_parser_test', spn)
        output_path.mkdir(parents=True, exist_ok=True)

        scene = simple_scene()
        sp = spm.get_scene_parser(spn)
        sp.write(output_path / 'scene.xml', scene)

if __name__ == '__main__':
    for test in tests:
        test()