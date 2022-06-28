import ivt.scene_parser
from ivt.scene_parser import SceneParserManager
from common import *

from pathlib import Path
import torch
import numpy as np

tests = []
def add_test(func):
    def wrapper():
        print(f'Test ({func.__name__}) starts.\n')
        func()
        print(f'\nTest ({func.__name__}) ends.')
    tests.append(wrapper)
    
@add_test # comment this to skip the test
def mitsuba():
    scene_parser_name = 'mitsuba'
    spm = SceneParserManager()
    assert spm.is_available(scene_parser_name)
    sp = spm.get_scene_parser(scene_parser_name)
    
    output_path = Path('tmp_output', 'scene_parser_test', 'mitsuba')
    output_path.mkdir(parents=True, exist_ok=True)

    scene = simple_scene()
    sp.write(output_path / 'scene.xml', scene)

if __name__ == '__main__':
    for test in tests:
        test()