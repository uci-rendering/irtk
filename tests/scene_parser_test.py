import ivt.scene_parser
from ivt.scene_parser import SceneParserManager
from ivt.connector import ConnectorManager
from ivt.io import write_png
from common import *

from pathlib import Path
import torch
import numpy as np
import os
import psdr_cpu

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

    cm = ConnectorManager()
    assert cm.is_available('psdr_enzyme')
    connector = cm.get_connector('psdr_enzyme')
    
    output_path = Path('tmp_output', 'scene_parser_test', 'mitsuba')
    output_path.mkdir(parents=True, exist_ok=True)

    # Connector + ivt scene
    print('Rendering the ivt scene representation with a psdr-enzyme connector...')
    scene = simple_scene()
    image_a = connector.renderC(scene)[0]
    write_png(output_path / 'image_a.png', image_a)

    # Mitsuba scene + psdr_cpu
    print('Rendering a scene file with psdr-enzyme directly...')
    sp.write(output_path / 'scene.xml', scene)
    old_path = os.getcwd()
    os.chdir(output_path)
    psdr_scene = psdr_cpu.Scene('scene.xml')
    os.chdir(old_path)
    integrator = psdr_cpu.Direct()
    render_options = psdr_cpu.RenderOptions(
        scene.render_options['seed'],
        scene.render_options['num_samples'],
        scene.render_options['max_bounces'],
        scene.render_options['num_samples_primary_edge'],
        scene.render_options['num_samples_secondary_edge'],
        scene.render_options['quiet'],
    )
    width, height = scene.film['resolution']
    image_b = integrator.renderC(psdr_scene, render_options).reshape(width, height, 3)
    write_png(output_path / 'image_b.png', image_b)

    print(f'L1 loss: {np.abs(image_a - image_b).mean()}')

if __name__ == '__main__':
    for test in tests:
        test()