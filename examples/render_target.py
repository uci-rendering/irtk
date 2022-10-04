import json
from ivt.io import write_exr, write_png
from ivt.renderer import Renderer
import torch
from pathlib import Path
from argparse import ArgumentParser

def render_target(config):
    # Scene setup
    scene = torch.load(config['cached_scene_path'])
    integrator_config = config['integrator'] 
    scene.add_integrator(integrator_config['type'], integrator_config['config'])
    scene.add_hdr_film(resolution=config['image_res'])

    num_sensors = len(scene.sensors)

    render = Renderer(config['connector_name'], device='cuda', dtype=torch.float32)
    render.set_render_options(config['render_options']['tar'])
    target_images = render(scene, sensor_ids=list(range(num_sensors)))

    target_images_path = Path(config['target_images_path'])
    target_images_path.mkdir(parents=True, exist_ok=True)
    for i, target_image in enumerate(target_images):
        write_exr(target_images_path / f'{i}.exr', target_image)
        write_png(target_images_path / f'{i}.png', target_image)
    print('Done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    render_target(config)

