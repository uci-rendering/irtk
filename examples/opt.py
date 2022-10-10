import json
from ivt.io import *
from ivt.renderer import Renderer
from ivt.loss import l1_loss, mesh_laplacian_smoothing
from ivt.model import *
import torch
from pathlib import Path
from time import time
from argparse import ArgumentParser

import gin

@gin.configurable
def optimize(
    scene, 
    film_res,
    vis_sensor_ids, 
    model_class, 
    num_epochs,
    batch_size,
    checkpoint_iter,
    result_path,
    target_images_path,
    integrator_type,
    integrator_config,
    render_opt,
    render_vis,
    render_tar=None,
    should_render_target=False,
    ):
    
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True) 

    # Configure scene
    print('Setting up the scene...')
    scene.add_integrator(integrator_type, integrator_config)
    scene.add_hdr_film(resolution=film_res)
    
    # Render target images if needed
    if should_render_target:
        render_target(
            scene, 
            render_tar,
            target_images_path)
    
    # Get target images 
    print('Reading target images..')
    tar_images = []
    target_images_path = Path(target_images_path)
    assert target_images_path.exists(), f"{target_images_path} doesn't exist!"
    for target_image_path in sorted(target_images_path.glob('*.exr'), key=lambda x : int(x.stem)):
        tar_images.append(torch.from_numpy(read_exr(target_image_path)))
    tar_images = torch.stack(tar_images).to('cuda').to(torch.float32)

    # Handle sensors
    vis_sensor_ids = torch.tensor(vis_sensor_ids)
    opt_sensor_ids = list(range(len(scene.sensors)))
    for vis_sensor_id in vis_sensor_ids:
        opt_sensor_ids.remove(vis_sensor_id)
    opt_sensor_ids = torch.tensor(opt_sensor_ids)
    num_opt_sensors = len(opt_sensor_ids)

    # Create render functions
    loss_record = []
    iter = 0

    model = model_class(scene)

    print('Starting the optimization...')
    for epoch in range(1, num_epochs + 1):
        
        # Randomly shuffle sensors
        sensor_perm = opt_sensor_ids[torch.randperm(num_opt_sensors)]
        sensor_batches = torch.split(sensor_perm, batch_size)

        for sensor_ids in sensor_batches:
            t0 = time()

            model.zero_grad()
            model.set_data()
            
            opt_images = render_opt(scene, sensor_ids=sensor_ids)
            
            loss = l1_loss(tar_images[sensor_ids], opt_images)
            loss += model.get_regularization()
            loss.backward()

            model.step()

            t1 = time()

            loss_record.append(loss.item())
            
            iter += 1
            print(f"[Epoch {epoch}/{num_epochs}] iter: {iter} | loss: {loss.item()} | time: {t1 - t0}")

            if iter == 1 or iter % checkpoint_iter == 0:
                iter_path = result_path / str(iter)

                model.write_results(iter_path)

                tar_images_cat = torch.cat([tar_image for tar_image in tar_images[vis_sensor_ids]], dim=1)

                vis_images = render_vis(scene, sensor_ids=vis_sensor_ids)
                vis_images_cat = torch.cat([vis_image for vis_image in vis_images], dim=1)

                final_image = torch.cat([tar_images_cat, vis_images_cat], dim=0)
                write_exr(iter_path / 'vis.exr', final_image)

    print('Done')

def render_target(
    scene, 
    render_tar,
    target_images_path):

    assert render_tar is not None, 'render_tar must be set in order to rendering target images!'

    # Scene setup
    num_sensors = len(scene.sensors)

    print('Rendering target images...')
    target_images = render_tar(scene, sensor_ids=list(range(num_sensors)))

    target_images_path = Path(target_images_path)
    target_images_path.mkdir(parents=True, exist_ok=True)
    for i, target_image in enumerate(target_images):
        write_exr(target_images_path / f'{i}.exr', target_image)
        write_png(target_images_path / f'{i}.png', target_image)

    scene.clear_cache()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('gin_config_file', type=str)
    parser.add_argument('--render_target', '-r', action='store_true')
    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)
    optimize(should_render_target=args.render_target)

