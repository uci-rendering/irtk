import json
from ivt.io import *
from ivt.renderer import Renderer
from ivt.loss import l1_loss, mesh_laplacian_smoothing
from ivt.optimizers import LargeStepsOptimizer
import torch
from pathlib import Path
from time import time
import asyncio
from argparse import ArgumentParser

async def write_results(out_path, cp_data):
    out_path.mkdir(parents=True, exist_ok=True)
    tar_images_cat = torch.cat([tar_image for tar_image in cp_data['tar_images']], dim=1)
    vis_images_cat = torch.cat([vis_image for vis_image in cp_data['vis_images']], dim=1)
    final_image = torch.cat([tar_images_cat, vis_images_cat], dim=0)
    write_exr(out_path / f'vis.exr', final_image)

    write_obj(
        out_path / 'opt.obj', 
        cp_data['V'], 
        cp_data['F'])

    torch.save(cp_data['loss'], out_path.parent / 'loss.pt')

def optimize_shape(config):
    result_path = Path(config['result_path'])
    result_path.mkdir(parents=True, exist_ok=True) 

    # Get target images 
    print('Reading target images..')
    tar_images = []
    target_images_path = Path(config['target_images_path'])
    for target_image_path in sorted(target_images_path.glob('*.exr'), key=lambda x : int(x.stem)):
        tar_images.append(torch.from_numpy(read_exr(target_image_path)))
    tar_images = torch.stack(tar_images).to('cuda').to(torch.float32)

    # Configure scene
    print('Setting up the scene...')
    scene = torch.load(config['cached_scene_path'])
    integrator_config = config['integrator'] 
    scene.add_integrator(integrator_config['type'], integrator_config['config'])
    scene.add_hdr_film(resolution=config['image_res'])

    # Handle sensors
    vis_sensor_ids = torch.tensor(config['vis_sensor_ids'])
    opt_sensor_ids = list(range(len(scene.sensors)))
    for vis_sensor_id in vis_sensor_ids:
        opt_sensor_ids.remove(vis_sensor_id)
    opt_sensor_ids = torch.tensor(opt_sensor_ids)
    num_opt_sensors = len(opt_sensor_ids)

    # Create render functions
    render = Renderer(config['connector_name'], device='cuda', dtype=torch.float32)
    render.set_render_options(config['render_options']['opt'])

    render_vis = Renderer(config['connector_name'], device='cuda', dtype=torch.float32)
    render_vis.set_render_options(config['render_options']['vis'])

    # Create objects for shape optimization
    if 'init_mesh_path' in config: 
        v, tc, n, f, ftc, fn = read_obj(config['init_mesh_path'])
        scene.meshes[0]['vertex_positions'].set(v)
        scene.meshes[0]['vertex_indices'].set(f)

    V = scene.meshes[0]['vertex_positions'].requires_grad_()
    F = scene.meshes[0]['vertex_indices']

    # Optimizer related stuff
    obj_optimizer = LargeStepsOptimizer(
        V.data, F.data, 
        lr=config['obj_lr'], lmbda=config['obj_lambda'])
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    loss_record = []
    iter = 0

    print('Starting the optimization...')
    for epoch in range(1, num_epochs + 1):
        
        # Randomly shuffle sensors
        sensor_perm = opt_sensor_ids[torch.randperm(num_opt_sensors)]
        sensor_batches = torch.split(sensor_perm, batch_size)

        for sensor_ids in sensor_batches:
            t0 = time()

            obj_optimizer.zero_grad()
            
            opt_images = render(scene, sensor_ids=sensor_ids)
            
            loss = l1_loss(tar_images[sensor_ids], opt_images)
            loss.backward()

            obj_optimizer.step()

            t1 = time()

            loss_record.append(loss.item())
            
            iter += 1
            print(f"[Epoch {epoch}/{num_epochs}] iter: {iter} | loss: {loss.item()} | time: {t1 - t0}")

            if iter == 1 or iter % config['checkpoint_iter'] == 0:
                vis_images = render_vis(scene, sensor_ids=vis_sensor_ids)
                cp_data = {
                    'tar_images': tar_images[vis_sensor_ids],
                    'vis_images': vis_images,
                    'V': V.data,
                    'F': F.data,
                    'loss': loss_record
                }

                asyncio.run(write_results(result_path / str(iter), cp_data))

    print('Done')

    return V.data.detach().cpu().numpy(), F.data.detach().cpu().numpy()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    optimize_shape(config)

