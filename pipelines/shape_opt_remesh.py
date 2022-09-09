from pathlib import Path
from shape_opt import optimize_shape
from ivt.io import * 
import shutil

import json
from argparse import ArgumentParser

def optimize_shape_with_remeshing(config):
    opt_obj_path = Path(config['result_path'], 'opt.obj')
    opt_obj_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(config['init_mesh_path'], opt_obj_path)
    config['init_mesh_path'] = str(opt_obj_path)

    v, _, _, f, _, _ = read_obj(config['init_mesh_path'])

    edge_length = config['init_edge_length']
    edge_length_decay = config['edge_length_decay']

    result_path = Path(config['result_path'])

    for i in range(config['num_remesh']):
        v, f = fix_mesh(v, f)
        v, f = remesh(v, f, edge_length=edge_length)
        v, f = fix_mesh(v, f)
        write_obj(config['init_mesh_path'], v, f)

        config['result_path'] = result_path / str(i)
        v, f = optimize_shape(config)
        
        edge_length *= edge_length_decay

    write_obj(config['init_mesh_path'], v, f)

    return v, f

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    optimize_shape_with_remeshing(config)
