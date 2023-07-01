from opt import optimize

from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import gin

pipelines = {}
def add_pipeline(pipeline):
    pipelines[pipeline.__name__] = pipeline
    def decorator(*args, **kwargs):
        return pipeline(*args, **kwargs)
    return decorator

def get_time_str():
    return datetime.now().strftime("%Y_%m_%d-%p%I_%M_%S")

@add_pipeline
@gin.configurable
def single_stage(dataset, stage_config, result_path):
    gin.parse_config_file(stage_config)
    result_path = Path(result_path, get_time_str(), Path(stage_config).stem)
    optimize(scene=dataset.get_scene(), 
             dataset=dataset,
             result_path=result_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('pipeline', type=str, help='pipeline to run', choices=pipelines.keys())
    parser.add_argument('pipeline_config', type=str, help='config file for the pipeline')
    args = parser.parse_args()

    gin.parse_config_file(args.pipeline_config)
    pipelines[args.pipeline]()