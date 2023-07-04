from opt import optimize

from argparse import ArgumentParser, RawTextHelpFormatter
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
def single_stage(
        dataset, 
        stage_config, 
        result_path):
    gin.parse_config_file(stage_config)
    result_path = Path(result_path, get_time_str(), Path(stage_config).stem)
    optimize(scene=dataset.get_scene(), 
             dataset=dataset,
             result_path=result_path)
    
@add_pipeline
@gin.configurable
def multi_stage(
        dataset, 
        stage_configs, 
        result_path):
    
    time_str = get_time_str()

    scene = dataset.get_scene()
    
    for stage_config in stage_configs:
        gin.parse_config_file(stage_config)
        sub_result_path = Path(result_path, time_str, Path(stage_config).stem)
        scene = optimize(scene=scene, 
                dataset=dataset,
                result_path=sub_result_path)
    
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('pipeline_config', type=str, 
                        help='pipeline config file of the form "**/{pipeline_name}.gin\n'
                        'available pipelines: \n' + '\n'.join(pipelines))
    args = parser.parse_args()

    gin.parse_config_file(args.pipeline_config)
    pipelines[Path(args.pipeline_config).stem]()