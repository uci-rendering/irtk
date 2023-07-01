# Inverse Rendering Project Template
This is a template for using `ivt` to do inverse rendering experiments. 

## Dependencies
We use `conda` to manage dependencies. The dependencies are specified in `environment.yml`, including `ivt` and `psdr-jit`. Make sure you have `cudatoolkit` installed as specified in [https://pypi.org/project/psdr-jit/](https://pypi.org/project/psdr-jit/).

To create a new conda environment with all the dependencies:
```
conda env create -f environment.yml
```

To update it:
```
conda env update --name ivt --file environment.yml
```

## Quick Start
Run 
```bash
python run.py single_stage configs/synthetic/armadillo/config.gin
```
This should render the target images of a synthetic scene consisting of an armadillo, as specified in `configs/synthetic/armadillo/config.gin`. Then, an inverse rendering experiment will start, which try to reconstruct the material of the armadillo. Results can be found in `results/synthetic/armadillo`.

## Walk Through
This project uses [`gin-config`](https://github.com/google/gin-config) for configuration. It's important to familiarize it.
### `run.py`
```
python run.py [pipeline] [pipeline_config]
```
will run "pipeline" defined in `run.py` with the configuration specified in "pipeline-config". 

In the above example, the `single_stage` pipeline takes `dataset`, `stage_config`, and `result_path` as input. `dataset` provides target images and a `ivt.Scene`. `stage_config` is a path to the optimization configuration for `opt.py`. The `result_path` is the path to the optimization results. You can see how they are set in `configs/synthetic/armadillo/config.gin`:
```
...
single_stage.dataset = @SyntheticDataset()
single_stage.result_path = 'results/synthetic/armadillo'
single_stage.stage_config = 'configs/synthetic/armadillo/microfacet_basis.gin'
...
```

You can define your own pipelines in `run.py` to do more complicated tasks, such as mutli-stages optimizations.

### `opt.py`
`opt.py` implements the main optimization loop and it is usually used as a stage in pipeline. Its most important input is the `model_class`, which is an `ivt.Model` that defines an optimization process of some scene parameters. 

For instance, in the quick start example, the `model_class` is defined in `configs/synthetic/armadillo/microfacet_basis.gin`:
```
optimize.model_class = @MultiOpt
MultiOpt.model_classes = [
    @MicrofacetBasis,
]
MicrofacetBasis.mat_id = 'mat'
MicrofacetBasis.N = 50
MicrofacetBasis.s_max = 0.04
MicrofacetBasis.s_min = 0.04
MicrofacetBasis.r_min = 0.1
MicrofacetBasis.d_lr = 5e-2
MicrofacetBasis.r_lr = 1e-2
MicrofacetBasis.weight_map_lr = 5e-2
MicrofacetBasis.t_res = 1024
```
`MultiOpt` is defined in `ivt.model` and it's used to wrap multiple `ivt.Model` together. `MicrofacetBasis` is defined in `./models/microfacet_basis.py` and it's used to optimize a microfacet BRDF. 


### models
You can define your own models to design novel inverse rendering algorithm. See the ones in the `models` directory for inspiration. 

### datasets
You can define your own dataset formats in the `datasets` directory. 


