# inv-render-toolkit
Inverse-Rendering Toolkit

# Quick Start
## Create a virtual environment
Clone the repository. Make sure you have Anaconda installed. To avoid package conflicts, create a virtual environment using Anaconda via
```
conda env create -f environment.yml
conda activate ivt
```
If `environment.yml` is updated and you want to update your environemnt accordingly, you can run
```
conda env update --name ivt --file environment.yml
```
Add a `--prune` at the end if you want to remove extra packages not listed in `environment.yml`.

## Install the backends
For now, only `psdr-enzyme` is supported. A note for compiling `psdr-enzyme`:
Since we are using a virtual environment `ivt`, which has its own Python, you need to compile `psdr-enzyme` with the correct Python version.
If CMake fails to find the correct one during configuration, try adding `-DPython_EXECUTABLE=$(which python)` (assuming `ivt` environment is activated and `python` is referring to the Anaconda one) when running CMake.

## Install the ivt packages
At the root directory of the cloned repo, run 
```
pip install . 
pip install connectors/*
pip install scene_parsers/*
```
If you want to modify the code but don't want to repeatedly install them, add `-e` after `install` (highly recommended if you are going to change the code), such as `pip install -e .`.

Notice that you can't use wildcards directly in Powershell. You can do the following instead
```
pip install . 
pip install (get-item .\connectors\*)
pip install (get-item .\scene_parsers\*)
```

## Run examples
There are some inverse rendering examples in the `examples` directory. Follow the instructions below to run them.

First, go the `examples` directory:
```
cd examples
```
Create a scene with a armadillo and cache it. This cached scene will be used to render the target images and initialize the optimization.
```
python scenes/armadillo.py
```
Pick a config file from the `configs` directory, such as `armadillo_joint_ch.json`. It contains settings of a particular inverse rendering problem. Render the target images for this config file:
```
python render_target.py configs/[config_file]
```
Finally, run the optimization:
```
python opt.py configs/[config_file]
```
Some inverse rendering problems might share the same set of target images, so you might not need to rerender the target images when switching tasks. Check the config file for details.


