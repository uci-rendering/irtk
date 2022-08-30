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
If you want to modify the code but don't want to repeatedly install them, add `-e` after `install`, such as `pip install -e .`.

## Run a simple test
I created a simple inverse rendering experiment in `tests/inv_rendering_test.py`. After all the requirements are met, run
```
python tests/inv_rendering_test.py
```
