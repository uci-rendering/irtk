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
Since we are using a virtual environment `ivt`, which has its own Python, you need to compile `psdr` with the correct Python version.
If CMake fails to find the correct one during configuration, try adding `-DPython_EXECUTABLE=$(which python)` (assuming `ivt` environment is activated and `python` is referring to the Anaconda one) when running CMake.

## Install the ivt packages
At the root directory of the cloned repo, run 
```
pip install . 
```
If you want to modify the code but don't want to repeatedly install them, add `-e` after `install` (highly recommended if you are going to change the code),
```
pip install -e .
```

## Run examples
There are some inverse rendering examples in the `examples` directory. Follow the instructions below to run them.

First, go the `examples` directory:
```
cd examples
```
Choose a config file from the `configs` directory. Take `configs/armadillo/joint_ch.gin` as an example: it contains configurations of a joint optimization problem, which aims to reconstruct the shape and material of an armadillo simultaneously. Use the follow line to perform the optimization:
```
python opt.py configs/armadillo/joint_ch.gin
```
If this is the first time you run this scene, you **will** run into error because there are no target images. Add a `--render_target` or `-r` flag will render the target images (if the scene is synthetic) before doing the optimization.
```
python opt.py -r configs/armadillo/joint_ch.gin
```
Some inverse rendering problems might share the same set of target images, so you might not need to rerender the target images when switching tasks. Check the config file for details.
