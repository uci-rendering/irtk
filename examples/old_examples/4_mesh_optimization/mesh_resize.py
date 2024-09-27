from irtk.io import *

input_file = './examples/data/meshes/sphere.obj'
output_file = f'./experiments/0_sphere_to_pig/data/meshes/sphere.obj'

v, f, uv, fuv = read_mesh(input_file)
v = to_torch_f(v) * 150.0

write_mesh(output_file, v, f, uv, fuv)