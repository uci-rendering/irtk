import irtk
from irtk.io import *
from irtk.metric import chamfer_distance

import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) >= 2:
    renderer = sys.argv[1]
elif len(irtk.get_connector_list()) == 1:
    renderer = irtk.get_connector_list()[0]
else:
    print("Please specify backend renderer. Currently available backend(s):")
    print(irtk.get_connector_list())
    exit()
    
mesh_target = 'armadillo_highres'
output_folder = 'armadillo_remesh'
num_mesh_samples = 1000000

v, f, uv, fuv = read_mesh(f'./examples/data/meshes/{mesh_target}.obj')
v = v * 0.007 - [0, 0.1, 0]
mesh_ref = { 'v': v, 'f': f }

file_prefix = f'output/4_mesh_optimization/{output_folder}/{renderer}/{renderer}'
file_list = os.listdir(f"{file_prefix}_mesh")
file_list.sort(key=lambda x:int(x[5:-4]))

iter = []
mesh_error = []
max_error = 0
min_error = 1000
min_error_iter = 0
for file in file_list:
    print(f"Evaluating mesh {file[5:-4]}...", end='', flush=True)
    v, f, uv, fuv = read_mesh(f"{file_prefix}_mesh/{file}")
    mesh_opt = { 'v': v, 'f': f }
    error = chamfer_distance(mesh_ref, mesh_opt, num_mesh_samples)
    mesh_error.append(error)
    
    iter.append(int(file[5:-4]))
    
    if max_error < error:
        max_error = error
    if min_error > error:
        min_error = error
        min_error_iter = int(file[5:-4])
        
    print(f"Error: {error}")

final_error = error

plt.title('Mesh Error')
plt.xlabel('iter')
plt.ylabel('error')
plt.ylim(bottom=0, top=max_error * 1.05)
plt.plot(iter, mesh_error, label='error')
plt.savefig(file_prefix + '_mesh_error.png')

output_file = open(file_prefix + '.txt', 'a')
output_file.write(f"Final mesh error: {final_error:.4g}\n")
output_file.write(f"Best mesh error: {min_error:.4g} at iter {min_error_iter}\n")
output_file.close()

output_file = open(file_prefix + '_errors.txt', 'w')
for i, e in zip(iter, mesh_error):
    output_file.write(f"Iter {i} mesh error: {e:.4g}\n")
output_file.close()