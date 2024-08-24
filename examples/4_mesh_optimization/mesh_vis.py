import irtk
from irtk.scene import *
from irtk.renderer import Renderer
from irtk.io import *

import imageio
from skimage import img_as_ubyte
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
    

output_folder = 'armadillo_remesh'
max_gif_duration = 5000 # ms

file_prefix = f'output/4_mesh_optimization/{output_folder}/{renderer}/{renderer}'
mesh_dir = f'{file_prefix}_mesh'

file_list = os.listdir(mesh_dir)
file_list.sort(key=lambda x:int(x[5:-4]))

images_gif = []
for file in file_list:

    print(f"Visualizing {file}...")
    scene = Scene()
    scene.set('object', Mesh.from_file(f'{mesh_dir}/{file}', mat_id='blue'))
    scene['object']['uv'] = []
    scene['object']['fuv'] = []
    scene.set('blue', DiffuseBRDF((0.2, 0.2, 0.9)))
    scene.set('sensor_main', PerspectiveCamera.from_lookat(fov=40, origin=(1.5, 1.5, -1.5), target=(0, 0, 0), up=(0, 1, 0)))
    scene.set('film', HDRFilm(width=512, height=512))

    if renderer == 'psdr_jit':
        scene.set('integrator', Integrator(type='collocated', config={
            'intensity': 20
        }))

        render = Renderer('psdr_jit', render_options={
            'spp': 128,
            'sppe': 0,
            'sppse': 0,
            'log_level': 0,
            'npass': 1
        })
    elif renderer == 'pytorch3d':
        render = Renderer(renderer, render_options={
            'npass': 1,
            'light_diffuse_color': (1.2, 1.2, 1.2)
        })
    elif renderer == 'nvdiffrast':
            render = Renderer('nvdiffrast', render_options={
            'npass': 1,
            'light_power': 3.0
        })
    elif renderer == 'mitsuba':
        scene.set('integrator', Integrator(type='path', config={
            'max_depth': 4,
            'hide_emitters': False
        }))
        
        render = Renderer('mitsuba', render_options={
            'spp': 128,
            'npass': 1,
            'point_light_intensity': 20
        })
    elif renderer == 'redner':
        scene.set('integrator', Integrator(type='path', config={
            'max_depth': 1,
            'hide_emitters': False
        }))

        render = Renderer('redner', render_options={
            'spp': 64,
            'npass': 1,
            'light_intensity': [27.0, 27.0, 27.0]
        })

    image = render(scene)[0]
    image_gif = img_as_ubyte(to_srgb(image))
    images_gif.append(image_gif)
    duration_this_frame = 0

    scene.clear_cache()

imageio.mimsave(file_prefix + '_mesh_vis.gif', images_gif, duration=max_gif_duration / len(file_list), loop=0)