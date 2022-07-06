from sqlite3 import connect
from psdr_cuda_connector import PSDRCudaConnector
from common import *

scene = simple_scene()
scene.add_render_options(simple_render_options['psdr_cuda'])

connector = PSDRCudaConnector()
connector.renderC(scene)
