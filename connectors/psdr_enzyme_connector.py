from venv import create
from ivt.connector import Connector
import psdr_cpu
import numpy as np

class PSDREnzymeConnector(Connector):
    
    def create_objects(self, scene):
        scene.backend = 'numpy'
        scene.device = 'cpu'
        scene.ftype = np.float64
        scene.itype = np.int64 
        scene.configure()
        
        film_config = scene.film
        
        width, height = film_config['resolution']
        sensors = []
        for sensor_config in scene.sensors:
            sensor = psdr_cpu.Camera(width, height, 
                                     float(sensor_config['fov'].data), 
                                     sensor_config['origin'].data, 
                                     sensor_config['target'].data, 
                                     sensor_config['up'].data)
            sensors.append(sensor)
            
        # for mesh_config in scene['meshes']:
        #     mesh = psdr_cpu.Shape(mesh_config)
        
        objects = {
            'sensors': sensors
        }
        
        return objects
    
    def renderC(self, scene, sensor_ids=[0]):
        
        objects = self.create_objects(scene)
        
        
            
    
    def renderD(self, scene, target_image, image_loss_func, sensor_ids=[0]):
        pass