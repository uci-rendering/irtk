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
        
        objects = {}
        
        integrator_config = scene.integrator
        if integrator_config['type'] == 'direct':
            objects['integrator'] = psdr_cpu.Direct()
            
        objects['render_options'] = psdr_cpu.RenderOptions(
            scene.render_options['seed'],
            scene.render_options['num_samples'],
            scene.render_options['max_bounces'],
            scene.render_options['num_samples_primary_edge'],
            scene.render_options['num_samples_secondary_edge'],
            scene.render_options['quiet'],
        )
        
        film_config = scene.film
        width, height = film_config['resolution']
        objects['film'] = {
            'shape': (width, height, 3)
        }
        
        sensors = []
        for sensor_config in scene.sensors:
            sensor = psdr_cpu.Camera(width, height, 
                                     float(sensor_config['fov'].data), 
                                     sensor_config['origin'].data, 
                                     sensor_config['target'].data, 
                                     sensor_config['up'].data)
            sensors.append(sensor)
        objects['sensors'] = sensors
        
        meshes = []
        for mesh_config in scene.meshes:
            mesh = psdr_cpu.Shape(mesh_config['vertex_positions'].data,
                                  mesh_config['vertex_indices'].data,
                                  mesh_config['uv_indices'].data, 
                                  mesh_config['vertex_normals'].data,
                                  mesh_config['vertex_positions'].data.shape[0],
                                  mesh_config['vertex_indices'].data.shape[0],
                                  -1, mesh_config['bsdf_id'], -1, -1)
            meshes.append(mesh)
        objects['meshes'] = meshes
            
        bsdfs = []
        for bsdf_config in scene.bsdfs:
            if bsdf_config['type'] == 'diffuse':
                bsdf = psdr_cpu.DiffuseBSDF(psdr_cpu.RGBSpectrum(0, 0, 0))
                reflectance_shape = bsdf_config['reflectance'].data.shape
                bsdf.reflectance = psdr_cpu.Bitmap(bsdf_config['reflectance'].data.reshape(-1), reflectance_shape[:2])
            bsdfs.append(bsdf)
        objects['bsdfs'] = bsdfs
            
        emitters = []
        for i, emitter_config in enumerate(scene.emitters):
            if emitter_config['type'] == 'area':
                r, g, b = emitter_config['radiance'].data
                emitter = psdr_cpu.AreaLight(emitter_config['mesh_id'], psdr_cpu.RGBSpectrum(r, g, b))
                meshes[emitter_config['mesh_id']].light_id = i
                emitters.append(emitter)
        objects['emitters'] = emitters
        
        return objects
    
    def renderC(self, scene, sensor_ids=[0]):    
        objects = self.create_objects(scene)
        images = []
        
        psdr_scene = psdr_cpu.Scene()
        psdr_scene.shapes = objects['meshes']
        psdr_scene.bsdfs = objects['bsdfs']
        psdr_scene.emitters = objects['emitters']
        
        for sensor_id in sensor_ids:
            psdr_scene.camera = objects['sensors'][sensor_id]
            psdr_scene.configure()
            image = objects['integrator'].renderC(psdr_scene, objects['render_options'])
            images.append(image.reshape(objects['film']['shape']))
        
        return images
    
    def renderD(self, scene, target_image, image_loss_func, sensor_ids=[0]):
        pass