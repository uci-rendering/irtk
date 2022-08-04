from multiprocessing.sharedctypes import Value
from ivt.connector import Connector
import psdr_cpu
import numpy as np
import torch
import os

class PSDREnzymeConnector(Connector):
    backend = 'numpy'
    device = 'cpu'
    ftype = np.float64
    itype = np.int64
    
    def create_objects(self, scene):
        """
        Create psdr_cpu objects from the scene data.
        """
        
        scene.backend = PSDREnzymeConnector.backend
        scene.device = PSDREnzymeConnector.device
        scene.ftype = PSDREnzymeConnector.ftype
        scene.itype = PSDREnzymeConnector.itype 
        scene.configure()
        
        objects = {}
        
        integrator_config = scene.integrator
        integrator_dict = {
            'direct': psdr_cpu.Direct,
            'volpath': psdr_cpu.Volpath2,
            # 'collocated': psdr_cpu.Collocated,
        }
        if integrator_config['type'] in integrator_dict:
            objects['integrator'] = integrator_dict[integrator_config['type']]()
        else:
            raise ValueError(f"integrator type [{integrator_config['type']}] is not supported.")
            
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
        
        rfilters = {'tent': 0, 'box': 1, 'gaussian': 2}
        rfilter = rfilters[film_config['rfilter']]
        sensors = []
        for sensor_config in scene.sensors:
            sensor = psdr_cpu.Camera(width, height, 
                                     float(sensor_config['fov'].data), 
                                     sensor_config['origin'].data, 
                                     sensor_config['target'].data, 
                                     sensor_config['up'].data,
                                     rfilter)
            sensors.append(sensor)
        objects['sensors'] = sensors
        
        meshes = []
        for mesh_config in scene.meshes:
            mesh = psdr_cpu.Shape(mesh_config['vertex_positions'].data,
                                  mesh_config['vertex_indices'].data,
                                  mesh_config['uv_indices'].data, 
                                  [],
                                  mesh_config['vertex_positions'].data.shape[0],
                                  mesh_config['vertex_indices'].data.shape[0],
                                  -1, mesh_config['bsdf_id'], -1, -1)
            # handle medium
            if 'med_ext_id' in mesh_config:
                mesh.med_ext_id = mesh_config['med_ext_id']
            if 'med_int_id' in mesh_config:
                mesh.med_int_id = mesh_config['med_int_id']
            meshes.append(mesh)
        objects['meshes'] = meshes
            
        bsdfs = []
        for bsdf_config in scene.bsdfs:
            if bsdf_config['type'] == 'diffuse':
                reflectance_data = bsdf_config['reflectance'].data
                reflectance_shape = reflectance_data.shape
                if reflectance_shape == ():
                    r = reflectance_data.item()
                    bsdf = psdr_cpu.DiffuseBSDF(psdr_cpu.RGBSpectrum(r, r, r))
                elif reflectance_shape == (3, ):
                    r, g, b = reflectance_data
                    bsdf = psdr_cpu.DiffuseBSDF(psdr_cpu.RGBSpectrum(r, g, b))
                elif len(reflectance_data) == 3:
                    bsdf = psdr_cpu.DiffuseBSDF(psdr_cpu.RGBSpectrum(0, 0, 0))
                    bsdf.reflectance = psdr_cpu.Bitmap(bsdf_config['reflectance'].data.reshape(-1), reflectance_shape[:2])
            elif bsdf_config['type'] == 'null':
                bsdf = psdr_cpu.NullBSDF()
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

        phases = []
        for i, phase_config in enumerate(scene.phases):
            if phase_config['type'] == 'isotropic':
                phase = psdr_cpu.Isotropic()
            phases.append(phase)
        objects['phases'] = phases

        mediums = []
        for i, medium_config in enumerate(scene.mediums):
            if medium_config['type'] == 'homogeneous':
                medium = psdr_cpu.Homogeneous(medium_config['sigmaT'].data,
                                              medium_config['albedo'].data,
                                              medium_config['phase_id'])
                mediums.append(medium)
        objects['mediums'] = mediums

        return objects
    
    def renderC(self, scene, sensor_ids=[0]):    
        objects = self.create_objects(scene)
        
        psdr_scene = psdr_cpu.Scene()
        psdr_scene.shapes = objects['meshes']
        psdr_scene.bsdfs = objects['bsdfs']
        psdr_scene.emitters = objects['emitters']
        psdr_scene.phases = objects['phases']
        psdr_scene.mediums = objects['mediums']
        
        images = []
        for sensor_id in sensor_ids:
            psdr_scene.camera = objects['sensors'][sensor_id]
            psdr_scene.configure()
            image = objects['integrator'].renderC(psdr_scene, objects['render_options'])
            images.append(image.reshape(objects['film']['shape']))
        
        return images
    
    def renderD(self, image_grads, scene, sensor_ids=[0]):
        assert len(image_grads) == len(sensor_ids) and len(image_grads) > 0

        t_dtype = image_grads[0].dtype
        t_device = image_grads[0].device
        
        # Transform the parameter names to extrac the gradient later.
        param_names = scene.get_requiring_grad()
        param_grads = []
        for i, param_name in enumerate(param_names):
            param = scene.param_map[param_name]
            if param_name.startswith('meshes'):
                param_name = param_name.replace('meshes', 'shapes')
                if param_name.endswith('vertex_positions'):
                    param_name = param_name.replace('vertex_positions', 'vertices')
            param_names[i] = param_name
            param_grads.append(torch.zeros(param.data.shape, dtype=t_dtype, device=t_device))
            
        objects = self.create_objects(scene)
        
        psdr_scene = psdr_cpu.Scene()
        psdr_scene.shapes = objects['meshes']
        psdr_scene.bsdfs = objects['bsdfs']
        psdr_scene.emitters = objects['emitters']
        psdr_scene.phases = objects['phases']
        psdr_scene.mediums = objects['mediums']
        
        for i, sensor_id in enumerate(sensor_ids):
            psdr_scene.camera = objects['sensors'][sensor_id]
            psdr_scene.configure()

            # Process image_grad
            image_grad = np.array(image_grads[i].detach().cpu().numpy(), dtype=PSDREnzymeConnector.ftype)
            image_grad = image_grad.reshape(-1)
            
            # Estimate the interior integral.
            psdr_scene_ad = psdr_cpu.SceneAD(psdr_scene)
            objects['integrator'].renderD(psdr_scene_ad, objects['render_options'], image_grad)
            
            # Estimate the boundary integral.
            boundary_integrator = psdr_cpu.BoundaryIntegrator(psdr_scene)
            boundary_integrator.renderD(psdr_scene_ad, objects['render_options'], image_grad)
            
            # Extrac the gradient.
            for j, param_name in enumerate(param_names):
                grad = eval("psdr_scene_ad.der." + param_name)
                if isinstance(grad, psdr_cpu.Bitmap):
                    grad = grad.m_data
                grad = np.array(grad, dtype=PSDREnzymeConnector.ftype)
                param_grads[j] += torch.tensor(grad, dtype=t_dtype, device=t_device)
        
        return param_grads
