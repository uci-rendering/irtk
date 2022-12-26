from ivt.connector import Connector
from ivt.scene_parser import get_scene_parser
from ivt.scene import split_param_name
from ivt.transform import lookat
from ivt.io import write_exr


import drjit
import psdr_jit
from drjit.cuda import Array3f as Vector3fC
from drjit.cuda.ad import Array3f as Vector3fD, Float32 as FloatD, Matrix4f as Matrix4fD, Matrix3f as Matrix3fD
from drjit.cuda.ad import Float32 as FloatD
import torch

from pathlib import Path
from shutil import rmtree
import os
import time

class PSDRJITConnector(Connector):

    connector_name = 'psdr_jit'

    backend = 'torch'
    device = 'cuda'
    ftype = torch.float32
    itype = torch.long

    def __init__(self):
        super().__init__()

    def create_scene(self, scene):
        # Create a temporary scene directory with all the data
        tmp_path = Path("__psdr_jit_tmp__")
        tmp_scene_path = tmp_path / 'scene.xml'
        sp = get_scene_parser('mitsuba')
        sp.write(tmp_scene_path, scene)
        
        # Load the scene 
        old_path = os.getcwd()
        os.chdir(tmp_path)
        psdr_scene = psdr_jit.Scene()
        psdr_scene.load_file('scene.xml', False)
        os.chdir(old_path)
        # Clean up
        rmtree(tmp_path)
        # exit()
        return psdr_scene

    def create_objects(self, scene, render_options):
        objects = {}
        
        psdr_scene = self.create_scene(scene)
        objects['scene'] = psdr_scene

        psdr_scene.opts.spp = render_options['spp']
        psdr_scene.opts.sppe = render_options['sppe']
        psdr_scene.opts.sppse = render_options['sppse']
        psdr_scene.opts.log_level = render_options['log_level']

        psdr_scene.configure()

        objects['integrators'] = []
        for it in range(len(scene.integrators)):
            integrator_config = scene.integrators[it]
            if integrator_config['type'] == 'direct':
                objects['integrators'].append(psdr_jit.DirectIntegrator())
                if 'hide_envmap' in integrator_config['params']:
                    objects['integrators'][it].hide_emitters = integrator_config['params']['hide_envmap']
            elif integrator_config['type'] == 'collocated':
                objects['integrators'].append(psdr_jit.CollocatedIntegrator(integrator_config['params']['intensity']))
            elif integrator_config['type'] == 'field':
                objects['integrators'].append(psdr_jit.FieldExtractionIntegrator(integrator_config['params']['name']))
            elif integrator_config['type'] == 'path':
                objects['integrators'].append(psdr_jit.PathTracer(integrator_config['params']['max_depth']))
                if 'hide_envmap' in integrator_config['params']:
                    objects['integrators'][it].hide_emitters = integrator_config['params']['hide_envmap']
            else:
                raise ValueError(f"integrator type [{integrator_config['type']}] is not supported.")
        
        objects['scene'].opts.spp = render_options['spp']
        objects['scene'].opts.sppe = render_options['sppe']
        objects['scene'].opts.sppse = render_options['sppse']
        objects['scene'].opts.log_level = render_options['log_level']
        
        film_config = scene.film
        height, width = film_config['resolution']
        objects['film'] = {
            'shape': (height, width, 3)
        }

        return objects

    def get_objects(self, scene, render_options):
        def convert_color(color, c=3):
            if c is None:
                return color.reshape(-1, )
            if color.shape == ():
                return color.tile(c)
            else:
                return color.reshape(-1, c)

        # Update the cached scene objects with new values
        if 'psdr_jit' in scene.cached:
            objects = scene.cached['psdr_jit']
            psdr_param_map = objects['scene'].param_map
            updated_params = scene.get_requiring_grad()

            for param_name in updated_params:
                param = scene[param_name]
                group, idx, prop = split_param_name(param_name)

                if group == 'meshes':
                    drjit_mesh = psdr_param_map[f'Mesh[{idx}]']
                    if prop == 'vertex_positions':
                        drjit_param = Vector3fD(param.data)
                        drjit_mesh.vertex_positions = drjit_param
                    elif prop == 'to_world':
                        drjit_param = Matrix4fD(param.data.reshape(1, 4, 4))
                        drjit_mesh.to_world = drjit_param

                elif group == 'bsdfs': 
                    bsdf_type = scene.bsdfs[idx]['type']
                    drjit_bsdf = psdr_param_map[f'BSDF[{idx}]']
                    
                    if bsdf_type == 'diffuse':
                        if prop == 'reflectance':
                            drjit_param = Vector3fD(convert_color(param.data))
                            drjit_bsdf.reflectance.data = drjit_param
                        elif prop == 'to_world':
                            drjit_param = Matrix3fD(param.data.reshape(1, 3, 3))
                            drjit_bsdf.reflectance.to_world = drjit_param

                    elif bsdf_type == 'microfacet':
                        if prop == 'diffuse_reflectance':
                            drjit_param = Vector3fD(convert_color(param.data))
                            drjit_bsdf.diffuseReflectance.data = drjit_param
                        elif prop == 'specular_reflectance':
                            drjit_param = Vector3fD(convert_color(param.data))
                            drjit_bsdf.specularReflectance.data = drjit_param
                        elif prop == 'roughness':
                            drjit_param = FloatD(convert_color(param.data, c=None))
                            drjit_bsdf.roughness.data = drjit_param

                elif group == 'sensors':
                    sensor = scene.sensors[idx]
                    sensor_type = sensor['type']
                    drjit_sensor = psdr_param_map[f'Sensor[{idx}]']

                    if sensor_type == 'perspective':
                        if prop == 'origin':
                            to_world = lookat(sensor['origin'].data, sensor['target'].data, sensor['up'].data).to('cuda').to(torch.float32)
                            drjit_sensor.to_world = Matrix4fD(to_world.reshape(1, 4, 4))
                
                elif group == 'emitters':
                    emitter = scene.emitters[idx]
                    emitter_type = emitter['type']
                    drjit_emitter = psdr_param_map[f'Emitter[{idx}]']

                    if emitter_type == 'env':
                        if prop == 'to_world':
                            drjit_param = Matrix4fD(param.data.reshape(1, 4, 4))
                            drjit_emitter.to_world = drjit_param
                        elif prop == 'env_map':
                            drjit_param = Vector3fD(convert_color(param.data))
                            drjit_emitter.radiance.data = drjit_param
                            
            return objects

        # Create scene objects if there is no cache
        else:
            objects = self.create_objects(scene, render_options)
            scene.cached['psdr_jit'] = objects
            return objects

    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        # Convert the scene into psdr_jit objects
        # t0 = time.time()
        objects = self.get_objects(scene, render_options)
        # t1 = time.time()
        # print(f"Time for [renderC.get_objects]: {t1 - t0}")
        objects['scene'].opts.spp = render_options['spp_c'] if 'spp_c' in render_options else render_options['spp']
        # t2 = time.time()
        objects['scene'].configure(sensor_ids)
        # print(f"Time for [renderC.configure2]: {t2 - t1}")
        # objects['scene'].configure()
        h, w, c = objects['film']['shape']
        npass = render_options['npass']

        # Render the images
        images = []
        for sensor_id in sensor_ids:
            image = torch.zeros((h * w, c)).to(PSDRJITConnector.device).to(PSDRJITConnector.ftype)
            for i in range(npass):
                image_pass = objects['integrators'][integrator_id].renderC(objects['scene'], sensor_id).torch()
                # write_exr('s' + str(sensor_id) + 'p_' + str(i) + '.exr', image_pass.reshape(h, w, c))
                image += image_pass / npass
            image = image.reshape(h, w, c)
            images.append(image)
        # drjit.registry_trim()
        # t3 = time.time()
        # print(f"Time for [renderC.renderC]: {t3 - t2}")
        return images
    
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        # Convert the scene into psdr_jit objects
        # t0 = time.time()
        objects = self.get_objects(scene, render_options)
        # t1 = time.time()
        # print(f"Time for [renderD.get_objects]: {t1 - t0}")
        objects['scene'].opts.spp = render_options['spp']
        psdr_param_map = objects['scene'].param_map
        npass = render_options['npass']

        param_names = scene.get_requiring_grad()
        drjit_params = []
        for param_name in param_names:
            param = scene[param_name]
            group, idx, prop = split_param_name(param_name)
            if group == 'meshes':
                drjit_mesh = psdr_param_map[f'Mesh[{idx}]']
                if prop == 'vertex_positions':
                    drjit_param = drjit_mesh.vertex_positions
                    drjit.enable_grad(drjit_param)
                    drjit_params.append(drjit_param)
                elif prop == 'to_world':
                    drjit_param = drjit_mesh.to_world
                    drjit.enable_grad(drjit_param)
                    drjit_params.append(drjit_param)

            elif group == 'bsdfs':
                bsdf_type = scene.bsdfs[idx]['type']
                drjit_bsdf = objects['scene'].param_map[f'BSDF[{idx}]']
                if bsdf_type == 'diffuse':
                    if prop == 'reflectance':
                        drjit_param = drjit_bsdf.reflectance.data 
                        drjit.enable_grad(drjit_param)
                        drjit_params.append(drjit_param)
                    elif prop == 'to_world':
                        drjit_param = drjit_bsdf.reflectance.to_world 
                        drjit.enable_grad(drjit_param)
                        drjit_params.append(drjit_param)
                    else:
                        raise ValueError(f"property not supported: {prop}")
                elif bsdf_type == 'microfacet':
                    if prop == 'diffuse_reflectance':
                        drjit_param = drjit_bsdf.diffuseReflectance.data
                        drjit.enable_grad(drjit_param)
                        drjit_params.append(drjit_param)
                    elif prop == 'specular_reflectance':
                        drjit_param = drjit_bsdf.specularReflectance.data
                        drjit.enable_grad(drjit_param)
                        drjit_params.append(drjit_param)
                    elif prop == 'roughness':
                        drjit_param = drjit_bsdf.roughness.data
                        drjit.enable_grad(drjit_param)
                        drjit_params.append(drjit_param)
                    else:
                        raise ValueError(f"property not supported: {prop}")
            elif group == 'emitters':
                emitter_type = scene.emitters[idx]['type']
                drjit_emitter = objects['scene'].param_map[f'Emitter[{idx}]']
                if emitter_type == 'env':
                    if prop == 'env_map':
                        drjit_param = drjit_emitter.radiance.data
                        drjit.enable_grad(drjit_param)
                        drjit_params.append(drjit_param)
        # t2 = time.time()
        # print(f"Time for [renderD.enable_grad]: {t2 - t1}")
        objects['scene'].configure(sensor_ids)
        # t3 = time.time()
        # print(f"Time for [renderD.configure2]: {t3 - t2}")
        param_grads = [torch.zeros_like(drjit_param.torch().cuda()) for drjit_param in drjit_params]

        for i, sensor_id in enumerate(sensor_ids):
            image_grad = Vector3fC(image_grads[i].reshape(-1, 3) / npass)
            for j in range(npass):
                image = objects['integrators'][integrator_id].renderD(objects['scene'], sensor_id)
                tmp = drjit.dot(image_grad, image)
                drjit.backward(tmp)
                for param_grad, drjit_param in zip(param_grads, drjit_params):
                    grad_tmp = drjit.grad(drjit_param)
                    param_grad += torch.nan_to_num(grad_tmp.torch().cuda())
        # t4 = time.time()
        # print(f"Time for [renderD.renderD]: {t4 - t3}")

        for i in range(len(param_names)):
            param_grads[i] = param_grads[i].reshape(scene[param_names[i]].data.shape)
        
        return param_grads
