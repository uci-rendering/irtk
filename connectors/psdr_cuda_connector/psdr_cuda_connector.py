from multiprocessing.sharedctypes import Value
from ivt.connector import Connector
from ivt.scene_parser import SceneParserManager
from ivt.scene import split_param_name
from ivt.transform import lookat

import psdr_cuda
import enoki
from enoki.cuda import Vector3f as Vector3fC
from enoki.cuda_autodiff import Vector3f as Vector3fD, Float32 as FloatD, Matrix4f as Matrix4fD, Matrix3f as Matrix3fD
from enoki.cuda_autodiff import Float32 as FloatD
import torch

from pathlib import Path
from shutil import rmtree
import os

class PSDRCudaConnector(Connector):
    backend = 'torch'
    device = 'cuda'
    ftype = torch.float32
    itype = torch.long

    def __init__(self):
        super().__init__()

    def create_scene(self, scene):
        # Create a temporary scene directory with all the data
        tmp_path = Path("__psdr_cuda_tmp__")
        tmp_scene_path = tmp_path / 'scene.xml'
        spm = SceneParserManager()
        sp = spm.get_scene_parser('mitsuba')
        sp.write(tmp_scene_path, scene)
        
        # Load the scene 
        old_path = os.getcwd()
        os.chdir(tmp_path)
        psdr_scene = psdr_cuda.Scene()
        psdr_scene.load_file('scene.xml', False)
        os.chdir(old_path)
        
        # Clean up
        rmtree(tmp_path)

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
                objects['integrators'].append(psdr_cuda.DirectIntegrator())
                if 'hide_envmap' in integrator_config['params']:
                    objects['integrators'][it].hide_emitters = integrator_config['params']['hide_envmap']
            elif integrator_config['type'] == 'collocated':
                objects['integrators'].append(psdr_cuda.CollocatedIntegrator(integrator_config['params']['intensity']))
            elif integrator_config['type'] == 'field':
                objects['integrators'].append(psdr_cuda.FieldExtractionIntegrator(integrator_config['params']['name']))
            elif integrator_config['type'] == 'path':
                objects['integrators'].append(psdr_cuda.PathTracer(integrator_config['params']['max_depth']))
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
        if 'psdr_cuda' in scene.cached:
            objects = scene.cached['psdr_cuda']
            psdr_param_map = objects['scene'].param_map
            updated_params = scene.get_updated()

            for param_name in updated_params:
                param = scene.param_map[param_name]
                group, idx, prop = split_param_name(param_name)

                if group == 'meshes':
                    enoki_mesh = psdr_param_map[f'Mesh[{idx}]']
                    if prop == 'vertex_positions':
                        enoki_param = Vector3fD(param.data)
                        enoki_mesh.vertex_positions = enoki_param
                    elif prop == 'to_world':
                        enoki_param = Matrix4fD(param.data.reshape(1, 4, 4))
                        enoki_mesh.to_world = enoki_param

                elif group == 'bsdfs': 
                    bsdf_type = scene.bsdfs[idx]['type']
                    enoki_bsdf = psdr_param_map[f'BSDF[{idx}]']
                    
                    if bsdf_type == 'diffuse':
                        if prop == 'reflectance':
                            enoki_param = Vector3fD(convert_color(param.data))
                            enoki_bsdf.reflectance.data = enoki_param
                        elif prop == 'to_world':
                            enoki_param = Matrix3fD(param.data.reshape(1, 3, 3))
                            enoki_bsdf.reflectance.to_world = enoki_param

                    elif bsdf_type == 'microfacet':
                        if prop == 'diffuse_reflectance':
                            enoki_param = Vector3fD(convert_color(param.data))
                            enoki_bsdf.diffuseReflectance.data = enoki_param
                        elif prop == 'specular_reflectance':
                            enoki_param = Vector3fD(convert_color(param.data))
                            enoki_bsdf.specularReflectance.data = enoki_param
                        elif prop == 'roughness':
                            enoki_param = FloatD(convert_color(param.data, c=None))
                            enoki_bsdf.roughness.data = enoki_param

                elif group == 'sensors':
                    sensor = scene.sensors[idx]
                    sensor_type = sensor['type']
                    enoki_sensor = psdr_param_map[f'Sensor[{idx}]']

                    if sensor_type == 'perspective':
                        if prop == 'origin':
                            to_world = lookat(sensor['origin'].data, sensor['target'].data, sensor['up'].data).to('cuda').to(torch.float32)
                            enoki_sensor.to_world = Matrix4fD(to_world.reshape(1, 4, 4))
                
                elif group == 'emitters':
                    emitter = scene.emitters[idx]
                    emitter_type = emitter['type']
                    enoki_emitter = psdr_param_map[f'Emitter[{idx}]']

                    if emitter_type == 'env':
                        if prop == 'to_world':
                            enoki_param = Matrix4fD(param.data.reshape(1, 4, 4))
                            enoki_emitter.to_world = enoki_param
                        elif prop == 'env_map':
                            enoki_param = Vector3fD(convert_color(param.data))
                            enoki_emitter.radiance.data = enoki_param
                            
                param.updated = False

            return objects

        # Create scene objects if there is no cache
        else:
            objects = self.create_objects(scene, render_options)
            scene.cached['psdr_cuda'] = objects
            for param_name in scene.get_updated():
                scene.param_map[param_name].updated = False
            return objects

    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        # Convert the scene into psdr_cuda objects
        objects = self.get_objects(scene, render_options)
        objects['scene'].opts.spp = render_options['spp_c'] if 'spp_c' in render_options else render_options['spp']
        objects['scene'].configure2(sensor_ids)
        h, w, c = objects['film']['shape']
        npass = render_options['npass']

        # Render the images
        images = []
        for sensor_id in sensor_ids:
            image = torch.zeros((h * w, c)).to(PSDRCudaConnector.device).to(PSDRCudaConnector.ftype)
            for i in range(npass):
                image += objects['integrators'][integrator_id].renderC(objects['scene'], sensor_id).torch() / npass
            image = image.reshape(h, w, c)
            images.append(image)
        return images
    
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        # Convert the scene into psdr_cuda objects
        objects = self.get_objects(scene, render_options)
        objects['scene'].opts.spp = render_options['spp']
        objects['scene'].configure2(sensor_ids)
        psdr_param_map = objects['scene'].param_map
        npass = render_options['npass']

        param_names = scene.get_requiring_grad()
        enoki_params = []
        for param_name in param_names:
            param = scene.param_map[param_name]
            group, idx, prop = split_param_name(param_name)
            if group == 'meshes':
                enoki_mesh = psdr_param_map[f'Mesh[{idx}]']
                if prop == 'vertex_positions':
                    enoki_param = enoki_mesh.vertex_positions
                    enoki.set_requires_gradient(enoki_param, True)
                    enoki_params.append(enoki_param)
                elif prop == 'to_world':
                    enoki_param = enoki_mesh.to_world
                    enoki.set_requires_gradient(enoki_param, True)
                    enoki_params.append(enoki_param)

            elif group == 'bsdfs':
                bsdf_type = scene.bsdfs[idx]['type']
                enoki_bsdf = objects['scene'].param_map[f'BSDF[{idx}]']
                if bsdf_type == 'diffuse':
                    if prop == 'reflectance':
                        enoki_param = enoki_bsdf.reflectance.data 
                        enoki.set_requires_gradient(enoki_param, True)
                        enoki_params.append(enoki_param)
                    elif prop == 'to_world':
                        enoki_param = enoki_bsdf.reflectance.to_world 
                        enoki.set_requires_gradient(enoki_param, True)
                        enoki_params.append(enoki_param)
                    else:
                        raise ValueError(f"property not supported: {prop}")
                elif bsdf_type == 'microfacet':
                    if prop == 'diffuse_reflectance':
                        enoki_param = enoki_bsdf.diffuseReflectance.data
                        enoki.set_requires_gradient(enoki_param, True)
                        enoki_params.append(enoki_param)
                    elif prop == 'specular_reflectance':
                        enoki_param = enoki_bsdf.specularReflectance.data
                        enoki.set_requires_gradient(enoki_param, True)
                        enoki_params.append(enoki_param)
                    elif prop == 'roughness':
                        enoki_param = enoki_bsdf.roughness.data
                        enoki.set_requires_gradient(enoki_param, True)
                        enoki_params.append(enoki_param)
                    else:
                        raise ValueError(f"property not supported: {prop}")
            elif group == 'emitters':
                emitter_type = scene.emitters[idx]['type']
                enoki_emitter = objects['scene'].param_map[f'Emitter[{idx}]']
                if emitter_type == 'env':
                    if prop == 'env_map':
                        enoki_param = enoki_emitter.radiance.data
                        enoki.set_requires_gradient(enoki_param, True)
                        enoki_params.append(enoki_param)
        objects['scene'].configure2(sensor_ids)

        param_grads = [torch.zeros_like(enoki_param.torch().cuda()) for enoki_param in enoki_params]

        for i, sensor_id in enumerate(sensor_ids):
            image_grad = Vector3fC(image_grads[i].reshape(-1, 3) / npass) 
            for j in range(npass):
                image = objects['integrators'][integrator_id].renderD(objects['scene'], sensor_id)
                enoki.set_gradient(image, image_grad)
                FloatD.backward()

                for param_grad, enoki_param in zip(param_grads, enoki_params):
                    param_grad += torch.nan_to_num(enoki.gradient(enoki_param).torch().cuda())

            # garbage collection
            del image, image_grad
            enoki.cuda_malloc_trim()

        for i in range(len(param_names)):
            param_grads[i] = param_grads[i].reshape(scene.param_map[param_names[i]].data.shape)
        
        return param_grads
