from ivt.connector import Connector
from ivt.scene_parser import SceneParserManager
import psdr_cuda
import enoki
from enoki.cuda import Vector3f as Vector3fC
from enoki.cuda_autodiff import Vector3f as Vector3fD
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
        psdr_scene.load_file(str('scene.xml'), False)
        os.chdir(old_path)
        
        # Clean up
        rmtree(tmp_path)

        return psdr_scene

    def create_objects(self, scene):
        objects = {}
        
        objects['scene'] = self.create_scene(scene)

        integrator_config = scene.integrator
        if integrator_config['type'] == 'direct':
            objects['integrator'] = psdr_cuda.DirectIntegrator()
        elif integrator_config['type'] == 'collocated':
            objects['integrator'] = psdr_cuda.CollocatedIntegrator(integrator_config['params']['intensity'])
        else:
            raise ValueError(f"integrator type [{integrator_config['type']}] is not supported.")
            
        objects['scene'].opts.spp = scene.render_options['spp']
        objects['scene'].opts.sppe = scene.render_options['sppe']
        objects['scene'].opts.sppse = scene.render_options['sppse']
        objects['scene'].opts.log_level = scene.render_options['log_level']
        
        film_config = scene.film
        width, height = film_config['resolution']
        objects['film'] = {
            'shape': (width, height, 3)
        }

        return objects

    def renderC(self, scene, sensor_ids=[0]):
        # Convert the scene into psdr_cuda objects
        objects = self.create_objects(scene)
        objects['scene'].configure2(sensor_ids)
        w, h, c = objects['film']['shape']
        npass = scene.render_options['npass']

        # Render the images
        images = []
        for sensor_id in sensor_ids:
            image = torch.zeros((w * h, c)).to(PSDRCudaConnector.device).to(PSDRCudaConnector.ftype)
            for i in range(npass):
                image += objects['integrator'].renderC(objects['scene'], sensor_id).torch() / npass
            image = image.reshape(w, h, c)
            images.append(image)

        # garbage collection
        del objects
        enoki.cuda_malloc_trim()
        
        return images
    
    def renderD(self, image_grads, scene, sensor_ids=[0]):
        # Convert the scene into psdr_cuda objects
        objects = self.create_objects(scene)
        objects['scene'].configure2(sensor_ids)
        npass = scene.render_options['npass']

        def split_parma_name(param_name):
            group, idx, prop = param_name.replace('[', '.').replace(']', '').split('.')
            idx = int(idx)
            return group, idx, prop

        param_names = scene.get_requiring_grad()
        enoki_params = []
        for param_name in param_names:
            group, idx, prop = split_parma_name(param_name)
            if group == 'meshes':
                if prop == 'vertex_positions':
                    enoki_param = Vector3fD(scene.param_map[param_name].data)
                    enoki.set_requires_gradient(enoki_param, True)
                    objects['scene'].param_map[f'Mesh[{idx}]'].vertex_positions = enoki_param
                    enoki_params.append(enoki_param)

            elif group == 'bsdfs':
                brdf_type = scene.bsdfs[idx]['type']
                if brdf_type == 'diffuse':
                    if prop == 'reflectance':
                        enoki_param = Vector3fD(scene.param_map[param_name].data.reshape(-1, 3))
                        enoki.set_requires_gradient(enoki_param, True)
                        objects['scene'].param_map[f'BSDF[{idx}]'].reflectance.data = enoki_param
                        enoki_params.append(enoki_param)

        objects['scene'].configure2(sensor_ids)

        for i, sensor_id in enumerate(sensor_ids):
            image_grad = Vector3fC(image_grads[i].reshape(-1, 3))
            for j in range(npass):
                if j == 0:
                    image = objects['integrator'].renderD(objects['scene'], sensor_id) / npass
                else:
                    image += objects['integrator'].renderD(objects['scene'], sensor_id) / npass
            
            enoki.set_gradient(image, image_grad)
            FloatD.backward()

        param_grads = [enoki.gradient(enoki_param).torch().cuda() for enoki_param in enoki_params]
        for i in range(len(param_names)):
            param_grads[i] = param_grads[i].reshape(scene.param_map[param_names[i]].data.shape)

        # garbage collection
        del objects
        enoki.cuda_malloc_trim()
        
        return param_grads
