from ivt.scene_parser import SceneParser
from ivt.io import read_image, write_image, write_obj
from ivt.transform import *
from lxml import etree
from pathlib import Path
from pyvista import read_texture
import torch
import numpy as np
import os

def str_to_tensor(s):
    assert s is not None
    return torch.from_numpy(np.array(s.split(',')).astype(np.float32))

class MitsubaParser(SceneParser):

    def read(self, scene_path, scene):
        scene_path = Path(scene_path)
        with open(scene_path, 'r') as f:
            tree = etree.parse(f)
        root = tree.getroot()

        old_path = os.getcwd()
        os.chdir(scene_path.parent)

        self.scene = scene

        self.bsdf_refs = {}

        readers = {
            'integrator': self.read_integrator,
            'sensor': self.read_sensor,
            'bsdf': self.read_bsdf,
            'shape': self.read_shape
        }

        for child in root:
            child_tag = child.tag
            
            if child_tag in readers:
                readers[child_tag](child)
            elif not isinstance(child_tag, str):
                pass
            else:
                print(f'Skipping tag <{child_tag}>.')

        os.chdir(old_path)

    def read_type(self, e):
        e_type = e.get('type')
        assert e_type is not None
        return e_type

    def read_transform(self, e):
        to_world = torch.eye(4)

        readers = {
            'lookat': self.read_lookat,
            'translate': self.read_translate,
            'rotate': self.read_rotate,
            'scale': self.read_scale,
        }
        
        for child in e:
            child_tag = child.tag

            if child_tag in readers:
                to_world = readers[child_tag](child) @ to_world
            else:
                print(f'Skipping tag <{child_tag}>.')

        return to_world

    def read_lookat(self, e):
        origin = str_to_tensor(e.get('origin'))
        target = str_to_tensor(e.get('target'))
        up = str_to_tensor(e.get('up'))
        return lookat(origin, target, up)

    def read_translate(self, e):
        x = 0.0
        y = 0.0
        z = 0.0
        xyz = None 

        if (x_v := e.get('x')) is not None: 
            x = float(x_v)
        
        if (y_v := e.get('y')) is not None: 
            y = float(y_v)

        if (z_v := e.get('z')) is not None: 
            z = float(z_v)

        if (xyz_v := e.get('value')) is not None: 
            xyz = str_to_tensor(xyz_v)

        if xyz:
            return translate(xyz)
        else:
            return translate((x, y, z))
    
    def read_rotate(self, e):
        x = 0.0
        y = 0.0
        z = 0.0
        xyz = None 
        angle = 0.0

        if (x_v := e.get('x')) is not None: 
            x = float(x_v)
        
        if (y_v := e.get('y')) is not None: 
            y = float(y_v)

        if (z_v := e.get('z')) is not None: 
            z = float(z_v)

        if (xyz_v := e.get('value')) is not None: 
            xyz = str_to_tensor(xyz_v)

        if (angle_v := e.get('angle')) is not None: 
            angle = float(angle_v)

        if xyz:
            return rotate(xyz, angle)
        else:
            return rotate((x, y, z), angle)

    def read_scale(self, e):
        x = 0.0
        y = 0.0
        z = 0.0
        xyz = None 

        if (x_v := e.get('x')) is not None: 
            x = float(x_v)
        
        if (y_v := e.get('y')) is not None: 
            y = float(y_v)

        if (z_v := e.get('z')) is not None: 
            z = float(z_v)

        if (xyz_v := e.get('value')) is not None: 
            xyz = str_to_tensor(xyz_v)

        if xyz:
            return scale(xyz)
        else:
            return scale((x, y, z))

    def read_float(self, e):
        name = e.get('name')
        value = str_to_tensor(e.get('value'))
        assert name is not None and value.numel() == 1
        return name, value.item()

    def read_int(self, e):
        name = e.get('name')
        value = str_to_tensor(e.get('value')).to(torch.int)
        assert name is not None and value.numel() == 1
        return name, value.item()

    def read_rgb(self, e):
        name = e.get('name')
        value = str_to_tensor(e.get('value'))
        assert name is not None and value.numel() == 3
        return name, value

    def read_spectrum(self, e):
        name = e.get('name')
        value = str_to_tensor(e.get('value'))
        assert name is not None
        return name, value

    def read_texture(self, e):
        texture_name = e.get('name')
        type = e.get('type')
        assert texture_name is not None and type is not None
        if type == 'bitmap':
            for child in e:
                child_tag = child.tag
                
                if child_tag == 'string':
                    name, value = self.read_str(child)
                    if name == 'filename':
                        filename = value

                    # TODO: handle other string types

                # TODO: handle other tags

            texture = read_image(filename)
        else:
            assert False, f'Unsupported tag <{child_tag}>'

        # TODO: handle other texture types

        return texture_name, texture

    def read_color(self, e):
        readers = {
            'float': self.read_float,
            'rgb': self.read_rgb,
            'texture': self.read_texture,
            'spectrum': self.read_spectrum
        }
        name, value = readers[e.tag](e)
        return name, value

    def read_str(self, e):
        name = e.get('name')
        value = e.get('value')
        assert name is not None and value is not None
        return name, value

    def read_integrator(self, e):
        integrator_type = self.read_type(e)
        self.scene.add_integrator(integrator_type)

    def read_sensor(self, e):
        sensor_type = self.read_type(e)
        if sensor_type == 'perspective':

            fov = 40
            to_world = torch.eye(4)

            for child in e:
                child_tag = child.tag

                if child_tag == 'float':
                    name, value = self.read_float(child)
                    if name == 'fov':
                        fov = value

                    # TODO: handle other float values

                elif child_tag == 'transform':
                    to_world = self.read_transform(child)

                elif child_tag == 'film':
                    self.read_film(child)

                # TODO: handle string values

            self.scene.add_perspective_camera(fov, use_to_world=True, to_world=to_world)

        else:
            assert False, f'Unsupported tag <{child_tag}>'

    def read_film(self, e):
        width = 768
        height = 576
        rfilter = 'gaussian'

        film_type = self.read_type(e)

        if film_type == 'hdrfilm':
            for child in e:
                child_tag = child.tag

                if child_tag == 'integer':
                    name, value = self.read_int(child)
                    if name == 'width':
                        width = value
                    elif name == 'height':
                        height = value
                    # TODO: handle other integer values

                elif child_tag == 'rfilter':
                    rfilter = self.read_type(child)
                
                # TODO: handle other fields

            self.scene.add_hdr_film((height, width), rfilter=rfilter)

        # TODO: handle other film type


    def read_bsdf(self, e):
        bsdf_type = self.read_type(e)

        if bsdf_type == 'diffuse':
            values = {
                'reflectance': 0.5
            }

            for child in e:
                name, value = self.read_color(child)
                values[name] = value

            self.scene.add_diffuse_bsdf(values['reflectance'])

        elif bsdf_type == 'microfacet':
            values = {
                'diffuseReflectance': 0.5,
                'specularReflectance': 0.04,
                'roughness': 0.5
            }

            for child in e:
                name, value = self.read_color(child)
                values[name] = value

            self.scene.add_microfacet_bsdf(
                values['diffuseReflectance'],
                values['specularReflectance'],
                values['roughness'])

        if (id := e.get('id')) is not None:
            self.bsdf_refs[id] = len(self.scene.bsdfs) - 1

    def read_shape(self, e):
        shape_type = self.read_type(e)

        if shape_type == 'obj':
            filename_n = e.find('string[@name]')
            name, value = self.read_str(filename_n)
            assert name == 'filename'
            obj_path = value

            ref_n = e.find('ref')
            assert ref_n is not None
            bsdf_id = self.bsdf_refs[ref_n.get('id')]

            to_world = torch.eye(4)
            if (transform_n := e.find('transform')) is not None:
                to_world = self.read_transform(transform_n)

            face_normals = False
            if (boolean_n := e.find('boolean')) is not None and boolean_n.get('name') == 'face_normals':
                value = boolean_n.get('value')
                face_normals = True if value == 'true' else False

            if (emitter_n := e.find('emitter')) is not None and emitter_n.get('type') == 'area':
                name, value = self.read_spectrum(emitter_n.find('spectrum'))
                self.scene.add_area_light(mesh_id=len(self.scene.meshes), radiance=value)

            self.scene.add_obj(obj_path, bsdf_id, to_world, face_normals)

    def write(self, scene_path, scene):
        scene_path = Path(scene_path)
        scene_dir = scene_path.parent
        data_dir = scene_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)

        def list_to_csl(data):
            if data.shape == ():
                return str(data.item())
            else:
                return ', '.join([str(v) for v in data.tolist()])

        def add_color(bsdf_id, color, name):
            if color.shape == () or color.shape == (1, ):
                etree.SubElement(bsdf_n, 'float', name=name, value=list_to_csl(color))
            elif color.shape == (3, ):
                etree.SubElement(bsdf_n, 'spectrum', name=name, value=list_to_csl(color))
            elif len(color.shape) == 3:
                texture_name = f"{bsdf_id}_{name}.exr"
                write_image(data_dir / texture_name, color)
                texture_n = etree.SubElement(bsdf_n, 'texture', name=name, type='bitmap')
                etree.SubElement(texture_n, 'string', name='filename', value=f'data/{texture_name}')
            else:
                raise ValueError(f'color shape of [{bsdf_id}.{name}] is not supported: {color.shape}')

        scene_n = etree.Element("scene", version='2.0.0')

        for i, integrator in enumerate(scene.integrators):
            etree.SubElement(scene_n, "integrator", type=integrator['type'])

        for i, sensor in enumerate(scene.sensors): 
            sensor_n = etree.SubElement(scene_n, 'sensor', type=sensor['type'])

            transform_n = etree.SubElement(sensor_n, 'transform', name='toWorld')
            if 'to_world' in sensor:
                etree.SubElement(transform_n, 'matrix', value=list_to_csl(sensor['to_world'].data.flatten()))
            else:
                etree.SubElement(transform_n, 'lookat', 
                    target=list_to_csl(sensor['target'].data),
                    origin=list_to_csl(sensor['origin'].data),
                    up=list_to_csl(sensor['up'].data),
                )

            etree.SubElement(sensor_n, 'float', name='fov', value=str(sensor['fov'].data.item()))

            if i == 0:
                sampler_n = etree.SubElement(sensor_n, 'sampler', type='independent')
                etree.SubElement(sampler_n, 'integer', name='sample_count', value='1')
                film = scene.film
                film_n = etree.SubElement(sensor_n, 'film', type=film['type'])
                etree.SubElement(film_n, 'integer', name='width', value=str(film['resolution'][1]))
                etree.SubElement(film_n, 'integer', name='height', value=str(film['resolution'][0]))
                etree.SubElement(film_n, 'rfilter', type=film['rfilter'])

        area_lights = {}
        for i, emitter in enumerate(scene.emitters):
            emitter_type = emitter['type']
            if emitter_type == 'area': 
                area_lights[emitter['mesh_id']] = i
            elif emitter_type == 'env':
                env_map_name = f"{emitter['id']}_envmap.exr"
                write_image(data_dir / env_map_name, emitter['env_map'].data)
                emitter_n = etree.SubElement(scene_n, 'emitter', type='envmap')
                etree.SubElement(emitter_n, 'string', name='filename', value=f"data/{env_map_name}")
                transform_n = etree.SubElement(emitter_n, 'transform', name='to_world')
                etree.SubElement(transform_n, 'matrix', value=list_to_csl(emitter['to_world'].data.flatten()))

        for bsdf in scene.bsdfs:
            bsdf_type = bsdf['type']
            bsdf_id = bsdf['id']
            bsdf_n = etree.SubElement(scene_n, 'bsdf', type=bsdf_type, id=bsdf_id)
            transform_n = etree.SubElement(bsdf_n, 'transform', name='to_world')
            etree.SubElement(transform_n, 'matrix', value=list_to_csl(bsdf['to_world'].data.flatten()))

            if bsdf_type == 'diffuse':
                add_color(bsdf_id, bsdf['reflectance'].data, 'reflectance')
            elif bsdf_type == 'microfacet':
                add_color(bsdf_id, bsdf['diffuse_reflectance'].data, 'diffuseReflectance')
                add_color(bsdf_id, bsdf['specular_reflectance'].data, 'specularReflectance')
                add_color(bsdf_id, bsdf['roughness'].data, 'roughness')
            else:
                raise ValueError(f"BSDF type {bsdf['type']} is not supported.")

        for i, mesh in enumerate(scene.meshes):
            obj_name = f"{mesh['id']}.obj"
            write_obj(data_dir / obj_name, 
                    mesh['vertex_positions'].data, 
                    mesh['vertex_indices'].data,
                    mesh['uv_positions'].data,
                    mesh['uv_indices'].data)

            shape_n = etree.SubElement(scene_n, 'shape', type='obj')
            etree.SubElement(shape_n, 'string', name='filename', value=f'data/{obj_name}')

            if i in area_lights:
                emitter = scene.emitters[area_lights[i]]
                emitter_n = etree.SubElement(shape_n, 'emitter', type='area')
                etree.SubElement(emitter_n, 'spectrum', name='radiance', value=list_to_csl(emitter['radiance'].data.flatten()))

            etree.SubElement(shape_n, 'ref', id=f"bsdfs[{mesh['bsdf_id']}]")

            transform_n = etree.SubElement(shape_n, 'transform', name='to_world')
            etree.SubElement(transform_n, 'matrix', value=list_to_csl(mesh['to_world'].data.flatten()))
            use_face_normal = 'true' if mesh['use_face_normal'] else 'false'
            etree.SubElement(shape_n, 'boolean', name='face_normals', value=use_face_normal)
        

        with open(scene_path, 'wb') as f:
            f.write(etree.tostring(scene_n, pretty_print=True, xml_declaration=True, encoding='UTF-8'))