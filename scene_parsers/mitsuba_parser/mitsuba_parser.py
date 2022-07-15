from cgitb import text
from multiprocessing.sharedctypes import Value
from torch import TracingState
from ivt.scene_parser import SceneParser
from ivt.io import write_exr, write_obj
from lxml import etree
from pathlib import Path

class MitsubaParser(SceneParser):

    def read(self, scene_path):
        pass
    
    def write(self, scene_path, scene):
        scene_path = Path(scene_path)
        scene_dir = scene_path.parent
        model_dir = scene_dir / 'model'
        model_dir.mkdir(parents=True, exist_ok=True)

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
                write_exr(model_dir / texture_name, color)
                texture_n = etree.SubElement(bsdf_n, 'texture', name=name, type='bitmap')
                filename_n = etree.SubElement(texture_n, 'string', name='filename', value=f'model/{texture_name}')
            else:
                raise ValueError(f'color shape of [{bsdf_id}.{name}] is not supported: {color.shape}')

        scene_n = etree.Element("scene", version='2.0.0')

        integrator_n = etree.SubElement(scene_n, "integrator", type=scene.integrator['type'])

        for i, sensor in enumerate(scene.sensors): 
            sensor_n = etree.SubElement(scene_n, 'sensor', type=sensor['type'])

            transform_n = etree.SubElement(sensor_n, 'transform', name='toWorld')
            lookat_n = etree.SubElement(transform_n, 'lookat', 
                    target=list_to_csl(sensor['target'].data),
                    origin=list_to_csl(sensor['origin'].data),
                    up=list_to_csl(sensor['up'].data),
            )

            fov_n = etree.SubElement(sensor_n, 'float', name='fov', value=str(sensor['fov'].item()))

            if i == 0:
                sampler_n = etree.SubElement(sensor_n, 'sampler', type='independent')
                sample_count_n = etree.SubElement(sampler_n, 'integer', name='sample_count', value='1')
                film = scene.film
                film_n = etree.SubElement(sensor_n, 'film', type=film['type'])
                width_n = etree.SubElement(film_n, 'integer', name='width', value=str(film['resolution'][0]))
                height_n = etree.SubElement(film_n, 'integer', name='height', value=str(film['resolution'][1]))
                rfilter_n = etree.SubElement(film_n, 'rfilter', type=film['rfilter'])

        area_lights = {}
        for i, emitter in enumerate(scene.emitters):
            if emitter['type'] == 'area': 
                area_lights[emitter['mesh_id']] = i

        for bsdf in scene.bsdfs:
            bsdf_type = bsdf['type']
            bsdf_id = bsdf['id']
            bsdf_n = etree.SubElement(scene_n, 'bsdf', type=bsdf_type, id=bsdf_id)

            if bsdf['type'] == 'diffuse':
                add_color(bsdf_id, bsdf['reflectance'].data, 'reflectance')
            elif bsdf['type'] == 'microfacet':
                add_color(bsdf_id, bsdf['diffuse_reflectance'].data, 'diffuseReflectance')
                add_color(bsdf_id, bsdf['specular_reflectance'].data, 'specularReflectance')
                add_color(bsdf_id, bsdf['roughness'].data, 'roughness')
            else:
                raise ValueError(f"BSDF type {bsdf['type']} is not supported.")

        for i, mesh in enumerate(scene.meshes):
            obj_name = f"{mesh['id']}.obj"
            write_obj(model_dir / obj_name, 
                    mesh['vertex_positions'].data, 
                    mesh['vertex_indices'].data,
                    mesh['uv_positions'].data,
                    mesh['uv_indices'].data)

            shape_n = etree.SubElement(scene_n, 'shape', type='obj')
            filename_n = etree.SubElement(shape_n, 'string', name='filename', value=f'model/{obj_name}')

            if i in area_lights:
                emitter = scene.emitters[area_lights[i]]
                emitter_n = etree.SubElement(shape_n, 'emitter', type='area')
                radiance_n = etree.SubElement(emitter_n, 'spectrum', name='radiance', value=list_to_csl(emitter['radiance'].data.flatten()))

            ref_n = etree.SubElement(shape_n, 'ref', id=f"bsdfs[{mesh['bsdf_id']}]")

        

        with open(scene_path, 'wb') as f:
            f.write(etree.tostring(scene_n, pretty_print=True, xml_declaration=True, encoding='UTF-8'))