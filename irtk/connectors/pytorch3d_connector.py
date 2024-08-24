from ..connector import Connector
from ..scene import *
from ..config import *
from ..utils import Timer

from pytorch3d.structures import Meshes, join_meshes_as_scene
import pytorch3d.renderer as pr
import torch
import time

class PyTorch3DConnector(Connector, connector_name='pytorch3d'):

    def __init__(self):
        super().__init__()

    def update_scene_objects(self, scene, render_options, sensor_ids):
        if 'pytorch3d' in scene.cached:
            cache = scene.cached['pytorch3d']
        else:
            cache = {}
            scene.cached['pytorch3d'] = cache

            cache['meshes'] = []
            cache['textures'] = dict()
            cache['materials'] = dict()
            cache['cameras'] = []
            cache['point_light'] = None
            cache['film'] = None

            cache['name_map'] = {}

        pytorch3d_params = []
        for name in scene.components:
            component = scene[name]
            if type(component) in self.extensions:
                pytorch3d_params += self.extensions[type(component)](name, scene)
            else:
                raise RuntimeError(f'Unsupported component for PyTorch3D: {component}')
        
        # construct some scene components
        cache['texture'] = pr.TexturesUV(
            [cache['textures'][mesh['mat_id']] for mesh in cache['meshes']],
            [mesh['fuv'] for mesh in cache['meshes']],
            [mesh['uv'] for mesh in cache['meshes']]
        )
        cache['material'] = pr.Materials(
            [cache['materials'][mesh['mat_id']]['ambient'] for mesh in cache['meshes']],
            [cache['materials'][mesh['mat_id']]['diffuse'] for mesh in cache['meshes']],
            [cache['materials'][mesh['mat_id']]['specular'] for mesh in cache['meshes']],
            [cache['materials'][mesh['mat_id']]['shininess'] for mesh in cache['meshes']],
            device=configs['device']
        )
        # construct mesh here in case the textures have changed
        cache['mesh'] = Meshes(
            verts=[mesh['verts'] for mesh in cache['meshes']],
            faces=[mesh['faces'] for mesh in cache['meshes']],
            textures=cache['texture'],
        )
        
        # extend mesh to all views(batches)
        if len(cache['meshes']) == 1:
            cache['mesh'] = cache['mesh'].extend(len(sensor_ids))
        else:
            cache['mesh'] = join_meshes_as_scene(cache['mesh']).extend(len(sensor_ids))
        
        selected_cameras = [cache['cameras'][i] for i in sensor_ids]
        cache['camera'] = pr.FoVPerspectiveCameras(
            znear=[cam['znear'] for cam in selected_cameras], 
            zfar=[cam['zfar'] for cam in selected_cameras], 
            fov=[cam['fov'] for cam in selected_cameras], 
            R=torch.stack([cam['R'] for cam in selected_cameras]), 
            T=torch.stack([cam['T'] for cam in selected_cameras]), 
            device=configs['device']
        )
        # change light here
        # lights = pr.AmbientLights(ambient_color=((1, 1, 1), ), device=configs['device'])
        if cache['point_light']:
            lights = pr.PointLights(
                location=[list(cache['point_light']['position']) for i in range(len(selected_cameras))], 
                ambient_color=((0, 0, 0), ),
                diffuse_color=(tuple(cache['point_light']['radiance']), ),
                specular_color=((1, 1, 1), ),
                device=configs['device']
            )
        else: 
            diffuse_color = (1, 1, 1)
            specular_color = (0, 0, 0)
            if 'light_diffuse_color' in render_options:
                diffuse_color = render_options['light_diffuse_color']
            if 'light_power' in render_options:
                diffuse_color = (render_options['light_power'], render_options['light_power'], render_options['light_power'])
                specular_color = diffuse_color
            lights = pr.PointLights(
                location=cache['camera'].get_camera_center(), 
                # location=((-1.5, 1.5, 1.5), ),
                ambient_color=((0, 0, 0), ),
                diffuse_color=(diffuse_color, ),
                specular_color=(specular_color, ),
                device=configs['device']
            )
        cache['light'] = lights
        # cache['mesh'] = join_meshes_as_batch(cache['meshes'])
        return scene.cached['pytorch3d'], pytorch3d_params

    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        cache, _ = self.update_scene_objects(scene, render_options, sensor_ids)

        blend_params = pr.BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
        raster_settings = pr.RasterizationSettings(
            image_size=cache['film'], 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        renderer = pr.MeshRenderer(
            rasterizer=pr.MeshRasterizer(
                cameras=cache['camera'], 
                raster_settings=raster_settings
            ),
            shader=pr.SoftPhongShader(
                device=configs['device'], 
                cameras=cache['camera'],
                lights=cache['light'],
                materials=cache['material'],
                blend_params=blend_params
            )
        )
        
        images = None
        npass = render_options['npass']
        with Timer('Forward'):
            for i in range(npass):
                image_pass = renderer(cache['mesh'])[..., :3]
                if images:
                    images += image_pass / npass
                else:
                    images = image_pass / npass

        return list(images)
        
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        with torch.enable_grad():
            cache, pytorch3d_params = self.update_scene_objects(scene, render_options, sensor_ids)

            npass = render_options['npass']
            
            param_grads = [torch.zeros_like(scene[param_name]) for param_name in scene.requiring_grad]
            
            blend_params = pr.BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
            raster_settings = pr.RasterizationSettings(
                image_size=cache['film'], 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )
            renderer = pr.MeshRenderer(
                rasterizer=pr.MeshRasterizer(
                    cameras=cache['camera'], 
                    raster_settings=raster_settings
                ),
                shader=pr.SoftPhongShader(
                    device=configs['device'], 
                    cameras=cache['camera'],
                    lights=cache['light'],
                    materials=cache['material'],
                    blend_params=blend_params
                )
            )
            image_grad = torch.stack(image_grads) / npass
            with Timer('Backward'):
                for j in range(npass):
                    image = renderer(cache['mesh'])[..., :3]
                    tmp = (image_grad[..., :3] * image).sum(dim=3)
                    pytorch3d_grads = torch.autograd.grad(tmp, pytorch3d_params, torch.ones_like(tmp), retain_graph=True)
                    for param_grad, pytorch3d_grad in zip(param_grads, pytorch3d_grads):
                        param_grad += pytorch3d_grad
                    
            return param_grads
    
    def renderGrad(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        def render(x):
            cache, _ = self.update_scene_objects(scene, render_options, sensor_ids)
            # change mesh position offset
            offset = torch.zeros(3)
            offset[0] = x[0]
            offset = offset.to(configs['device'])
            cache['mesh'].offset_verts_(offset)

            blend_params = pr.BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
            raster_settings = pr.RasterizationSettings(
                image_size=cache['film'], 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )
            renderer = pr.MeshRenderer(
                rasterizer=pr.MeshRasterizer(
                    cameras=cache['camera'], 
                    raster_settings=raster_settings
                ),
                shader=pr.SoftPhongShader(
                    device=configs['device'], 
                    cameras=cache['camera'],
                    lights=cache['light'],
                    blend_params=blend_params
                )
            )
            
            images = None
            npass = render_options['npass']
            for i in range(npass):
                image_pass = renderer(cache['mesh'])[..., :3]
                if images:
                    images += image_pass / npass
                else:
                    images = image_pass / npass
                
            return images
        
        # mesh position.x offset
        x = torch.tensor([0.0], requires_grad=True)

        images = render(x)
        grads = torch.zeros_like(images).reshape(images.shape[0], -1)
        for i, image in enumerate(list(images)):
            for j in range(image.nelement()):
                grad = torch.autograd.grad(image.reshape(-1)[j], x, retain_graph=True)
                grads[i, j] = grad[0]
                
                if j % 100 == 0 or j == image.nelement() - 1:
                    print(f'\rImage {i}: {j}/{image.nelement() - 1}', end='')

        grads = grads.reshape(images.shape)
        # grads = torch.autograd.functional.jacobian(render, x, vectorize=True)
        # grads.squeeze_(4)
        
        # cm = plt.get_cmap('gist_rainbow')
        # colored_grads = cm(grads.sum(-1).sigmoid().cpu().numpy())
        # print(to_torch_f(colored_grads).shape)
        
        return list(grads)


"""
@PSDRJITConnector.register(Integrator)
def process_integrator(name, scene):
    integrator = scene[name]
    cache = scene.cached['psdr_jit']

    if integrator['type'] == 'field':
        psdr_integrator = psdr_jit.FieldExtractionIntegrator(integrator['config']['type'])
        cache['integrators'][name] = psdr_integrator
    elif integrator['type'] == 'collocated':
        psdr_integrator = psdr_jit.CollocatedIntegrator(integrator['config']['intensity'])
        cache['integrators'][name] = psdr_integrator
    elif integrator['type'] == 'path':
        psdr_integrator = psdr_jit.PathTracer(integrator['config']['max_depth'])
        cache['integrators'][name] = psdr_integrator
        if 'hide_emitters' in integrator['config']:
            psdr_integrator.hide_emitters = integrator['config']['hide_emitters']
    else:
        raise RuntimeError(f"unrecognized integrator type: {integrator['type']}")

    return []
"""

@PyTorch3DConnector.register(HDRFilm)
def process_hdr_film(name, scene):
    film = scene[name]
    cache = scene.cached['pytorch3d']

    cache['film'] = (film['height'], film['width'])

    return []

@PyTorch3DConnector.register(PerspectiveCamera)
def process_perspective_camera(name, scene):
    sensor = scene[name]
    cache = scene.cached['pytorch3d']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        pytorch_sensor = {
            'znear': sensor['near'],
            'zfar': sensor['far'],
            'fov': sensor['fov'],
            'to_world': sensor['to_world']
        }
        R = pytorch_sensor['to_world'][:3, :3]
        T = -torch.mm(pytorch_sensor['to_world'][:3, 3][None], R).squeeze()
        
        pytorch_sensor['R'] = R
        pytorch_sensor['T'] = T
        
        cache['cameras'].append(pytorch_sensor)
        cache['name_map'][name] = pytorch_sensor

    pytorch_sensor = cache['name_map'][name]
    
    # Update parameters
    updated = sensor.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == "to_world":
                pytorch_sensor['to_world'] = sensor['to_world']
                R = pytorch_sensor['to_world'][:3, :3]
                T = -torch.mm(pytorch_sensor['to_world'][:3, 3][None], R).squeeze()
                pytorch_sensor['R'] = R
                pytorch_sensor['T'] = T
            sensor.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    torch_params = []
    requiring_grad = sensor.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == "to_world":
                pytorch_sensor['to_world'].requires_grad_()
                # ensure 'to_world' is linked with R,T
                R = pytorch_sensor['to_world'][:3, :3]
                T = -torch.mm(pytorch_sensor['to_world'][:3, 3][None], R).squeeze()
                pytorch_sensor['R'] = R
                pytorch_sensor['T'] = T
                torch_params.append(pytorch_sensor['to_world'])
                
    return torch_params

@PyTorch3DConnector.register(Mesh)
def process_mesh(name, scene):
    mesh = scene[name]
    cache = scene.cached['pytorch3d']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        # Create its material first
        mat_id = mesh['mat_id']
        if mat_id not in scene:
            raise RuntimeError(f"The material of the mesh {name} doesn't exist: mat_id={mat_id}")
        
        verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(configs['device'])), dim=1)
        verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
        if mesh['uv'].nelement() == 0:
            mesh['uv'] = torch.zeros((1, 2)).to(configs['device'])
        if mesh['fuv'].nelement() == 0:
            mesh['fuv'] = torch.zeros_like(mesh['f']).to(configs['device'])

        pytorch3d_mesh = {
            'verts': verts,
            'faces': mesh['f'],
            'uv': mesh['uv'][..., :2],
            'fuv': mesh['fuv'].long(),
            'mat_id': mesh['mat_id']
        }
        
        cache['meshes'].append(pytorch3d_mesh)
        cache['name_map'][name] = pytorch3d_mesh

    pytorch3d_mesh = cache['name_map'][name]
    
    # Update parameters
    updated = mesh.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'v' or param_name == 'to_world':
                verts = torch.cat((mesh['v'], torch.ones((mesh['v'].shape[0], 1)).to(configs['device'])), dim=1)
                verts = torch.matmul(verts, mesh['to_world'].transpose(0, 1))[..., :3]
                if mesh['uv'].nelement() == 0:
                    mesh['uv'] = torch.zeros((1, 2)).to(configs['device'])
                if mesh['fuv'].nelement() == 0:
                    mesh['fuv'] = torch.zeros_like(mesh['f']).to(configs['device'])
                    
                if mesh['can_change_topology']:
                    pytorch3d_mesh['verts'] = verts
                    pytorch3d_mesh['faces'] = mesh['f']
                    pytorch3d_mesh['uv'] = mesh['uv'][..., :2]
                    pytorch3d_mesh['fuv'] = mesh['fuv'].long()
                    pytorch3d_mesh['mat_id'] = mesh['mat_id']
                else:
                    pytorch3d_mesh['verts'] = verts
                    pytorch3d_mesh['faces'] = mesh['f']

            mesh.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    pytorch3d_params = []
    requiring_grad = mesh.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'v':
                pytorch3d_mesh['verts'].requires_grad = True
                pytorch3d_params.append(pytorch3d_mesh['verts'])

    return pytorch3d_params

"""
def convert_color(color, c, bitmap=True):
    if color.shape == ():
        color = color.tile(c)
    if color.shape == (c,):
        color = color.reshape(1, 1, c)
    h, w, _ = color.shape
    if c == 3:
        color = Vector3fD(color.reshape(-1, c))
        if bitmap:
            color = psdr_jit.Bitmap3fD(w, h, color)
    elif c == 1:
        color = FloatD(color.reshape(-1))
        if bitmap:
            color = psdr_jit.Bitmap1fD(w, h, color)
    else:
        raise RuntimeError("Not support bitmap channel number: {c}")
    return color
"""

@PyTorch3DConnector.register(DiffuseBRDF)
def process_diffuse_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['pytorch3d']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        if brdf['d'].dim() == 1:
            pytorch3d_brdf = brdf['d'].reshape(1, 1, 3)
        else: 
            pytorch3d_brdf = brdf['d']
        
        cache['textures'][name] = pytorch3d_brdf
        cache['materials'][name] = {
            'ambient': (0, 0, 0),
            'diffuse': (1, 1, 1),
            'specular': (0, 0, 0),
            'shininess': 0
        }
        cache['name_map'][name] = pytorch3d_brdf

    pytorch3d_brdf = cache['name_map'][name]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                if brdf['d'].dim() == 1:
                    pytorch3d_brdf = brdf['d'].reshape(1, 1, 3)
                else: 
                    pytorch3d_brdf = brdf['d']
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    pytorch3d_params = []
    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'd':
                pytorch3d_brdf.requires_grad = True
                pytorch3d_params.append(pytorch3d_brdf)

    return pytorch3d_params

@PyTorch3DConnector.register(MicrofacetBRDF)
def process_microfacet_brdf(name, scene):
    brdf = scene[name]
    cache = scene.cached['pytorch3d']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        if brdf['d'].dim() == 1:
            pytorch3d_brdf = brdf['d'].reshape(1, 1, 3)
        else: 
            pytorch3d_brdf = brdf['d']
        
        if brdf['s'].shape == ():
            brdf['s'] = brdf['s'].repeat(3)
        
        cache['textures'][name] = pytorch3d_brdf
        cache['materials'][name] = {
            'ambient': (0, 0, 0),
            'diffuse': (1, 1, 1),
            'specular': tuple(brdf['s']),
            'shininess': brdf['r']
        }
        cache['name_map'][name] = pytorch3d_brdf

    pytorch3d_brdf = cache['name_map'][name]

    # Update parameters
    updated = brdf.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'd':
                if brdf['d'].dim() == 1:
                    pytorch3d_brdf = brdf['d'].reshape(1, 1, 3)
                else: 
                    pytorch3d_brdf = brdf['d']
            brdf.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    pytorch3d_params = []
    requiring_grad = brdf.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'd':
                pytorch3d_brdf.requires_grad = True
                pytorch3d_params.append(pytorch3d_brdf)

    return pytorch3d_params

"""
@PSDRJITConnector.register(EnvironmentLight)
def process_environment_light(name, scene):
    emitter = scene[name]
    cache = scene.cached['psdr_jit']
    psdr_scene = cache['scene']
    
    # Create the object if it has not been created
    if name not in cache['name_map']:
        radiance = convert_color(emitter['radiance'], 3)

        psdr_emitter = psdr_jit.EnvironmentMap()
        psdr_emitter.radiance = radiance
        psdr_emitter.set_transform(Matrix4fD(emitter['to_world'].reshape(1, 4, 4)))
        psdr_scene.add_EnvironmentMap(psdr_emitter)
        cache['name_map'][name] = f"Emitter[{psdr_scene.get_num_emitters() - 1}]"

    psdr_emitter = psdr_scene.param_map[cache['name_map'][name]]

    # Update parameters
    updated = emitter.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == 'radiance':
                psdr_emitter.radiance = convert_color(emitter['radiance'], 3)
            elif param_name == 'to_world':
                psdr_emitter.set_transform(Matrix4fD(emitter['to_world'].reshape(1, 4, 4)))
            emitter.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    drjit_params = []
    
    def enable_grad(drjit_param):
        drjit.enable_grad(drjit_param)
        drjit_params.append(drjit_param)

    requiring_grad = emitter.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == 'radiance':
                enable_grad(psdr_emitter.radiance.data)
                
    return drjit_params
"""

@PyTorch3DConnector.register(PointLight)
def process_point_light(name, scene):
    light = scene[name]
    cache = scene.cached['pytorch3d']

    # Create the object if it has not been created
    if name not in cache['name_map']:
        pytorch_point_light = {
            'radiance': light['radiance'],
            'position': light['position']
        }
        
        cache['point_light'] = pytorch_point_light
        cache['name_map'][name] = pytorch_point_light

    pytorch_point_light = cache['name_map'][name]
    
    # Update parameters
    updated = light.get_updated()
    if len(updated) > 0:
        for param_name in updated:
            if param_name == "position":
                pytorch_point_light['position'] = light['position']
            light.params[param_name]['updated'] = False

    # Enable grad for parameters requiring grad
    torch_params = []
    requiring_grad = light.get_requiring_grad()
    if len(requiring_grad) > 0:
        for param_name in requiring_grad:
            if param_name == "position":
                pytorch_point_light['position'].requires_grad_()
                torch_params.append(pytorch_point_light['position'])
                
    return torch_params