from .config import TensorLike
from .parameter import ParamGroup
from .transform import (
    lookat,
    perspective,
    perspective_full,
    batched_transform_pos,
    batched_transform_dir,
)
from .io import read_image, read_mesh, to_torch_f, to_torch_i

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class Scene:
    """A class representing a scene with various components."""

    def __init__(self) -> None:
        """Initializes a Scene object."""
        self.components: Dict[str, Any] = OrderedDict()
        self.requiring_grad: Tuple[str, ...] = ()
        self.cached: Dict[str, Any] = {}

    def set(self, name: str, component: ParamGroup) -> None:
        """Sets a component in the scene.

        Args:
            name (str): The name of the component.
            component (ParamGroup): The component to be added.
        """
        self.components[name] = component

    def __getitem__(self, name: str) -> Any:
        """Gets a component by name.

        Args:
            name (str): The name of the component.

        Returns:
            Any: The component.
        """
        item = self.components
        for c in name.split("."):
            item = item[c]
        return item

    def __setitem__(self, name: str, param_value: Any) -> None:
        """Sets a parameter value for a component.

        Args:
            name (str): The name of the component and parameter.
            param_value (Any): The value to be set.
        """
        component_name, param_name = name.rsplit(".", 1)
        self[component_name][param_name] = param_value

    def __contains__(self, name: str) -> bool:
        """Checks if a component exists in the scene.

        Args:
            name (str): The name of the component.

        Returns:
            bool: True if the component exists, False otherwise.
        """
        item = self.components
        for c in name.split("."):
            if c not in item:
                return False
            item = item[c]
        return True

    def configure(self) -> None:
        """Configures the scene by identifying components requiring gradients."""
        requiring_grad = []
        for cname in self.components:
            requiring_grad += [
                f"{cname}.{pname}"
                for pname in self.components[cname].get_requiring_grad()
            ]
        self.requiring_grad = tuple(requiring_grad)

    def __str__(self) -> str:
        """Returns a string representation of the scene.

        Returns:
            str: A string representation of the scene.
        """
        lines = []

        for name in self.components:
            lines.append(f"{name}:")
            lines.append(str(self.components[name]))
            lines.append("\n")

        return "\n".join(lines)

    def clear_cache(self) -> None:
        """Clears the cache and detaches tensors requiring gradients."""
        for connector in self.cached.keys():
            if "clean_up" in self.cached[connector]:
                self.cached[connector]["clean_up"]()
        self.cached = {}
        # Detach the tensors requiring grad
        for param_name in self.requiring_grad:
            self[param_name] = self[param_name].detach()

    def filter(self, component_type: type) -> List[str]:
        """Filters components by type.

        Args:
            component_type (type): The type of components to filter.

        Returns:
            List[str]: A list of component names matching the specified type.
        """
        return [
            cname
            for cname in self.components
            if component_type == type(self.components[cname])
        ]


class Integrator(ParamGroup):
    """A class representing an integrator."""

    def __init__(self, type: str, config: Optional[Dict[str, Any]] = {}) -> None:
        """Initializes an Integrator object.

        Args:
            type (str): The type of the integrator.
            config (Optional[Dict[str, Any]]): The configuration of the integrator.
        """
        super().__init__()

        self.add_param("type", type, help_msg="integrator type")
        self.add_param("config", config, help_msg="integrator config")


class HDRFilm(ParamGroup):
    """A class representing an HDR film."""

    def __init__(
        self,
        width: int,
        height: int,
        crop_window: Optional[TensorLike] = None,  # float
        pixel_idx: Optional[TensorLike] = None,  # int
    ) -> None:
        """Initializes an HDRFilm object.

        Args:
            width (int): The width of the film.
            height (int): The height of the film.
            crop_window (Optional[TensorLike]): The crop window of the film (float).
            pixel_idx (Optional[TensorLike]): The pixel indices of the film (int).
        """
        super().__init__()

        self.add_param("width", width, help_msg="film width")
        self.add_param("height", height, help_msg="film height")
        self.add_param(
            "crop_window",
            crop_window,
            help_msg="film crop window (format: [h_lower, w_lower, h_upper, w_upper])",
        )


class PerspectiveCamera(ParamGroup):
    """A class representing a perspective camera."""

    def __init__(
        self,
        fov: float,
        to_world: TensorLike,  # float, shape (4, 4)
        near: float = 1e-6,
        far: float = 1e7,
    ) -> None:
        """Initializes a PerspectiveCamera object.

        Args:
            fov (float): The field of view of the camera.
            to_world (TensorLike): The transformation matrix to world coordinates (float, shape (4, 4)).
            near (float): The near clipping plane.
            far (float): The far clipping plane.
        """
        super().__init__()

        self.add_param("fov", fov, help_msg="sensor fov")
        self.add_param("near", near, help_msg="sensor near clip")
        self.add_param("far", far, help_msg="sensor far clip")
        self.add_param(
            "to_world",
            to_torch_f(to_world),
            is_tensor=True,
            is_diff=True,
            help_msg="sensor to_world matrix",
        )

    def get_rays(
        self, samples: torch.Tensor, aspect_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates rays from the camera.

        Args:
            samples (torch.Tensor): The sample points.
            aspect_ratio (float): The aspect ratio of the camera.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the ray origins and directions.
        """
        samples = torch.cat([samples, torch.zeros_like(samples)[:, 0:1]], dim=1)
        sample_to_camera = torch.inverse(
            perspective(self["fov"], aspect_ratio, self["near"], self["far"])
        )
        rays_o = batched_transform_pos(self["to_world"], to_torch_f([[0, 0, 0]]))
        rays_d = F.normalize(batched_transform_pos(sample_to_camera, samples), dim=1)
        rays_d = batched_transform_dir(self["to_world"], rays_d)
        return rays_o.repeat(samples.shape[0], 1), rays_d

    @classmethod
    def from_lookat(
        cls,
        fov: float,
        origin: TensorLike,  # float, shape (3, )
        target: TensorLike,  # float, shape (3, )
        up: TensorLike,  # float, shape (3, )
        near: float = 1e-6,
        far: float = 1e7,
    ) -> "PerspectiveCamera":
        """Creates a PerspectiveCamera object from lookat parameters.

        Args:
            fov (float): The field of view of the camera.
            origin (TensorLike): The origin of the camera (float, shape (3, )).
            target (TensorLike): The target point of the camera (float, shape (3, )).
            up (TensorLike): The up vector of the camera (float, shape (3, )).
            near (float): The near clipping plane.
            far (float): The far clipping plane.

        Returns:
            PerspectiveCamera: A PerspectiveCamera object.
        """
        sensor = cls(fov, torch.eye(4), near, far)
        origin = to_torch_f(origin)
        target = to_torch_f(target)
        up = to_torch_f(up)
        sensor["to_world"] = lookat(origin, target, up)
        return sensor


# To be merged with PerspectiveCamera
class PerspectiveCameraFull(ParamGroup):
    """A class representing a full perspective camera."""

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        to_world: TensorLike = torch.eye(4),  # float, shape (4, 4)
        near: float = 1e-6,
        far: float = 1e7,
    ) -> None:
        """Initializes a PerspectiveCameraFull object.

        Args:
            fx (float): The focal length in the x axis.
            fy (float): The focal length in the y axis.
            cx (float): The principal point offset in the x axis.
            cy (float): The principal point offset in the y axis.
            to_world (TensorLike): The transformation matrix to world coordinates (float, shape (4, 4)).
            near (float): The near clipping plane.
            far (float): The far clipping plane.
        """
        super().__init__()

        self.add_param("fx", fx, help_msg="sensor focal length in x axis")
        self.add_param("fy", fy, help_msg="sensor focal length in y axis")
        self.add_param("cx", cx, help_msg="sensor principal point offset in x axis")
        self.add_param("cy", cy, help_msg="sensor principal point offset in y axis")
        self.add_param("near", near, help_msg="sensor near clip")
        self.add_param("far", far, help_msg="sensor far clip")
        self.add_param(
            "to_world",
            to_torch_f(to_world),
            is_tensor=True,
            is_diff=True,
            help_msg="sensor to_world matrix",
        )

    def get_rays(
        self, samples: torch.Tensor, aspect_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates rays from the camera.

        Args:
            samples (torch.Tensor): The sample points.
            aspect_ratio (float): The aspect ratio of the camera.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the ray origins and directions.
        """
        samples = torch.cat([samples, torch.zeros_like(samples)[:, 0:1]], dim=1)
        sample_to_camera = torch.inverse(
            perspective_full(
                self["fx"],
                self["fy"],
                self["cx"],
                self["cy"],
                aspect_ratio,
                self["near"],
                self["far"],
            )
        )
        rays_o = batched_transform_pos(self["to_world"], to_torch_f([[0, 0, 0]]))
        rays_d = F.normalize(batched_transform_pos(sample_to_camera, samples), dim=1)
        rays_d = batched_transform_dir(self["to_world"], rays_d)
        return rays_o.repeat(samples.shape[0], 1), rays_d

    @classmethod
    def from_lookat(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        origin: TensorLike,  # float, shape (3, )
        target: TensorLike,  # float, shape (3, )
        up: TensorLike,  # float, shape (3, )
        near: float = 1e-6,
        far: float = 1e7,
    ) -> "PerspectiveCameraFull":
        """Creates a PerspectiveCameraFull object from lookat parameters.

        Args:
            fx (float): The focal length in the x axis.
            fy (float): The focal length in the y axis.
            cx (float): The principal point offset in the x axis.
            cy (float): The principal point offset in the y axis.
            origin (TensorLike): The origin of the camera (float, shape (3, )).
            target (TensorLike): The target point of the camera (float, shape (3, )).
            up (TensorLike): The up vector of the camera (float, shape (3, )).
            near (float): The near clipping plane.
            far (float): The far clipping plane.

        Returns:
            PerspectiveCameraFull: A PerspectiveCameraFull object.
        """
        origin = to_torch_f(origin)
        target = to_torch_f(target)
        up = to_torch_f(up)
        to_world = lookat(origin, target, up)
        return cls(fx, fy, cx, cy, to_world, near=near, far=far)


class Mesh(ParamGroup):
    """A class representing a mesh."""

    def __init__(
        self,
        v: TensorLike,  # float, shape (num_v, 3)
        f: TensorLike,  # int, shape (num_f, 3)
        uv: TensorLike,  # float, shape (num_uv, 2)
        fuv: TensorLike,  # int, shape (num_f, 3)
        mat_id: Optional[str] = None,
        to_world: TensorLike = torch.eye(4),  # float
        use_face_normal: bool = True,
        can_change_topology: bool = False,
        radiance: Optional[TensorLike] = None,  # float, shape (3, )
    ) -> None:
        """Initializes a Mesh object.

        Args:
            v (TensorLike): The vertex positions (float, shape (num_v, 3)).
            f (TensorLike): The face indices (int, shape (num_f, 3)).
            uv (TensorLike): The uv coordinates (float, shape (num_uv, 2)).
            fuv (TensorLike): The uv face indices (int, shape (num_f, 3)).
            mat_id (Optional[str]): The material ID.
            to_world (TensorLike): The transformation matrix to world coordinates (float).
            use_face_normal (bool): Whether to use face normal.
            can_change_topology (bool): Whether the topology can be changed. This might lead to certain optimizations.
            radiance (Optional[TensorLike]): The radiance if used as an emitter (float, shape (3, )).
        """
        super().__init__()

        self.add_param(
            "v",
            to_torch_f(v),
            is_tensor=True,
            is_diff=True,
            help_msg="mesh vertex positions",
        )
        self.add_param("f", to_torch_i(f), is_tensor=True, help_msg="mesh face indices")
        self.add_param(
            "uv", to_torch_f(uv), is_tensor=True, help_msg="mesh uv coordinates"
        )
        self.add_param(
            "fuv", to_torch_i(fuv), is_tensor=True, help_msg="mesh uv face indices"
        )
        self.add_param(
            "to_world",
            to_torch_f(to_world),
            is_tensor=True,
            is_diff=True,
            help_msg="mesh to world matrix",
        )
        self.add_param(
            "use_face_normal", use_face_normal, help_msg="whether to use face normal"
        )
        self.add_param(
            "can_change_topology",
            can_change_topology,
            help_msg="whether to the topology can be chagned",
        )

        assert not (mat_id is None and radiance is None)

        if mat_id is not None:
            self.add_param(
                "mat_id",
                mat_id,
                help_msg="name of the material of the mesh if used as an non-emitter",
            )

        if radiance is not None:
            radiance = to_torch_f(radiance)
            self.add_param(
                "radiance",
                radiance,
                is_tensor=True,
                is_diff=True,
                help_msg="radiance if used as an emitter",
            )

    @classmethod
    def from_file(
        cls,
        filename: str,
        mat_id: Optional[str] = None,
        to_world: TensorLike = torch.eye(4),  # float
        use_face_normal: bool = True,
        can_change_topology: bool = False,
        radiance: Optional[TensorLike] = None,  # float, shape (3, )
        flip_tex: bool = True,
    ) -> "Mesh":
        """Creates a Mesh object from a file.

        Args:
            filename (str): The filename of the mesh.
            mat_id (Optional[str]): The material ID.
            to_world (TensorLike): The transformation matrix to world coordinates (float).
            use_face_normal (bool): Whether to use face normal.
            can_change_topology (bool): Whether the topology can be changed.
            radiance (Optional[TensorLike]): The radiance if used as an emitter (float, shape (3, )).
            flip_tex (bool): Whether to flip the texture coordinates.

        Returns:
            Mesh: A Mesh object.
        """
        v, f, uv, fuv = read_mesh(filename)
        if flip_tex:
            uv[:, 1] = 1 - uv[:, 1]
        return cls(
            v,
            f,
            uv,
            fuv,
            mat_id,
            to_world,
            use_face_normal,
            can_change_topology,
            radiance,
        )


class DiffuseBRDF(ParamGroup):
    """A class representing a diffuse BRDF."""

    def __init__(self, d: TensorLike) -> None:  # float, shape (3, ) or (m, n, 3)
        """Initializes a DiffuseBRDF object.

        Args:
            d (TensorLike): The diffuse reflectance (float, shape (3, ) or (m, n, 3)).
        """
        super().__init__()

        self.add_param(
            "d",
            to_torch_f(d),
            is_tensor=True,
            is_diff=True,
            help_msg="diffuse reflectance",
        )

    @classmethod
    def from_file(cls, filename: str, is_srgb: Optional[bool] = None) -> "DiffuseBRDF":
        """Creates a DiffuseBRDF object from a file.

        Args:
            filename (str): The filename of the texture.
            is_srgb (Optional[bool]): Whether the texture is in sRGB format.

        Returns:
            DiffuseBRDF: A DiffuseBRDF object.
        """
        texture = read_image(filename, is_srgb)
        return cls(texture)


class MicrofacetBRDF(ParamGroup):
    """A class representing a microfacet BRDF."""

    def __init__(
        self,
        d: TensorLike,  # float, shape (3, ) or (m, n, 3)
        s: TensorLike,  # float, shape (3, ) or (m, n, 3)
        r: TensorLike,  # float, shape (1, ) or (m, n, 1)
    ) -> None:
        """Initializes a MicrofacetBRDF object.

        Args:
            d (TensorLike): The diffuse reflectance (float, shape (3, ) or (m, n, 3)).
            s (TensorLike): The specular reflectance (float, shape (3, ) or (m, n, 3)).
            r (TensorLike): The roughness (float, shape (1, ) or (m, n, 1)).
        """
        super().__init__()

        self.add_param(
            "d",
            to_torch_f(d),
            is_tensor=True,
            is_diff=True,
            help_msg="diffuse reflectance",
        )
        self.add_param(
            "s",
            to_torch_f(s),
            is_tensor=True,
            is_diff=True,
            help_msg="specular reflectance",
        )
        self.add_param(
            "r", to_torch_f(r), is_tensor=True, is_diff=True, help_msg="roughness"
        )

    @classmethod
    def from_file(
        cls,
        d_filename: str,
        s_filename: str,
        r_filename: str,
        d_is_srgb: Optional[bool] = None,
        s_is_srgb: Optional[bool] = None,
        r_is_srgb: Optional[bool] = None,
    ) -> "MicrofacetBRDF":
        """Creates a MicrofacetBRDF object from files.

        Args:
            d_filename (str): The filename of the diffuse texture.
            s_filename (str): The filename of the specular texture.
            r_filename (str): The filename of the roughness texture.
            d_is_srgb (Optional[bool]): Whether the diffuse texture is in sRGB format.
            s_is_srgb (Optional[bool]): Whether the specular texture is in sRGB format.
            r_is_srgb (Optional[bool]): Whether the roughness texture is in sRGB format.

        Returns:
            MicrofacetBRDF: A MicrofacetBRDF object.
        """
        d_texture = read_image(d_filename, d_is_srgb)
        s_texture = read_image(s_filename, s_is_srgb)
        r_texture = read_image(r_filename, r_is_srgb)[..., 0:1]

        return cls(d_texture, s_texture, r_texture)


class SmoothDielectricBRDF(ParamGroup):
    """A class representing a smooth dielectric BRDF."""

    def __init__(
        self,
        int_ior: float,
        ext_ior: float,
        s_reflect: TensorLike,  # float
        s_transmit: TensorLike,  # float
    ) -> None:
        """Initializes a SmoothDielectricBRDF object.

        Args:
            int_ior (float): The interior index of refraction.
            ext_ior (float): The exterior index of refraction.
            s_reflect (TensorLike): The specular reflection component (float).
            s_transmit (TensorLike): The specular transmission component (float).
        """
        super().__init__()

        self.add_param(
            "int_ior",
            int_ior,
            is_tensor=False,
            is_diff=False,
            help_msg="interior index of refraction",
        )
        self.add_param(
            "ext_ior",
            ext_ior,
            is_tensor=False,
            is_diff=False,
            help_msg="exterior index of refraction",
        )
        self.add_param(
            "s_reflect",
            to_torch_f(s_reflect),
            is_tensor=True,
            is_diff=False,
            help_msg="specular reflection component",
        )
        self.add_param(
            "s_transmit",
            to_torch_f(s_transmit),
            is_tensor=True,
            is_diff=False,
            help_msg="specular transmission component",
        )


class RoughDielectricBSDF(ParamGroup):
    """A class representing a rough dielectric BSDF."""

    def __init__(self, alpha: TensorLike, i_ior: float, e_ior: float) -> None:  # float
        """Initializes a RoughDielectricBSDF object.

        Args:
            alpha (TensorLike): The roughness (float).
            i_ior (float): The interior index of refraction.
            e_ior (float): The exterior index of refraction.
        """
        super().__init__()
        self.add_param("alpha", to_torch_f(alpha), help_msg="roughness")
        self.add_param(
            "i_ior", to_torch_f(i_ior), help_msg="interior index of refraction"
        )
        self.add_param(
            "e_ior", to_torch_f(e_ior), help_msg="exterior index of refraction"
        )


class RoughConductorBRDF(ParamGroup):
    """A class representing a rough conductor BRDF."""

    def __init__(
        self,
        alpha_u: TensorLike,  # float
        alpha_v: TensorLike,  # float
        eta: TensorLike,  # float
        k: TensorLike,  # float
        s: TensorLike,  # float
    ) -> None:
        """Initializes a RoughConductorBRDF object.

        Args:
            alpha_u: The alpha_u parameter (float).
            alpha_v: The alpha_v parameter (float).
            eta: The eta parameter (float).
            k: The k parameter (float).
            s: The specular reflectance (float).
        """
        super().__init__()

        self.add_param(
            "alpha_u",
            to_torch_f(alpha_u),
            is_tensor=True,
            is_diff=True,
            help_msg="alpha_u",
        )
        self.add_param(
            "alpha_v",
            to_torch_f(alpha_v),
            is_tensor=True,
            is_diff=True,
            help_msg="alpha_v",
        )
        self.add_param(
            "eta", to_torch_f(eta), is_tensor=True, is_diff=True, help_msg="eta"
        )
        self.add_param("k", to_torch_f(k), is_tensor=True, is_diff=True, help_msg="k")
        self.add_param(
            "s",
            to_torch_f(s),
            is_tensor=True,
            is_diff=True,
            help_msg="specular reflectance",
        )

    # @classmethod
    # def from_file(cls, d_filename, s_filename, r_filename, d_is_srgb=None, s_is_srgb=None, r_is_srgb=None):
    #     d_texture = read_image(d_filename, d_is_srgb)
    #     s_texture = read_image(s_filename, s_is_srgb)
    #     r_texture = read_image(r_filename, r_is_srgb)[..., 0:1]

    #     return cls(d_texture, s_texture, r_texture)


class EnvironmentLight(ParamGroup):
    """A class representing an environment light."""

    def __init__(
        self, radiance: TensorLike, to_world: TensorLike = torch.eye(4)  # float  # float
    ) -> None:
        """Initializes an EnvironmentLight object.

        Args:
            radiance: The environment light radiance (float, shape (m, n, 3)).
            to_world: The transformation matrix to world coordinates (float, shape (4, 4)).
        """
        super().__init__()

        self.add_param(
            "radiance",
            to_torch_f(radiance),
            is_tensor=True,
            is_diff=True,
            help_msg="environment light radiance (shape: (m, n, 3))",
        )
        self.add_param(
            "to_world",
            to_torch_f(to_world),
            is_tensor=True,
            is_diff=False,
            help_msg="environment to_world matrix (shape: (4, 4))",
        )

    @classmethod
    def from_file(
        cls,
        radiance_filename: str,
        radiance_is_srgb: Optional[bool] = None,
        to_world: TensorLike = torch.eye(4),  # float, shape (4, 4)
    ) -> "EnvironmentLight":
        """Creates an EnvironmentLight object from a file.

        Args:
            radiance_filename: The filename of the radiance texture.
            radiance_is_srgb: Whether the radiance texture is in sRGB format.
            to_world: The transformation matrix to world coordinates (float, shape (4, 4)).

        Returns:
            An EnvironmentLight object.
        """
        radiance_texture = read_image(radiance_filename, radiance_is_srgb)

        return cls(radiance_texture, to_world)


class PointLight(ParamGroup):
    """A class representing a point light."""

    def __init__(
        self, radiance: TensorLike, position: TensorLike  # float
    ) -> None:  # float
        """Initializes a PointLight object.

        Args:
            radiance: The point light radiance (float).
            position: The point light position (float).
        """
        super().__init__()

        self.add_param(
            "radiance",
            to_torch_f(radiance),
            is_tensor=True,
            is_diff=False,
            help_msg="point light radiance",
        )
        self.add_param(
            "position",
            to_torch_f(position),
            is_tensor=True,
            is_diff=True,
            help_msg="point light position",
        )
