"""Inverse-Rendering Toolkit"""

__version__ = "0.1.0"

from .connector import is_connector_available, get_connector_list, get_connector

try:
    from .connectors.psdr_jit_connector import PSDRJITConnector
except:
    pass

try:
    from .connectors.pytorch3d_connector import PyTorch3DConnector
except:
    pass

from irt.renderer import Renderer
