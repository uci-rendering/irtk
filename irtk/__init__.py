"""Inverse-Rendering Toolkit"""

__version__ = "0.1.5"

from .connector import is_connector_available, get_connector_list, get_connector

try:
    from .connectors.psdr_jit_connector import PSDRJITConnector
except:
    pass

try:
    from .connectors.pytorch3d_connector import PyTorch3DConnector
except:
    pass

try:
    from .connectors.nvdiffrast_connector import NvdiffrastConnector
except:
    pass

try:
    from .connectors.mitsuba_connector import MitsubaConnector
except:
    pass

try:
    from .connectors.redner_connector import RednerConnector
except:
    pass

from .renderer import Renderer
