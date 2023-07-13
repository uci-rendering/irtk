"""Inverse-Rendering Toolkit"""

__version__ = "0.1.5"

from .connector import is_connector_available, get_connector_list, get_connector

try:
    from .connectors.psdr_jit_connector import PSDRJITConnector
except:
    pass

from ivt.renderer import Renderer