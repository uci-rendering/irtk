from abc import ABC, abstractmethod

_connector_table = {}

def register_connector(cls):
    _connector_table[cls.connector_name] = cls

class Connector(ABC):
    def __init_subclass__(cls, **kwargs):
          # always make it colaborative:
          super().__init_subclass__(**kwargs)
          register_connector(cls)

    @property
    @abstractmethod
    def connector_name(self):
        pass
    
    @abstractmethod
    def renderC(self, scene, render_options, sensor_ids=[0]):
        pass
    
    @abstractmethod
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0]):
        pass

def is_connector_available(connector_name):
    return connector_name in _connector_table

def get_connector_list():
    return list(_connector_table.keys())

def get_connector(connector_name):
    return _connector_table[connector_name]()