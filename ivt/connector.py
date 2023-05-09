from abc import ABC, abstractmethod

_connector_table = {}


class Connector(ABC):

    def __init_subclass__(cls, connector_name, **kwargs):
        super().__init_subclass__(**kwargs)
        _connector_table[connector_name] = cls
        cls.extensions = {}

    @abstractmethod
    def renderC(self, scene, render_options, sensor_ids=[0], integrator_id=0):
        pass

    @abstractmethod
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0], integrator_id=0):
        pass

    @classmethod
    def register(cls, class_name):
        def wrapper(func):
            cls.extensions[class_name] = func
            return func
        return wrapper

def is_connector_available(connector_name):
    return connector_name in _connector_table


def get_connector_list():
    return list(_connector_table.keys())


def get_connector(connector_name):
    return _connector_table[connector_name]()
