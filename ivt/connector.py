from abc import ABC, abstractmethod
import stevedore.extension

class Connector(ABC):
    
    @abstractmethod
    def renderC(self, scene, sensor_ids=[0]):
        pass
    
    @abstractmethod
    def renderD(self, scene, image_grads, sensor_ids=[0]):
        pass

class ConnectorManager:

    def __init__(self):
        self.em = stevedore.extension.ExtensionManager('ivt_connectors')

    def is_available(self, connector_name):
        return connector_name in self.em

    def get_availability_list(self):
        return [ext.name for ext in self.em]

    def get_connector(self, connector_name):
        return self.em[connector_name].plugin()