from abc import ABC, abstractmethod

class Connector(ABC):
    
    @abstractmethod
    def renderC(self, scene, sensor_ids=[0]):
        pass
    
    @abstractmethod
    def renderD(self, scene, image_grads, sensor_ids=[0]):
        pass