from abc import ABC, abstractmethod

class Connector(ABC):
    
    @abstractmethod
    def renderC(self, scene, sensor_ids=[0]):
        pass
    
    @abstractmethod
    def renderD(self, scene, target_image, image_loss_func, sensor_ids=[0]):
        pass