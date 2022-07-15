from abc import ABC, abstractmethod
import stevedore.extension

class Connector(ABC):
    
    @abstractmethod
    def renderC(self, scene, render_options, sensor_ids=[0]):
        pass
    
    @abstractmethod
    def renderD(self, image_grads, scene, render_options, sensor_ids=[0]):
        pass

class ConnectorManager:

    def __init__(self, quiet=True):

        def on_load_failure_callback(manager, entrypoint, exception):
            print(f'Failed to load {entrypoint}: {exception}')

        self.em = stevedore.extension.ExtensionManager('ivt_connectors',
            on_load_failure_callback=None if quiet else on_load_failure_callback
        )

    def is_available(self, connector_name):
        return connector_name in self.em

    def get_availability_list(self):
        return [ext.name for ext in self.em]

    def get_connector(self, connector_name):
        return self.em[connector_name].plugin()