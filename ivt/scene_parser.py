from abc import ABC, abstractmethod
import stevedore.extension

class SceneParser(ABC):
    
    @abstractmethod
    def read(self, scene_path, scene):
        pass
    
    @abstractmethod
    def write(self, scene_path, scene):
        pass

class SceneParserManager:

    def __init__(self, quiet=True):

        def on_load_failure_callback(manager, entrypoint, exception):
            print(f'Failed to load {entrypoint}: {exception}')

        self.em = stevedore.extension.ExtensionManager('ivt_parsers',
            on_load_failure_callback=None if quiet else on_load_failure_callback
        )

    def is_available(self, scene_parser_name):
        return scene_parser_name in self.em

    def get_availability_list(self):
        return [ext.name for ext in self.em]

    def get_scene_parser(self, scene_parser_name):
        return self.em[scene_parser_name].plugin()