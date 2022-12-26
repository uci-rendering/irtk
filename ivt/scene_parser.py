from abc import ABC, abstractmethod

_scene_parser_table = {}

def register_scene_parser(cls):
    _scene_parser_table[cls.scene_parser_name] = cls

class SceneParser(ABC):
    def __init_subclass__(cls, **kwargs):
          # always make it colaborative:
          super().__init_subclass__(**kwargs)
          register_scene_parser(cls)

    @property
    @abstractmethod
    def scene_parser_name(self):
        pass
    
    @abstractmethod
    def read(self, scene_path, scene):
        pass
    
    @abstractmethod
    def write(self, scene_path, scene):
        pass

def is_scene_parser_available(scene_parser_name):
    return scene_parser_name in _scene_parser_table

def get_scene_parser_list():
    return list(_scene_parser_table.keys())

def get_scene_parser(scene_parser_name):
    return _scene_parser_table[scene_parser_name]()