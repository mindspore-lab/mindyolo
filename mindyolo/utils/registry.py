import inspect
import os


class Registry:
    """
    a registry that maps string to class
    """

    def __init__(self, name):
        """
        Args:
            name (str): registry name
        """
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f"(name={self._name}, total={len(self._module_dict)})\n"
        class2path = lambda c: os.path.sep.join(c.__module__.split('.')) + '.py'
        format_str += ''.join(
            [f"  ({i}): {k} in {class2path(v)}\n" for i, (k, v) in enumerate(self._module_dict.items())]
                              )
        return format_str

    @property
    def name(self):
        # registry name cannot be changed from outside
        return self._name

    @property
    def module_dict(self):
        # module dict cannot be changed from outside
        return self._module_dict

    def get(self, key):
        """query the registry record"""
        return self._module_dict.get(key, None)

    def registry_module(self, module_name=None):
        """
        Registry a module. A record will be added to 'self._module_dict', whose key is the class name (by default) or
        the specified name, and value is the class itself.
        It is used as a decorator

        Example:
            >>> network = Registry('network')
            >>> # case1: default module name
            >>> @network.registry_module()
            >>> class ResNet()
            >>>     pass
            >>> resnet = network.get('ResNet')
            >>>
            >>> # case2: customized module name
            >>> @network.registry_module('yolov3')
            >>> class YOLOv3()
            >>>     pass
            >>> yolov3 = network.get('yolov3')
        """
        if module_name is not None:
            assert isinstance(module_name, str), f"module_name should be a str but got {type(module_name)} instead"

        # use as a decorator
        def _registry(cls):
            return self._registry_module(module_class=cls, module_name=module_name)

        return _registry

    def _registry_module(self, module_class, module_name=None):
        """
        main worker of registry
        """
        assert inspect.isclass(
            module_class
        ), f"module to register should be a class but got {type(module_class)} instead"
        if module_name in self:
            raise KeyError(f"{module_name} is already registered in {self._name}")
        if module_name is None:
            module_name = module_class.__name__
        self._module_dict[module_name] = module_class

        return module_class
