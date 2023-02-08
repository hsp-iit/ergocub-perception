import __main__
import collections
import inspect
import json
from pathlib import Path

# Useful links
# https://stackoverflow.com/questions/980249/difference-between-dir-and-vars-keys-in-python
# https://stackoverflow.com/questions/9659361/order-of-results-from-pythons-dir-function
# https://stackoverflow.com/questions/980249/difference-between-dir-and-vars-keys-in-python

class IterMeta(type):

    def __init__(cls, name, bases, clsdict):
        if len(cls.mro()) == 3:
            cls._subclass()
        super(IterMeta, cls).__init__(name, bases, clsdict)

    def __iter__(cls):
        # With dict, we obtain the class attributes and the ones from the base classes in alphabetical order
        # With vars we obtain only the class attributes in order of definition
        # https://stackoverflow.com/questions/980249/difference-between-dir-and-vars-keys-in-python

        target_attr = list(vars(cls).keys())
        # This removes variables from superclasses
        for i in range(3, len(cls.__mro__) - 1):
            # The if allows inheritance between configs, but I guess there are better solutions
            if 'configs' not in cls.__mro__[i].__module__:
                target_attr = target_attr - set(dir(cls.__mro__[i]))
        target_attr = list(filter(lambda x: not x.startswith('_') and x not in cls._skip_attr, target_attr))

        for k in target_attr:
            yield getattr(cls, k)


def init(cls):
    cls._subclass()


class BaseConfig(metaclass=IterMeta):
    _skip_attr = ['to_dict', 'to_json', 'to_list', 'init', 'to_flat_dict']

    @classmethod
    def to_json(cls):
        return json.dumps(cls.to_dict())

    @classmethod
    def to_dict(cls):
        target = cls

        res = {}

        target_attr = set(dir(target))
        # This removes variables from superclasses
        for i in range(3, len(target.__mro__) - 1):
            # The if allows inheritance between configs but I guess there are better solutions
            if 'configs' not in target.__mro__[i].__module__:
                target_attr = target_attr - set(dir(target.__mro__[i]))

        for k in target_attr:
            if not k.startswith('_') and k not in ['to_dict', 'to_json', 'to_list', 'init', 'to_flat_dict']:
                attr = getattr(target, k)

                # If it's a module get inside
                if hasattr(attr, '__module__'):

                    # If it's a class inside config, get inside it,
                    # else just log module and name in the dict as a string.

                    # if we are executing the config the module is __main__. If we are importing it is config
                    if type(attr).__name__ == 'function':
                        if attr.__name__ == '<lambda>':
                            funcString = str(inspect.getsourcelines(attr)[0])
                            res[k] = funcString.strip("['\\n']").split(" = ")[1]
                        else:
                            res[k] = f'function : {attr.__name__}'
                    elif attr.__module__.split('.')[0] == '__main__' or 'config' in attr.__module__:
                        res[k] = attr.to_dict()
                    else:
                        # End up here if attr is not a class defined inside module.
                        # e.g. built-in types, functions, etc.
                        if type(attr).__name__ == 'type':
                            name = attr.__name__
                        else:
                            name = type(attr).__name__
                        res[k] = f'{attr.__module__}.{name}'
                # If it's not a class save it. This is done for basic types.
                # Could cause problems with complex objects
                else:
                    res[k] = attr
        return res

    @classmethod
    def _subclass(cls):

        target = cls

        res = {}
        for k in dir(target):
            if not k.startswith('_') and k not in ['to_dict', 'to_json', 'to_list', 'init', 'to_flat_dict']:
                attr = getattr(target, k)
                # If it's a class inside config, get inside it,
                # else just log module and name in the dict as a string
                if hasattr(attr, '__module__') and type(attr).__name__ != 'function':

                    # if we are executing the config the module is __main__. If we are importing it is config
                    # Not ideal but config could be anywhere in the name
                    # Need to find a better way to do this
                    if attr.__module__.split('.')[0] == '__main__' or 'config' in attr.__module__:
                        # This way the module keeps its subclasses but it is also subclassed by
                        # BaseConfig inheriting its method. A security check could be used to assure
                        # that the new methods are not overriding any old one.
                        setattr(target, k, type(f'{k}_mod', (BaseConfig,) + tuple(attr.__mro__), dict(
                            list(dict(vars(BaseConfig)).items()) + list(dict(vars(attr)).items()))))
                        getattr(target, k)._subclass()
        return res

    @classmethod
    def to_flat_dict(cls) -> dict:
        import torch
        res = cls.to_dict()
        res = flatten(res)
        # res = {k.split('.')[-1]: torch.tensor(v) if isinstance(v, list) else v for k, v in res.items()}
        return res

    @classmethod
    def to_list(cls):
        target = cls

        res = []
        for k in dir(target):
            if not k.startswith('_') and k not in ['to_dict', 'to_json', 'to_list', 'init', 'to_flat_dict']:
                attr = getattr(target, k)
                # If it's a class inside config, get inside it,
                # else just log module and name in the dict as a string
                if type(attr) == type:
                    if attr.__module__.split('.')[0] in ['configs', '__main__']:
                        res.append(attr.to_list())
                    else:
                        res.append(f'{attr.__module__}.{attr.__name__}')
                # If it's not a class save it. This is done for basic types.
                # Could cause problems with complex objects
                else:
                    res.append(attr)
        return res

    def __getattribute__(self, item):
        return object.__getattribute__(self, item)


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def to_class(module):
    return type(module.__name__, (),
                {key: staticmethod(value) if callable(value) else value
                 for key, value in ((name, getattr(module, name))
                                    for name in dir(module))})

# def to_base_class(module):
#     cls = to_class(module)
#
#     target = BaseConfig
#     for k in dir(target):
#         if k not in ['__class__', '__dict__']:
#             setattr(cls, k, getattr(target, k))
#
#     cls._subclass()
#     return cls
