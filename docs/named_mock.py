from types import FunctionType, MethodType, ModuleType
from sphinx.ext.autodoc import importer


class _MockObject(object):
    __display_name__ = '_MockObject'

    def __new__(cls, *args, **kwargs):
        if len(args) == 3 and isinstance(args[1], tuple):
            superclass = args[1][-1].__class__
            if superclass is cls:
                # subclassing MockObject
                return _make_subclass(args[0], superclass.__display_name__,
                                      superclass=superclass, attributes=args[2])

        return super(_MockObject, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        self.__qualname__ = ''

    def __len__(self):
        return 0

    def __contains__(self, key):
        return False

    def __iter__(self):
        return iter([])

    def __mro_entries__(self, bases):
        return bases

    def __getitem__(self, key):
        return _make_subclass(key, self.__display_name__, self.__class__)()

    def __getattr__(self, key):
        return _make_subclass(key, self.__display_name__, self.__class__)()

    def __call__(self, *args, **kw):
        if args and type(args[0]) in [FunctionType, MethodType]:
            # Appears to be a decorator, pass through unchanged
            return args[0]
        return self

    def __repr__(self):
        return self.__display_name__


def _make_subclass(name, module, superclass=_MockObject, attributes=None):
    attrs = {'__module__': module, '__display_name__': module + '.' + name}
    attrs.update(attributes or {})

    return type(name, (superclass,), attrs)


class _MockModule(ModuleType):
    __file__ = '/dev/null'

    def __init__(self, name, loader):
        self.__name__ = self.__package__ = name
        self.__loader__ = loader
        self.__all__ = []
        self.__path__ = []

    def __getattr__(self, name):
        return _make_subclass(name, self.__name__)()


importer._MockObject = _MockObject
importer._MockModule = _MockModule
