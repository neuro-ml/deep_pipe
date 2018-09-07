from types import FunctionType, MethodType, ModuleType
from sphinx.ext.autodoc import importer


class _MockObject:
    def __new__(cls, *args, **kwargs):
        # type: (Any, Any) -> Any
        if len(args) == 3 and isinstance(args[1], tuple) and args[1][-1].__class__ is cls:
            # subclassing MockObject
            return type(args[0], (_MockObject,), args[2], **kwargs)  # type: ignore
        else:
            return super(_MockObject, cls).__new__(cls)

    def __init__(self, name):
        # type: (Any, Any) -> None
        self.name = name
        self.__qualname__ = ''

    def __len__(self):
        # type: () -> int
        return 0

    def __contains__(self, key):
        # type: (str) -> bool
        return False

    def __iter__(self):
        # type: () -> Iterator
        return iter([])

    def __mro_entries__(self, bases):
        # type: (Tuple) -> Tuple
        return bases

    def __getitem__(self, key):
        # type: (str) -> _MockObject
        return _MockObject(self.name + '.' + key)

    def __getattr__(self, key):
        # type: (str) -> _MockObject
        return _MockObject(self.name + '.' + key)

    def __call__(self, *args, **kw):
        # type: (Any, Any) -> Any
        if args and type(args[0]) in [FunctionType, MethodType]:
            # Appears to be a decorator, pass through unchanged
            return args[0]
        return self

    def __repr__(self):
        return self.name


class _MockModule(ModuleType):
    """Used by autodoc_mock_imports."""
    __file__ = '/dev/null'

    def __init__(self, name, loader):
        self.__name__ = self.__package__ = name
        self.__loader__ = loader
        self.__all__ = []
        self.__path__ = []

    def __getattr__(self, name):
        o = _MockObject(self.__name__ + '.' + name)
        o.__module__ = self.__name__
        return o


importer._MockObject = _MockObject
importer._MockModule = _MockModule
