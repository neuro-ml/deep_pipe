import unittest

from .base import get_module_builders, module_type2path


class TestModuleBuildersAccess(unittest.TestCase):
    def test_modules_access(self):
        for module_type in module_type2path:
            builders = get_module_builders(module_type)
