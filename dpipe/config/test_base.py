import unittest

from .base import get_resource, module_type2path


class TestModuleBuildersAccess(unittest.TestCase):
    def test_modules_access(self):
        for module_type in module_type2path:
            builders = get_resource(module_type)
