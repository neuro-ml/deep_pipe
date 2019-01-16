from distutils.core import setup
from setuptools import find_packages

from dpipe import __version__

classifiers = '''Development Status :: 3 - Alpha
Programming Language :: Python :: 3.6'''

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

setup(
    name='deep_pipe',
    packages=find_packages(include=('deep_pipe',)),
    include_package_data=True,
    version=__version__,
    long_description=long_description,
    license='MIT',
    url='https://github.com/neuro-ml/deep_pipe',
    download_url='https://github.com/neuro-ml/deep_pipe/v%s.tar.gz' % __version__,
    keywords=[],
    classifiers=classifiers.splitlines(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'dpipe = dpipe.config.base:render_config_resource',
        ],
    },
)
