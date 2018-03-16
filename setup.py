from distutils.core import setup
from setuptools import find_packages

classifiers = '''Development Status :: 3 - Alpha
Programming Language :: Python :: 3.6'''

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

version = '0.0.4'

setup(
    name='deep_pipe',
    packages=find_packages(),
    include_package_data=True,
    version=version,
    long_description=long_description,
    license='MIT',
    url='https://github.com/neuro-ml/deep_pipe',
    download_url='https://github.com/neuro-ml/deep_pipe/v%s.tar.gz' % version,
    keywords=[],
    classifiers=classifiers.splitlines(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn>=0.19',
        'snakemake',
        'tqdm',
        'pdp>=0.2.1,<0.3',
        'tensorboard-easy',
        'resource-manager~=0.6.4',
    ]
)
