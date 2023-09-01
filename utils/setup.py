from setuptools import setup, find_packages
from pkg_resources import parse_requirements

setup(
    name='utils',
    version='1.0',
    description='',
    author='Riccardo Emanuele Landi',
    author_email='riccardo.landi@innoida.it',
    packages=find_packages(),
    install_requires=[
        'matplotlib==3.7.2',
        'numpy==1.24.4'
    ]
)