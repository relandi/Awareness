from setuptools import setup, find_packages
from pkg_resources import parse_requirements

setup(
    name='awareness',
    version='1.0',
    description='The official implementation of the Awareness model.',
    author='Riccardo Emanuele Landi',
    author_email='riccardo.landi@innoida.it',
    packages=find_packages(),
    install_requires=[
        'torch==2.0.1'
    ]
)