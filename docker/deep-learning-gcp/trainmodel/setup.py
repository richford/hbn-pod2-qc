from setuptools import find_packages
from setuptools import setup

setup(
    name='trainmodel',
    description='Train a tensorflow model on cloud TPU.',
    author="Adam Richie-Halford",
    author_email="richiehalford@gmail.com",
    install_requires=[
        'gcsfs==2021.7.0',
        'nobrainer @ git+https://github.com/richford/nobrainer.git@enh/four-d',
        'numpy',
        'tensorflow',
    ],
    packages=find_packages()
)