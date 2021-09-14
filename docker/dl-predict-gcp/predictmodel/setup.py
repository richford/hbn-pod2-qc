from setuptools import find_packages
from setuptools import setup

setup(
    name='predictmodel',
    description='Predict using a tensorflow model on GCP.',
    author="Adam Richie-Halford",
    author_email="richiehalford@gmail.com",
    install_requires=[
        'gcsfs==2021.7.0',
        'nobrainer @ git+https://github.com/richford/nobrainer.git@enh/four-d',
        'numpy',
        'pandas',
        'tensorflow',
    ],
    packages=find_packages()
)