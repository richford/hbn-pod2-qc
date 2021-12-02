from setuptools import find_packages
from setuptools import setup

setup(
    name="ig",
    description="Compute integrated gradients using a tensorflow model on GCP.",
    author="Adam Richie-Halford",
    author_email="richiehalford@gmail.com",
    install_requires=[
        "gcsfs==2021.7.0",
        "s3fs",
        "nilearn",
        "nibabel",
        "nobrainer @ git+https://github.com/neuronets/nobrainer.git",
        "numpy",
        "pandas",
        "tensorflow",
    ],
    packages=find_packages(),
)