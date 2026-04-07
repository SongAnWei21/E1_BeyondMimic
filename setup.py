import os
from setuptools import setup, find_packages

setup(
    name="e1_lab",
    packages=find_packages(),
    version="1.0.0",
    install_requires=[
        "mujoco==3.3.3",
        "mujoco-python-viewer",
        "psutil",
        "joblib>=1.2.0",
        "pynput",
    ],
)
