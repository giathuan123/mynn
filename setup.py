from setuptools import setup, find_packages

setup(
    name="optensor",
    version="1.0",
    description="Autograd engine using Numpy for Neural Network",
    packages=find_packages(include="engine"),
    install_requires=["numpy"]
)
