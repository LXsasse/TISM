from setuptools import find_packages, setup

setup(
    name='tism',
    version='0.0.1',
    author='Alexander Sasse',
    author_email='alexander.sasse@gmail.com',
    packages=find_packages(),
    license='LICENSE.txt',
    description='tism implements an approximation of ISM from gradient, connecting attribution values from gradients with those from ISM.',
    install_requires=[
        "numpy >= 1.14.2",
        "torch >= 1.9.0",
    ],
)
