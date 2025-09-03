import setuptools
from setuptools import setup, find_packages
setup(
    name="ptgalois",
    version="0.0.1",
    author="Thomas Steckmann",
    author_email="tmsteckm@gmail.com",
    description="A collection of tools for dealing with Pauli strings. Includes Binary Symplectic manipulations, Clifford Operations, Diagonalization, and group operations",
    packages=find_packages(),
    install_requires=['numpy', 'numba']
)

