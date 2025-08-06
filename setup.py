import setuptools
from setuptools import setup, find_packages

setup(
    name="paulitools",
    version="2.0",
    author="Thomas Steckmann",
    author_email="tmsteckm@gmail.com",
    description="A collection of tools for dealing with Pauli strings. Includes Binary Symplectic manipulations, Clifford Operations, Diagonalization, and group operations",
    package_dir={"": "src"},  # Tell setuptools packages are under src/
    packages=find_packages(where="src"),  # Look for packages in src/
    install_requires=['numpy', 'numba'],
    python_requires=">=3.7",
)