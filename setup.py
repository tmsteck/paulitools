import setuptools
from setuptools import setup, find_packages

setup(
    name="paulitools",
    version="2.0",
    description="A Python package for Pauli string manipulation and analysis",
    author="Thomas Steckmann",
    author_email="tmsteckm@gmail.com",
    packages=["paulitools"],
    package_dir={"paulitools": "src"},
    install_requires=[
        "numpy",
        "numba",
        "galois",
        "joblib",
    ],
    python_requires=">=3.7",
)