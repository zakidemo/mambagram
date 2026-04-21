"""Setup script for the mambagram package."""
from setuptools import setup, find_packages

setup(
    name="mambagram",
    version="0.0.2",
    description="Adaptive Time-Frequency Representations via Selective State Space Models",
    author="zakineli",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)