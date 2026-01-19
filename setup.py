from setuptools import setup, find_packages

setup(
    name="complex_deformations_analysis",
    version="0.4.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
