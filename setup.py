"""Setup script for Python Potter."""

from setuptools import setup, find_packages

setup(
    name="potter-fpga-router",
    version="1.0.0",
    description="Python implementation of Potter - A parallel overlap-tolerant FPGA router",
    author="Potter Team",
    packages=find_packages(),
    install_requires=[
        "psutil>=5.9.0",
        "pycapnp>=1.3.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "potter=main:main",
        ],
    },
)
