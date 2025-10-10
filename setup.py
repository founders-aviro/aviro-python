#!/usr/bin/env python3
"""Setup script for aviro package."""

from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from __init__.py
import re
with open("aviro/__init__.py", "r", encoding="utf-8") as fh:
    version = re.search(r'__version__ = ["\']([^"\']*)["\']', fh.read()).group(1)

setup(
    name="aviro",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for [describe your package's purpose here]",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aviro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Add your package dependencies here
        # "requests>=2.25.0",
        # "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
        ],
    },
)
