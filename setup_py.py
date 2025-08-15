#!/usr/bin/env python3
"""
Neuron Analyzer Setup Script
Advanced neuron analysis tool for precise dendritic morphology analysis
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    # Package Information
    name="neuron-analyzer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Advanced neuron analysis tool for precise dendritic morphology analysis using freehand ROI selection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neuron-analyzer",
    
    # Package Structure
    packages=find_packages(),
    include_package_data=True,
    
    # Requirements
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "czi": ["aicsimageio[all]>=4.9.0"],
        "full": ["aicsimageio[all]>=4.9.0", "seaborn>=0.11.0", "plotly>=5.0.0"]
    },
    
    # Python Version Support
    python_requires=">=3.8",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords
    keywords="neuron analysis morphology dendrites sholl roi microscopy imaging",
    
    # Entry Points
    entry_points={
        "console_scripts": [
            "neuron-analyzer=neuron_analyzer:main",
            "neuron-gui=neuron_analyzer:main_gui",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/neuron-analyzer/issues",
        "Source": "https://github.com/yourusername/neuron-analyzer",
        "Documentation": "https://github.com/yourusername/neuron-analyzer#readme",
    },
)