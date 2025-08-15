# 🔧 Installation Guide - Neuron Analyzer

Complete installation instructions for different user types and operating systems.

## 📋 Quick Installation

### For End Users (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/DragosFont/Sholl-Analysis
cd neuron-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python neuron_analyzer.py
```

### For Developers

```bash
# 1. Clone and setup
git clone https://github.com/DragosFont/Sholl-Analysis
cd neuron-analyzer

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install development dependencies
pip install -r requirements-dev.txt

# 4. Install in development mode
pip install -e .
```

## 🖥️ Operating System Specific

### Windows 10/11

#### Prerequisites
```powershell
# Install Python 3.8+ from Microsoft Store or python.org
python --version

# Install Git (optional, for cloning)
# Download from: https://git-scm.com/download/win
```

#### Installation
```powershell
# Option 1: Download ZIP
# Download from GitHub and extract

# Option 2: Clone with Git
git clone https://github.com/yourusername/neuron-analyzer.git
cd neuron-analyzer

# Install dependencies
pip install -r requirements.txt

# For enhanced CZI support
pip install aicsimageio[all]
```

#### Potential Windows Issues
```powershell
# If you get SSL certificate errors:
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# If tkinter is missing:
# Reinstall Python with "tcl/tk and IDLE" option checked
```

### macOS

#### Prerequisites
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.8+
brew install python@3.11

# Verify installation
python3 --version
```

#### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/neuron-analyzer.git
cd neuron-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For enhanced CZI support
pip install aicsimageio[all]
```

#### macOS Specific Notes
```bash
# If you get matplotlib backend issues:
pip install --upgrade matplotlib

# For M1/M2 Macs, you might need:
arch -arm64 pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install python3 python3-pip python3-venv python3-dev
sudo apt install build-essential libssl-dev libffi-dev

# Install GUI dependencies
sudo apt install python3-tk

# Optional: Install Git
sudo apt install git
```

#### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/neuron-analyzer.git
cd neuron-analyzer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# For enhanced CZI support
pip install aicsimageio[all]
```

#### Linux Specific Notes
```bash
# If you get OpenCV errors:
sudo apt install libopencv-dev python3-opencv

# If you get Qt/GUI errors:
sudo apt install python3-pyqt5-dev qttools5-dev-tools

# For headless servers (no GUI):
pip install matplotlib
# Then set backend in your code: matplotlib.use('Agg')
```

### CentOS/RHEL/Fedora

```bash
# Install prerequisites
sudo dnf install python3 python3-pip python3-devel
sudo dnf install gcc gcc-c++ make
sudo dnf install tkinter

# Clone and install
git clone https://github.com/yourusername/neuron-analyzer.git
cd neuron-analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🐍 Python Environment Management

### Using Conda (Recommended for Scientific Computing)

```bash
# Create conda environment
conda create -n neuron-analyzer python=3.9
conda activate neuron-analyzer

# Install from conda-forge where possible
conda install -c conda-forge numpy scipy matplotlib pandas pillow
conda install -c conda-forge scikit-image opencv

# Install remaining with pip
pip install aicsimageio
```

### Using Poetry (For Developers)

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone and setup
git clone https://github.com/yourusername/neuron-analyzer.git
cd neuron-analyzer

# Install dependencies
poetry install

# Activate environment
poetry shell

# Run application
python neuron_analyzer.py
```

### Using Pipenv

```bash
# Install Pipenv
pip install pipenv

# Clone repository
git clone https://github.com/yourusername/neuron-analyzer.git
cd neuron-analyzer

# Install dependencies
pipenv install -r requirements.txt

# Activate environment
pipenv shell
```

## 📦 Package Installation Options

### Option 1: Direct Installation from GitHub

```bash
# Install directly from GitHub
pip install git+https://github.com/yourusername/neuron-analyzer.git

# With enhanced CZI support
pip install "git+https://github.com/yourusername/neuron-analyzer.git[czi]"

# With all optional dependencies
pip install "git+https://github.com/yourusername/neuron-analyzer.git[full]"
```

### Option 2: Local Development Installation

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/neuron-analyzer.git
cd neuron-analyzer
pip install -e .

# Development mode with all extras
pip install -e ".[dev,czi,full]"
```

### Option 3: From PyPI (when published)

```bash
# Standard installation
pip install neuron-analyzer

# With enhanced CZI support
pip install "neuron-analyzer[czi]"

# With all features
pip install "neuron-analyzer[full]"
```

## 🧪 Verify Installation

### Basic Test

```python
# Test basic imports
python -c "
import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import pandas as pd
print('✅ Basic dependencies working!')
"
```

### CZI Support Test

```python
# Test CZI file support
python -c "
try:
    from aicsimageio import AICSImage
    print('✅ CZI support available!')
except ImportError:
    print('⚠️ CZI support not installed. Run: pip install aicsimageio[all]')
"
```

### GUI Test

```python
# Test GUI components
python -c "
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
print('✅ GUI components working!')
"
```

### Full Application Test

```bash
# Run with test mode (if implemented)
python neuron_analyzer.py --test

# Or run with sample data
python neuron_analyzer.py sample_data/test_image.tif
```

## 🐳 Docker Installation (Advanced)

### Dockerfile

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-tk \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install application
RUN pip install -e .

# Expose port for GUI (if using X11 forwarding)
EXPOSE 6000

# Set entry point
ENTRYPOINT ["python", "neuron_analyzer.py"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  neuron-analyzer:
    build: .
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - /tmp/.X11-unix:/tmp/.X11-unix:rw
    environment:
      - DISPLAY=${DISPLAY}
    network_mode: host
```

### Running with Docker

```bash
# Build image
docker build -t neuron-analyzer .

# Run with X11 forwarding (Linux)
xhost +local:docker
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  neuron-analyzer

# Run on Windows (with VcXsrv or similar)
docker run -it --rm \
  -v "%cd%/data:/app/data" \
  -v "%cd%/outputs:/app/outputs" \
  -e DISPLAY=host.docker.internal:0.0 \
  neuron-analyzer
```

## 🚨 Troubleshooting

### Common Issues

#### **ImportError: No module named 'tkinter'**
```bash
# Ubuntu/Debian
sudo apt install python3-tk

# CentOS/RHEL
sudo dnf install tkinter

# macOS
brew install python-tk

# Windows
# Reinstall Python with "tcl/tk and IDLE" option
```

#### **OpenCV Import Error**
```bash
# Try different OpenCV packages
pip uninstall opencv-python
pip install opencv-python-headless

# Or on Linux
sudo apt install python3-opencv
```

#### **Matplotlib Backend Issues**
```python
# Add to your script
import matplotlib
matplotlib.use('TkAgg')  # For GUI
# or
matplotlib.use('Agg')    # For headless
```

#### **Memory Issues with Large Images**
```bash
# Increase available memory or use 64-bit Python
# Monitor memory usage with:
pip install memory-profiler
python -m memory_profiler neuron_analyzer.py
```

#### **Permission Errors**
```bash
# Don't use sudo with pip, use virtual environments instead
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Performance Optimization

```bash
# Install optimized BLAS libraries
pip install numpy[fast]

# Use conda for better performance
conda install -c conda-forge numpy scipy scikit-image

# For large files, consider installing with Intel MKL
conda install mkl mkl-service
```

## 📊 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended for large images)
- **Storage**: 1GB free space
- **Display**: 1024x768 resolution

### Recommended Requirements
- **OS**: Latest stable versions
- **Python**: 3.9 or 3.10
- **RAM**: 16GB for processing large CZI files
- **Storage**: 10GB for sample data and outputs
- **Display**: 1920x1080 or higher
- **GPU**: Not required, but helps with large image processing

### Supported File Formats

| Format | Support Level | Requirements |
|--------|---------------|--------------|
| CZI | Full | `aicsimageio[all]` |
| TIFF | Full | `scikit-image` |
| PNG | Full | `Pillow` |
| JPEG | Basic | `Pillow` |

---

## 🆘 Getting Help

If you encounter issues:

1. **Check this installation guide** for your specific OS
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce
   - Output of `pip list`

**Installation successful? Star the repository! ⭐**