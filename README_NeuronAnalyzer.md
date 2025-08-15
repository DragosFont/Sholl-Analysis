# 🔬 Neuron Analyzer - Exact Freehand ROI Analysis

An advanced neuron analysis tool for precise dendritic morphology analysis using freehand ROI selection and Sholl analysis. Supports multiple image formats including `.czi` files with Maximum Intensity Projection (MIP).

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [File Formats](#-file-formats)
- [Output](#-output)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🎯 **Exact Freehand ROI Selection**
- Draw precise freehand regions around neurons
- No rectangular bounding boxes - uses exact drawn shape
- Multiple ROI selection in single session
- Visual feedback with color-coded ROIs

### 🧬 **Advanced Channel Processing**
- **Green Channel**: Automatic neuron detection via threshold
- **Blue Channel**: Soma detection and localization
- **Red Channel**: Dendrite extraction and skeletonization
- **Smart Intersection**: Combines freehand selection with automatic detection

### 📊 **Comprehensive Analysis**
- **Sholl Analysis**: Quantitative dendritic complexity measurement
- **Morphological Metrics**: Peak intersections, radius, AUC
- **CSV Export**: Cumulative data storage with timestamp tracking
- **Debug Images**: Complete pipeline visualization

### 🔧 **Robust File Support**
- **CZI Files**: Native support with proper channel mapping
- **TIFF/TIF**: Standard microscopy formats
- **PNG/JPG**: General image formats
- **MIP Processing**: Automatic Maximum Intensity Projection for z-stacks

## 🚀 Installation

### Prerequisites

```bash
# Python 3.8+ required
python --version
```

### Dependencies

```bash
pip install numpy matplotlib scikit-image aicsimageio opencv-python pandas pillow scipy tkinter
```

### Optional Dependencies

For enhanced CZI support:
```bash
pip install aicsimageio[all]
```

### Clone Repository

```bash
git clone https://github.com/yourusername/neuron-analyzer.git
cd neuron-analyzer
```

## 🚀 Quick Start

### 1. GUI Mode (Recommended)

```bash
python neuron_analyzer.py
```

1. **Select Image**: Click "🔍 Alege Imaginea de Neuroni"
2. **Choose Output**: Set output directory (default: `outputs/`)
3. **Start Analysis**: Click "🎯 ÎNCEPE ANALIZA ROI EXACT + CSV APPEND"
4. **Draw ROIs**: Use freehand drawing to select neurons
5. **Finish**: Press ENTER or click "FINALIZEAZĂ"

### 2. Command Line Mode

```bash
python neuron_analyzer.py path/to/your/image.czi
```

### 3. Programmatic Usage

```python
from neuron_analyzer import CombinedNeuronAnalyzerFixed

# Initialize analyzer
analyzer = CombinedNeuronAnalyzerFixed("image.czi", "output_directory")

# Run analysis with callback
def results_callback(results):
    print(f"Analysis complete! {len(results)} ROIs processed")

analyzer.run_complete_analysis_main_thread_fixed(results_callback)
```

## 📖 Usage

### Workflow Overview

```
1. Image Loading → 2. ROI Selection → 3. Channel Processing → 4. Analysis → 5. Export
     (MIP)           (Freehand)         (G∩R masking)      (Sholl)     (CSV)
```

### Detailed Steps

#### 1. **Image Loading & MIP**
- Automatic format detection
- MIP generation for z-stacks
- Channel mapping (especially for .czi files)
- RGB visualization creation

#### 2. **Freehand ROI Selection**
- **Drawing**: Hold left mouse button and drag to draw
- **Complete ROI**: Right-click to finish current ROI
- **Multiple ROIs**: Draw as many as needed
- **Controls**: 
  - `ENTER`: Finish selection
  - `ESC`: Cancel
  - Buttons: Clear all, Undo last, Finalize

#### 3. **Channel Processing**

**Green Channel Masking:**
```
Green Channel → Gaussian Blur → Threshold → Clean → Green Mask
```

**Intersection Creation:**
```
Freehand Mask ∩ Green Mask = Final Analysis Mask
```

**Red Channel Processing:**
```
Red Channel → Denoise → Threshold → Skeletonize → Dendrites
(within Final Analysis Mask only)
```

**Blue Channel Processing:**
```
Blue Channel → Find Maximum → Soma Center
(within Final Analysis Mask only)
```

#### 4. **Sholl Analysis**
- Concentric circles from soma center
- Count dendrite intersections at each radius
- Calculate metrics: Peak, Radius at Peak, AUC, Slope

#### 5. **Data Export**
- **CSV Append Mode**: New data added to existing file
- **Image Refresh**: Old analysis images replaced
- **Debug Images**: Complete pipeline visualization

### ROI Selection Tips

#### ✅ **Good Practices**
- Draw ROIs **slightly larger** than the neuron
- Ensure **complete closure** of drawn regions
- **Avoid overlapping** ROIs
- Draw **smooth, continuous** lines

#### ❌ **Avoid**
- Very small ROIs (< 50 pixels)
- Incomplete/open contours
- Multiple disconnected regions in one ROI
- Drawing outside image boundaries

### Channel Mapping

#### CZI Files (Automatic Detection)
- **Channel 0**: Blue (DAPI/Hoechst) → Nuclei/Soma
- **Channel 1**: Green (FITC/GFP) → Neurons
- **Channel 2**: Red (Cy3/Texas Red) → Dendrites

#### Other Formats (Standard RGB)
- **R**: Red channel → Dendrites
- **G**: Green channel → Neurons  
- **B**: Blue channel → Soma

## 📁 File Formats

### Supported Formats

| Format | Extension | Support Level | Notes |
|--------|-----------|---------------|-------|
| CZI | `.czi` | ⭐⭐⭐⭐⭐ | Native support, automatic MIP |
| TIFF | `.tif`, `.tiff` | ⭐⭐⭐⭐⭐ | Standard microscopy format |
| PNG | `.png` | ⭐⭐⭐⭐ | RGB images |
| JPEG | `.jpg`, `.jpeg` | ⭐⭐⭐ | Compressed images |

### CZI Special Features
- **Automatic Channel Detection**: Reads metadata channel names
- **MIP Generation**: Maximum intensity projection for z-stacks
- **Proper Channel Mapping**: Matches channels to biological markers
- **Metadata Preservation**: Keeps original image information

## 📊 Output

### Directory Structure

```
outputs/
├── exact_freehand_sholl_results.csv     # Main results database
├── roi_1_exact_freehand/                # ROI 1 analysis
│   ├── debug/                           # Debug images
│   │   ├── 01_green_original.png
│   │   ├── 02_green_smooth.png
│   │   ├── 03_green_threshold.png
│   │   ├── 04_freehand_exact.png
│   │   ├── 05_intersection_direct_FINAL.png
│   │   └── 06_intersection_direct_only.png
│   ├── roi_rgb_exact_freehand.png       # RGB visualization
│   ├── roi_results_final_mask.png       # Final results
│   ├── roi_binary_final_mask.tif        # Binary mask for Sholl
│   └── sholl_analysis_exact_freehand.png # Sholl plot
├── roi_2_exact_freehand/                # ROI 2 analysis
│   └── ...
└── full_rgb.png                         # Original image RGB
```

### CSV Output Format

The main results are stored in `exact_freehand_sholl_results.csv`:

| Column | Description | Example |
|--------|-------------|---------|
| `timestamp` | Analysis timestamp | `2024-01-15_14-30-25_01` |
| `image_name` | Source image filename | `neuron_sample.czi` |
| `roi_index` | ROI number | `1` |
| `roi_type` | ROI selection method | `exact_freehand` |
| `roi_area_pixels` | ROI area in pixels | `156789` |
| `roi_perimeter_pixels` | ROI perimeter length | `2847.32` |
| `peak_number` | Peak intersection count | `45` |
| `radius_at_peak` | Radius at peak (μm) | `85` |
| `auc` | Area under curve | `1247.8` |
| `regression_coef` | Sholl decay slope | `-0.0234` |
| `total_intersections` | Sum of all intersections | `678` |
| `max_radius` | Maximum analysis radius | `150` |
| `mean_intersections` | Average intersections | `22.6` |
| `roi_folder` | Results directory | `roi_1_exact_freehand` |

### Debug Images Explained

| Image | Purpose | Shows |
|-------|---------|-------|
| `01_green_original.png` | Raw green channel | Original fluorescence |
| `02_green_smooth.png` | Smoothed green | After Gaussian filtering |
| `03_green_threshold.png` | Threshold mask | Detected neuron regions |
| `04_freehand_exact.png` | User selection | Your drawn ROI |
| `05_intersection_direct_FINAL.png` | Final mask | Freehand ∩ Green detection |
| `06_intersection_direct_only.png` | Process overview | Complete pipeline |

## 🔧 Technical Details

### Algorithm Pipeline

#### 1. **Image Preprocessing**
```python
# For CZI files
data = AICSImage(file_path).get_image_data("CZYX")
mip = data.max(axis=1)  # Maximum Intensity Projection

# Channel normalization
for channel in channels:
    normalized = exposure.rescale_intensity(channel, out_range=(0, 1))
```

#### 2. **Green Channel Masking**
```python
# Gaussian smoothing
green_smooth = filters.gaussian(green_norm, sigma=0.8)

# Threshold calculation (50th percentile)
threshold = np.percentile(green_smooth[green_smooth > 0], 50)

# Morphological cleaning
mask = morphology.remove_small_objects(binary_mask, min_size=30)
mask = morphology.binary_closing(mask, morphology.disk(2))
```

#### 3. **Exact Intersection**
```python
# Direct intersection (no complex strategies)
final_mask = green_threshold_mask & freehand_mask
```

#### 4. **Dendrite Extraction**
```python
# Processing within final mask only
red_in_mask = red_channel * final_mask.astype(float)

# Denoising pipeline
denoised = filters.gaussian(red_in_mask, sigma=0.5)
denoised = filters.median(denoised, morphology.disk(1))

# Conservative threshold (70th percentile)
threshold = np.percentile(denoised[denoised > 0], 70)

# Skeletonization
skeleton = morphology.skeletonize(binary_dendrites)
```

#### 5. **Sholl Analysis**
```python
# Concentric circles from soma center
for radius in range(step_size, max_radius, step_size):
    circle_mask = create_circle_mask(soma_center, radius)
    intersections = count_skeleton_intersections(skeleton, circle_mask)
    
# Calculate metrics
peak_number = max(intersection_counts)
radius_at_peak = radii[np.argmax(intersection_counts)]
auc = np.trapz(intersection_counts, radii)
```

### Performance Considerations

- **Memory Usage**: ~2-4GB RAM for large CZI files
- **Processing Time**: 30-60 seconds per ROI
- **Disk Space**: ~50-100MB per analysis (with debug images)

### Threading Architecture

- **Main Thread**: GUI and matplotlib (freehand selection)
- **Secondary Thread**: Image processing and analysis
- **Thread-Safe**: Proper cleanup and resource management

## 🛠 Troubleshooting

### Common Issues

#### **"Exception ignored in: <function Variable.__del__>"**
```
✅ FIXED: Use the latest version with proper tkinter cleanup
```

#### **"No ROIs selected"**
```
Check: 
- Draw complete, closed contours
- Right-click to finish each ROI
- Press ENTER to confirm selection
```

#### **"Green mask is empty"**
```
Solutions:
- Adjust green channel brightness/contrast
- Try different threshold percentile
- Ensure neurons are visible in green channel
```

#### **"Crazy skeleton results"**
```
✅ FIXED: Latest version has noise suppression
- Uses conservative thresholds
- Removes isolated artifacts
- Validates component quality
```

#### **"Results from other neurons"**
```
✅ FIXED: ROI isolation ensures proper channel extraction
- Each ROI processed independently
- Threshold calculated within ROI only
- Complete boundary checking
```

### Debug Mode

Enable detailed debugging:

```python
# In your analysis method
debug_mode = True  # Saves all intermediate steps
verbose_logging = True  # Detailed console output
```

### Log Analysis

Check console output for:
- ROI isolation verification
- Channel extraction statistics  
- Threshold values used
- Processing warnings

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/neuron-analyzer.git
cd neuron-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

### Code Style

- **PEP 8**: Python code formatting
- **Type hints**: For function signatures
- **Docstrings**: Google style documentation
- **Comments**: Explain complex algorithms

### Testing

```bash
# Run tests
python -m pytest tests/

# Test with sample data
python test_analysis.py --image sample_data/test_neuron.czi
```

### Submit Changes

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/neuron-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/neuron-analyzer/discussions)
- **Email**: your.email@institution.edu

## 🏆 Citation

If you use this software in your research, please cite:

```bibtex
@software{neuron_analyzer_2024,
  title = {Neuron Analyzer: Exact Freehand ROI Analysis Tool},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/neuron-analyzer},
  version = {1.0.0}
}
```

## 🙏 Acknowledgments

- **aicsimageio**: For excellent CZI file support
- **scikit-image**: For robust image processing algorithms
- **matplotlib**: For interactive ROI selection
- **NumPy/SciPy**: For mathematical operations

---

<div align="center">

**⭐ Star this repository if it helped your research! ⭐**

</div>