# Computer Vision Homework 1

This project implements various computer vision algorithms for CIS 4930/5930 Computer Vision course.

## Completed Parts

### Part A: Image Responses with Filters ✅

Implements loading and processing of images with Leung-Malik filter bank:

- Loads 6 animal images (cardinal, leopard, panda)
- Applies 48 Leung-Malik filters to each image
- Generates visualizations showing filter responses
- Creates `same_animal_similar.png` and `different_animals_similar.png`

### Part B: Image Description with Texture ✅

Implements texture representation functions:

- `texture_repr()` - individual image texture features (96 features)
- `texture_repr_concat()` - concatenated representations (192 features)
- `texture_repr_mean()` - mean representations (96 features)
- Generates visualization plots and saves data to `.npz` file

### Part C: Canny Edge Detection ✅

Complete implementation of 5-step Canny edge detection algorithm:

- Gaussian smoothing with configurable sigma
- Gradient computation (magnitude and direction)
- Non-maximum suppression
- Double thresholding (low/high thresholds)
- Edge tracking by hysteresis
- Generates step-by-step visualizations for all images

### Part D: Harris Feature Point Detection ✅

Harris corner detection with orientation estimation:

- Computes Harris response using structure tensor
- Localizes keypoints with non-maximum suppression
- Estimates orientation using gradient information
- Generates visualizations with detected corners and orientations

### Part E: SIFT Descriptors and Bag-of-Words ✅

SIFT feature extraction and bag-of-words representation:

- Computes SIFT descriptors for detected keypoints
- Builds visual vocabulary using K-means clustering
- Creates bag-of-words histograms for image representation
- Includes comprehensive analysis and comparison

### Part F: Comparison Analysis ✅

Statistical comparison of different representation methods:

- Analyzes within-class vs between-class distances
- Compares texture representations vs SIFT bag-of-words
- Generates detailed performance metrics and visualizations
- Creates comprehensive analysis report

### Part G: computeBOWRepr Function ✅

Dedicated implementation of bag-of-words computation:

- Implements `computeBOWRepr(features, means)` function per specification
- Step-by-step implementation: initialization, distance computation, counting, normalization
- Validates against Part E results
- Generates individual BoW representations for all images

### Part H: Image Descriptions Comparison ✅

Quality assessment of different representation methods:

- Computes within-class and between-class distance analysis
- Calculates quality ratios for all three representation methods
- Generates Word document with detailed results
- Provides ranking of representation methods by discriminative power

## Setup Instructions

1. Clone this repository
2. Create virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note**: The `data/` folder is now included in the repository with all required images and filter files.

## Usage

### Run All Parts

To run the complete assignment (all parts A-H):

```bash
source .venv/bin/activate
python src/part_a.py --data-dir data --filters-mat data/filters.mat --out-dir out --size 100
python src/part_b.py --data-dir data --filters-mat data/filters.mat --size 100
python src/part_c.py --data-dir data --out-dir out --size 100
python src/part_d.py --data-dir data --out-dir out --size 100
python src/part_e.py --data-dir data --out-dir out --vocab-size 50
python src/part_f.py --out-dir out
python src/part_g.py --data-dir data --out-dir out --vocab-size 50
python src/part_h.py --data-dir data --out-dir out
```

### Individual Parts

#### Part A: Image Responses with Filters

```bash
source .venv/bin/activate
python src/part_a.py --data-dir data --filters-mat data/filters.mat --out-dir out --size 100
```

#### Part B: Image Description with Texture

```bash
source .venv/bin/activate
python src/part_b.py --data-dir data --filters-mat data/filters.mat --size 100
```

#### Part C: Canny Edge Detection

```bash
source .venv/bin/activate
python src/part_c.py --data-dir data --out-dir out --size 100
```

#### Part D: Harris Feature Points

```bash
source .venv/bin/activate
python src/part_d.py --data-dir data --out-dir out --size 100
```

#### Part E: SIFT Descriptors

```bash
source .venv/bin/activate
python src/part_e.py --data-dir data --out-dir out --vocab-size 50
```

#### Part F: Comparison Analysis

```bash
source .venv/bin/activate
python src/part_f.py --out-dir out
```

#### Part G: computeBOWRepr Function

```bash
source .venv/bin/activate
python src/part_g.py --data-dir data --out-dir out --vocab-size 50
```

#### Part H: Image Descriptions Comparison

```bash
source .venv/bin/activate
python src/part_h.py --data-dir data --out-dir out
```

## Outputs

- `out/filters/` - Filter response visualizations (48 PNG files)
- `out/part_b/` - Texture representation plots and data
- `out/part_c/` - Canny edge detection step-by-step results
- `out/part_d/` - Harris feature points with orientations
- `out/part_e/` - SIFT descriptors and bag-of-words analysis
- `out/part_f/` - Comprehensive comparison analysis and report
- `out/part_g/` - computeBOWRepr function results and validation
- `out/part_h/` - Image descriptions comparison with Word document
- `out/same_animal_similar.png` - Example filter showing similar responses for same animal
- `out/different_animals_similar.png` - Example filter showing distinct responses for different animals

## Project Structure

```
CV_Homework/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── utils.py          # Common utility functions
│   ├── part_a.py         # Image responses with filters
│   ├── part_b.py         # Texture representations
│   ├── part_c.py         # Canny edge detection
│   ├── part_d.py         # Harris feature points
│   ├── part_e.py         # SIFT descriptors and BoW
│   ├── part_f.py         # Comparison analysis
│   ├── part_g.py         # computeBOWRepr function
│   └── part_h.py         # Image descriptions comparison
├── data/                 # Input images and filter bank (now included)
│   ├── filters.mat       # Leung-Malik filter bank
│   ├── means.mat         # K-means cluster centers
│   ├── cardinal1.jpg     # Cardinal images
│   ├── cardinal2.jpg
│   ├── leopard1.jpg      # Leopard images
│   ├── leopard2.jpg
│   ├── panda1.jpg        # Panda images
│   ├── panda2.jpg
│   └── ...               # Additional test images
└── out/                  # Generated outputs (gitignored)
```

## GitHub Repository Setup

To push this project to GitHub:

1. Create a new repository on GitHub (e.g., `CV_Homework`)
2. Add remote and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/CV_Homework.git
git push -u origin main
```

## Requirements

- Python 3.8+
- numpy
- scipy
- Pillow
- matplotlib
- scikit-image
- scikit-learn
- python-docx (for Part H Word document generation)

## Notes

- Output files are excluded from git (.gitignored) due to size
- Virtual environment (.venv) is excluded from git
- Assignment PDF is excluded from git
- **Data folder is now included** in the repository for easy setup

## Assignment Summary

This project successfully implements all 8 parts of the Computer Vision homework:

1. **Parts A-B**: Texture analysis using Leung-Malik filters
2. **Parts C-D**: Classical edge and corner detection algorithms  
3. **Parts E-G**: Modern SIFT features and bag-of-words representation
4. **Parts F,H**: Comprehensive comparison and evaluation of methods

The implementation demonstrates proficiency in:
- Image filtering and convolution operations
- Classical computer vision algorithms (Canny, Harris)
- Modern feature descriptors (SIFT)
- Machine learning for computer vision (K-means, BoW)
- Statistical analysis and performance evaluation
