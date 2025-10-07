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

## Setup Instructions

1. Clone this repository
2. Create virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Create a `data/` folder and add:
   - `filters.mat` (Leung-Malik filter bank)
   - `cardinal1.jpg`, `cardinal2.jpg`
   - `leopard1.jpg`, `leopard2.jpg`
   - `panda1.jpg`, `panda2.jpg`

## Usage

### Part A: Image Responses with Filters

```bash
source .venv/bin/activate
python src/part_a.py --data-dir data --filters-mat data/filters.mat --out-dir out --size 100
```

### Part B: Image Description with Texture

```bash
source .venv/bin/activate
python src/part_b.py --data-dir data --filters-mat data/filters.mat --size 100
```

## Outputs

- `out/filters/` - Filter response visualizations (48 PNG files)
- `out/part_b/` - Texture representation plots and data
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
│   └── part_b.py         # Texture representations
├── data/                 # Input images and filter bank (gitignored)
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

## Notes

- Data files and outputs are excluded from git (.gitignored)
- Virtual environment (.venv) is excluded from git
- Assignment PDF is excluded from git
