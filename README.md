Part A: Image Responses with Filters

This small project implements Part A of the homework: load a Leung-Malik filter bank (MAT file), resize and grayscale images, convolve each image with all filters, and save visualizations per filter.

How to run

1. Create a folder `data/` inside the repository root and place the following files there:

   - `filters.mat` (the Leung-Malik filter bank)
   - `cardinal1.jpg`, `cardinal2.jpg`, `leopard1.jpg`, `leopard2.jpg`, `panda1.jpg`, `panda2.jpg`

2. (Optional) create a Python virtualenv and install requirements:

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

3. Run the Part A script (example):

   python src/part_a.py --data-dir data --filters-mat data/filters.mat --out-dir out --size 100

Outputs

- Per-filter visualization images will be written to `out/filters/` (one PNG per filter).
- If you pass `--same-filter` or `--diff-filter` indices, two additional PNGs will be written to `out/` with those names.

Notes

- The script tries multiple possible variable names when reading the .mat file; if it cannot find a filter bank variable, open the .mat in MATLAB or use `scipy.io.loadmat` in a quick REPL to inspect variable names.
