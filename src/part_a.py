#!/usr/bin/env python3
"""Part A: Image responses with filters

Usage example:
    python src/part_a.py --data-dir data --filters-mat data/filters.mat --out-dir out --size 100
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal

from utils import load_and_preprocess_image, ensure_dir

EXPECTED_IMAGES = [
    ("cardinal1", "cardinal1.jpg"),
    ("cardinal2", "cardinal2.jpg"),
    ("leopard1", "leopard1.jpg"),
    ("leopard2", "leopard2.jpg"),
    ("panda1", "panda1.jpg"),
    ("panda2", "panda2.jpg"),
]


def find_filterbank(mat):
    for k in ("filterBank", "filters", "F", "LMfilters", "bank", "filter_bank", "fb"):
        if k in mat:
            return mat[k]
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim in (3, 4):
            if 48 in v.shape:
                return v
    return None


def interpret_filters_array(arr):
    if arr is None:
        return []
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.ndim == 3:
        h, w, n = arr.shape
        if n == 48:
            return [arr[:, :, i] for i in range(n)]
        if h == 48:
            return [arr[i, :, :] for i in range(h)]
    if arr.ndim == 4:
        s = arr.shape
        arr_s = np.squeeze(arr)
        return interpret_filters_array(arr_s)
    flat = arr.ravel()
    if flat.size % 48 == 0:
        per = flat.size // 48
        side = int(np.sqrt(per))
        if side * side == per:
            resh = flat.reshape((side, side, 48))
            return [resh[:, :, i] for i in range(48)]
    raise ValueError("Could not interpret filter bank shape: {}".format(arr.shape))


def normalize(x):
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def visualize_filter(filter_kernel, responses_dict, out_path, idx):
    fig, axes = plt.subplots(4, 2, figsize=(6, 12))
    axes = axes.reshape(-1)

    axes[0].imshow(normalize(filter_kernel), cmap='gray')
    axes[0].set_title(f"Filter {idx}")
    axes[1].axis('off')

    order = ["cardinal1", "cardinal2", "leopard1", "leopard2", "panda1", "panda2"]
    for i, name in enumerate(order):
        ax = axes[2 + i]
        if name in responses_dict:
            ax.imshow(normalize(responses_dict[name]), cmap='jet')
            ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--filters-mat', default='data/filters.mat')
    p.add_argument('--out-dir', default='out')
    p.add_argument('--size', type=int, default=100)
    p.add_argument('--same-filter', type=int, default=None, help='Index of filter to save also as same_animal_similar.png')
    p.add_argument('--diff-filter', type=int, default=None, help='Index of filter to save also as different_animals_similar.png')
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, 'filters'))

    images = {}
    for name, fname in EXPECTED_IMAGES:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"Warning: expected image not found: {path}. Skipping {name}.")
            continue
        images[name] = load_and_preprocess_image(path, args.size)

    if len(images) == 0:
        print("No images found in data dir. Exiting.")
        return

    if not os.path.exists(args.filters_mat):
        raise FileNotFoundError(f"filters.mat not found at {args.filters_mat}")
    mat = loadmat(args.filters_mat)
    fb = find_filterbank(mat)
    if fb is None:
        raise ValueError("Could not find a filter bank variable in the .mat file. Inspect the file with scipy.io.loadmat and provide the right variable name.")
    filters = interpret_filters_array(fb)
    print(f"Found {len(filters)} filters")

    for i, filt in enumerate(filters):
        filt = np.squeeze(np.array(filt, dtype=float))
        responses = {}
        for k, img in images.items():
            responses[k] = signal.convolve2d(img, filt, mode='same', boundary='symm')
        out_path = os.path.join(out_dir, 'filters', f'filter_{i+1:02d}.png')
        visualize_filter(filt, responses, out_path, i+1)

        if args.same_filter is not None and i == args.same_filter:
            plt.imsave(os.path.join(out_dir, 'same_animal_similar.png'), normalize(responses.get('cardinal1', np.zeros_like(next(iter(images.values()))))), cmap='jet')
        if args.diff_filter is not None and i == args.diff_filter:
            plt.imsave(os.path.join(out_dir, 'different_animals_similar.png'), normalize(responses.get('panda1', np.zeros_like(next(iter(images.values()))))), cmap='jet')

    print(f"Wrote filter visualizations to {os.path.join(out_dir, 'filters')}")


if __name__ == '__main__':
    main()
