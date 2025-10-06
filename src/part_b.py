#!/usr/bin/env python3
"""Part B: Image Description with Texture

Usage example:
    python src/part_b.py --data-dir data --filters-mat data/filters.mat --size 100
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from utils import (
    load_and_preprocess_image, ensure_dir, load_filters,
    compute_filter_responses, texture_repr, texture_repr_concat, texture_repr_mean
)

EXPECTED_IMAGES = [
    ("cardinal1", "cardinal1.jpg"),
    ("cardinal2", "cardinal2.jpg"),
    ("leopard1", "leopard1.jpg"),
    ("leopard2", "leopard2.jpg"),
    ("panda1", "panda1.jpg"),
    ("panda2", "panda2.jpg"),
]


def visualize_representation(repr_vec, title, save_path=None):
    """Visualize a texture representation vector."""
    plt.figure(figsize=(12, 4))
    plt.plot(repr_vec, 'b-', linewidth=1)
    plt.title(f'{title} (dim={len(repr_vec)})')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    plt.close()  # Close the figure to free memory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--filters-mat', default='data/filters.mat')
    p.add_argument('--out-dir', default='out')
    p.add_argument('--size', type=int, default=100)
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, 'part_b'))

    # Load images
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

    # Load filters
    filters = load_filters(args.filters_mat)
    print(f"Loaded {len(filters)} filters")

    # Group images by category
    cardinals = [images[k] for k in ["cardinal1", "cardinal2"] if k in images]
    leopards = [images[k] for k in ["leopard1", "leopard2"] if k in images]
    pandas = [images[k] for k in ["panda1", "panda2"] if k in images]

    print(f"Cardinal images: {len(cardinals)}")
    print(f"Leopard images: {len(leopards)}")
    print(f"Panda images: {len(pandas)}")

    # Compute individual texture representations
    print("\n=== Individual Texture Representations ===")
    individual_reprs = {}
    for name, image in images.items():
        responses = compute_filter_responses(image, filters)
        repr_vec = texture_repr(responses)
        individual_reprs[name] = repr_vec
        print(f"{name}: {len(repr_vec)} features")

    # Compute concatenated representations
    print("\n=== Concatenated Texture Representations ===")
    cardinal_concat = texture_repr_concat(cardinals, filters) if cardinals else np.array([])
    leopard_concat = texture_repr_concat(leopards, filters) if leopards else np.array([])
    panda_concat = texture_repr_concat(pandas, filters) if pandas else np.array([])

    print(f"Cardinal concatenated: {len(cardinal_concat)} features")
    print(f"Leopard concatenated: {len(leopard_concat)} features")
    print(f"Panda concatenated: {len(panda_concat)} features")

    # Compute mean representations
    print("\n=== Mean Texture Representations ===")
    cardinal_mean = texture_repr_mean(cardinals, filters) if cardinals else np.array([])
    leopard_mean = texture_repr_mean(leopards, filters) if leopards else np.array([])
    panda_mean = texture_repr_mean(pandas, filters) if pandas else np.array([])

    print(f"Cardinal mean: {len(cardinal_mean)} features")
    print(f"Leopard mean: {len(leopard_mean)} features")
    print(f"Panda mean: {len(panda_mean)} features")

    # Visualize some representations
    out_b = os.path.join(out_dir, 'part_b')
    
    # Plot individual representations for first image of each category
    if 'cardinal1' in individual_reprs:
        visualize_representation(individual_reprs['cardinal1'], 'Cardinal1 Individual Representation',
                               os.path.join(out_b, 'cardinal1_individual.png'))
    
    if 'leopard1' in individual_reprs:
        visualize_representation(individual_reprs['leopard1'], 'Leopard1 Individual Representation',
                               os.path.join(out_b, 'leopard1_individual.png'))
    
    if 'panda1' in individual_reprs:
        visualize_representation(individual_reprs['panda1'], 'Panda1 Individual Representation',
                               os.path.join(out_b, 'panda1_individual.png'))

    # Plot concatenated representations
    if len(cardinal_concat) > 0:
        visualize_representation(cardinal_concat, 'Cardinal Concatenated Representation',
                               os.path.join(out_b, 'cardinal_concat.png'))
    
    if len(leopard_concat) > 0:
        visualize_representation(leopard_concat, 'Leopard Concatenated Representation',
                               os.path.join(out_b, 'leopard_concat.png'))
    
    if len(panda_concat) > 0:
        visualize_representation(panda_concat, 'Panda Concatenated Representation',
                               os.path.join(out_b, 'panda_concat.png'))

    # Plot mean representations
    if len(cardinal_mean) > 0:
        visualize_representation(cardinal_mean, 'Cardinal Mean Representation',
                               os.path.join(out_b, 'cardinal_mean.png'))
    
    if len(leopard_mean) > 0:
        visualize_representation(leopard_mean, 'Leopard Mean Representation',
                               os.path.join(out_b, 'leopard_mean.png'))
    
    if len(panda_mean) > 0:
        visualize_representation(panda_mean, 'Panda Mean Representation',
                               os.path.join(out_b, 'panda_mean.png'))

    # Save representations to files
    np.savez(os.path.join(out_b, 'texture_representations.npz'),
             cardinal1=individual_reprs.get('cardinal1', np.array([])),
             cardinal2=individual_reprs.get('cardinal2', np.array([])),
             leopard1=individual_reprs.get('leopard1', np.array([])),
             leopard2=individual_reprs.get('leopard2', np.array([])),
             panda1=individual_reprs.get('panda1', np.array([])),
             panda2=individual_reprs.get('panda2', np.array([])),
             cardinal_concat=cardinal_concat,
             leopard_concat=leopard_concat,
             panda_concat=panda_concat,
             cardinal_mean=cardinal_mean,
             leopard_mean=leopard_mean,
             panda_mean=panda_mean)

    print(f"\nTexture representations saved to {os.path.join(out_b, 'texture_representations.npz')}")
    print(f"Visualization plots saved to {out_b}/")

    # Compare representations - compute some basic statistics
    print("\n=== Representation Statistics ===")
    if len(individual_reprs) > 0:
        first_repr = list(individual_reprs.values())[0]
        print(f"Individual representation dimension: {len(first_repr)}")
        print(f"Individual representation range: [{np.min(first_repr):.4f}, {np.max(first_repr):.4f}]")
    
    if len(cardinal_concat) > 0:
        print(f"Concatenated representation dimension: {len(cardinal_concat)}")
        print(f"Concatenated representation range: [{np.min(cardinal_concat):.4f}, {np.max(cardinal_concat):.4f}]")
    
    if len(cardinal_mean) > 0:
        print(f"Mean representation dimension: {len(cardinal_mean)}")
        print(f"Mean representation range: [{np.min(cardinal_mean):.4f}, {np.max(cardinal_mean):.4f}]")


if __name__ == '__main__':
    main()