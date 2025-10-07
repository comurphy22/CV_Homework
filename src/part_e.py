#!/usr/bin/env python3
"""Part E: Feature Description and SIFT Bag-of-Words

Usage example:
    python src/part_e.py --data-dir data --out-dir out --size 100 --vocab-size 50 --patch-size 16
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import pickle

from utils import load_and_preprocess_image, ensure_dir

EXPECTED_IMAGES = [
    ("cardinal1", "cardinal1.jpg"),
    ("cardinal2", "cardinal2.jpg"),
    ("leopard1", "leopard1.jpg"),
    ("leopard2", "leopard2.jpg"),
    ("panda1", "panda1.jpg"),
    ("panda2", "panda2.jpg"),
]


def load_harris_keypoints(keypoints_file):
    """Load Harris keypoints from text file."""
    if not os.path.exists(keypoints_file):
        return []
    
    data = np.loadtxt(keypoints_file)
    if data.size == 0:
        return []
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    keypoints = []
    for row in data:
        x, y, score, orientation = row
        keypoints.append({
            'x': int(x), 'y': int(y), 
            'score': score, 'orientation': orientation
        })
    return keypoints


def extract_patch(image, x, y, patch_size, orientation=0):
    """Extract and normalize a patch around a keypoint.
    
    Args:
        image: Input grayscale image
        x, y: Keypoint coordinates
        patch_size: Size of patch to extract
        orientation: Keypoint orientation for rotation normalization
    
    Returns:
        patch: Normalized patch
    """
    half_size = patch_size // 2
    height, width = image.shape
    
    patch_coords = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
    
    cos_o = np.cos(-orientation)
    sin_o = np.sin(-orientation)
    
    rotated_x = cos_o * patch_coords[1] - sin_o * patch_coords[0] + x
    rotated_y = sin_o * patch_coords[1] + cos_o * patch_coords[0] + y
    
    patch = ndimage.map_coordinates(image, [rotated_y, rotated_x], 
                                   order=1, mode='constant', cval=0)
    
    if patch.std() > 1e-10:
        patch = (patch - patch.mean()) / patch.std()
    else:
        patch = patch - patch.mean()
    
    return patch


def compute_sift_descriptor(patch, num_bins=8, grid_size=4):
    """Compute SIFT-like descriptor from normalized patch.
    
    Args:
        patch: Normalized image patch
        num_bins: Number of orientation bins
        grid_size: Size of spatial grid (4x4 = 16 subregions)
    
    Returns:
        descriptor: SIFT-like feature vector (128D for 4x4x8)
    """
    patch_size = patch.shape[0]
    cell_size = patch_size // grid_size
    
    grad_x = ndimage.convolve1d(patch, [-1, 0, 1], axis=1)
    grad_y = ndimage.convolve1d(patch, [-1, 0, 1], axis=0)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x)
    
    descriptor = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * cell_size
            y_end = min((i + 1) * cell_size, patch_size)
            x_start = j * cell_size
            x_end = min((j + 1) * cell_size, patch_size)
            
            cell_mag = magnitude[y_start:y_end, x_start:x_end]
            cell_ori = orientation[y_start:y_end, x_start:x_end]
            
            hist, _ = np.histogram(cell_ori, bins=num_bins, 
                                 range=(-np.pi, np.pi), weights=cell_mag)
            descriptor.extend(hist)
    
    descriptor = np.array(descriptor)
    
    if np.linalg.norm(descriptor) > 1e-10:
        descriptor = descriptor / np.linalg.norm(descriptor)
        descriptor = np.clip(descriptor, 0, 0.2)
        if np.linalg.norm(descriptor) > 1e-10:
            descriptor = descriptor / np.linalg.norm(descriptor)
    
    return descriptor


def extract_descriptors_from_image(image, keypoints, patch_size=16):
    """Extract SIFT-like descriptors from all keypoints in an image."""
    descriptors = []
    valid_keypoints = []
    
    for kpt in keypoints:
        x, y = kpt['x'], kpt['y']
        orientation = kpt['orientation']
        
        half_size = patch_size // 2
        if (x - half_size >= 0 and x + half_size < image.shape[1] and
            y - half_size >= 0 and y + half_size < image.shape[0]):
            
            patch = extract_patch(image, x, y, patch_size, orientation)
            descriptor = compute_sift_descriptor(patch)
            descriptors.append(descriptor)
            valid_keypoints.append(kpt)
    
    return np.array(descriptors), valid_keypoints


def build_visual_vocabulary(all_descriptors, vocab_size=50):
    """Build visual vocabulary using K-means clustering.
    
    Args:
        all_descriptors: List of descriptor arrays from all images
        vocab_size: Number of visual words
    
    Returns:
        kmeans: Trained K-means model
        vocabulary: Cluster centers (visual words)
    """
    if len(all_descriptors) == 0:
        raise ValueError("No descriptors provided for vocabulary building")
    
    concatenated = np.vstack(all_descriptors)
    print(f"Building vocabulary from {len(concatenated)} descriptors")
    
    kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init=10)
    kmeans.fit(concatenated)
    
    vocabulary = kmeans.cluster_centers_
    return kmeans, vocabulary


def compute_bow_histogram(descriptors, kmeans, vocab_size):
    """Compute bag-of-words histogram for an image.
    
    Args:
        descriptors: SIFT descriptors for the image
        kmeans: Trained K-means model
        vocab_size: Size of vocabulary
    
    Returns:
        histogram: Normalized bag-of-words histogram
    """
    if len(descriptors) == 0:
        return np.zeros(vocab_size)
    
    word_assignments = kmeans.predict(descriptors)
    
    histogram, _ = np.histogram(word_assignments, bins=vocab_size, range=(0, vocab_size))
    
    if histogram.sum() > 0:
        histogram = histogram.astype(float) / histogram.sum()
    
    return histogram


def visualize_bow_histograms(histograms, image_names, save_path):
    """Visualize bag-of-words histograms for all images."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    
    for i, (name, hist) in enumerate(zip(image_names, histograms)):
        axes[i].bar(range(len(hist)), hist)
        axes[i].set_title(f'{name} BoW Histogram')
        axes[i].set_xlabel('Visual Word')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved BoW histograms visualization: {save_path}")


def compute_distance_matrix(histograms, metric='euclidean'):
    """Compute pairwise distance matrix between BoW histograms."""
    return pairwise_distances(histograms, metric=metric)


def visualize_distance_matrix(distance_matrix, image_names, save_path):
    """Visualize the distance matrix between images."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(distance_matrix, cmap='viridis')
    ax.set_xticks(range(len(image_names)))
    ax.set_yticks(range(len(image_names)))
    ax.set_xticklabels(image_names, rotation=45)
    ax.set_yticklabels(image_names)
    
    for i in range(len(image_names)):
        for j in range(len(image_names)):
            ax.text(j, i, f'{distance_matrix[i, j]:.3f}', 
                   ha="center", va="center", color="white")
    
    plt.colorbar(im, ax=ax, label='Distance')
    plt.title('BoW Histogram Distance Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distance matrix visualization: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='out')
    p.add_argument('--size', type=int, default=100)
    p.add_argument('--vocab-size', type=int, default=50, help='Size of visual vocabulary')
    p.add_argument('--patch-size', type=int, default=16, help='Size of patches for SIFT descriptors')
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, 'part_e'))
    
    save_dir = os.path.join(out_dir, 'part_e')
    keypoints_dir = os.path.join(out_dir, 'part_d')

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

    print(f"Processing {len(images)} images for SIFT BoW representation")
    print(f"Parameters: vocab_size={args.vocab_size}, patch_size={args.patch_size}")

    all_descriptors = []
    image_descriptors = {}
    image_names = []
    
    for name, image in images.items():
        print(f"\nProcessing {name}...")
        
        keypoints_file = os.path.join(keypoints_dir, f'{name}_harris_keypoints.txt')
        keypoints = load_harris_keypoints(keypoints_file)
        print(f"  Loaded {len(keypoints)} Harris keypoints")
        
        descriptors, valid_keypoints = extract_descriptors_from_image(
            image, keypoints, args.patch_size)
        
        print(f"  Extracted {len(descriptors)} SIFT descriptors")
        if len(descriptors) > 0:
            print(f"  Descriptor dimension: {descriptors.shape[1]}")
        
        image_descriptors[name] = descriptors
        if len(descriptors) > 0:
            all_descriptors.append(descriptors)
        image_names.append(name)

    if len(all_descriptors) == 0:
        print("No descriptors extracted from any image. Exiting.")
        return

    print(f"\nBuilding visual vocabulary with {args.vocab_size} words...")
    kmeans, vocabulary = build_visual_vocabulary(all_descriptors, args.vocab_size)
    
    vocab_path = os.path.join(save_dir, 'visual_vocabulary.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump({'kmeans': kmeans, 'vocabulary': vocabulary}, f)
    print(f"Saved visual vocabulary to: {vocab_path}")

    print(f"\nComputing bag-of-words histograms...")
    bow_histograms = []
    
    for name in image_names:
        descriptors = image_descriptors[name]
        histogram = compute_bow_histogram(descriptors, kmeans, args.vocab_size)
        bow_histograms.append(histogram)
        
        print(f"  {name}: {len(descriptors)} descriptors -> BoW histogram (sum={histogram.sum():.3f})")
        
        np.savetxt(os.path.join(save_dir, f'{name}_bow_histogram.txt'), 
                  histogram, fmt='%.6f', header=f'BoW histogram for {name}')

    bow_histograms = np.array(bow_histograms)

    visualize_bow_histograms(bow_histograms, image_names,
                           os.path.join(save_dir, 'bow_histograms.png'))

    print(f"\nComputing distance matrix...")
    distance_matrix = compute_distance_matrix(bow_histograms, metric='euclidean')
    
    visualize_distance_matrix(distance_matrix, image_names,
                            os.path.join(save_dir, 'bow_distance_matrix.png'))

    np.savetxt(os.path.join(save_dir, 'bow_distance_matrix.txt'), 
              distance_matrix, fmt='%.6f', 
              header='BoW distance matrix (rows/cols: ' + ' '.join(image_names) + ')')

    np.savez(os.path.join(save_dir, 'bow_histograms.npz'),
             histograms=bow_histograms, image_names=image_names,
             distance_matrix=distance_matrix)

    print(f"\n=== SIFT Bag-of-Words Analysis ===")
    print(f"Visual vocabulary size: {args.vocab_size}")
    print(f"Total descriptors used: {sum(len(desc) for desc in all_descriptors)}")
    
    cardinal_indices = [i for i, name in enumerate(image_names) if 'cardinal' in name]
    leopard_indices = [i for i, name in enumerate(image_names) if 'leopard' in name]
    panda_indices = [i for i, name in enumerate(image_names) if 'panda' in name]
    
    within_class_distances = []
    between_class_distances = []
    
    for indices in [cardinal_indices, leopard_indices, panda_indices]:
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    within_class_distances.append(distance_matrix[indices[i], indices[j]])
    
    all_indices = list(range(len(image_names)))
    for i in range(len(all_indices)):
        for j in range(i+1, len(all_indices)):
            name_i, name_j = image_names[i], image_names[j]
            if not ((('cardinal' in name_i and 'cardinal' in name_j) or
                    ('leopard' in name_i and 'leopard' in name_j) or
                    ('panda' in name_i and 'panda' in name_j))):
                between_class_distances.append(distance_matrix[i, j])
    
    if within_class_distances and between_class_distances:
        avg_within = np.mean(within_class_distances)
        avg_between = np.mean(between_class_distances)
        ratio = avg_between / avg_within if avg_within > 0 else float('inf')
        
        print(f"Average within-class distance: {avg_within:.6f}")
        print(f"Average between-class distance: {avg_between:.6f}")
        print(f"Between/within ratio: {ratio:.6f}")

    print(f"\nSIFT Bag-of-Words completed!")
    print(f"Results saved to: {save_dir}")
    print("Generated files:")
    print("  - visual_vocabulary.pkl: Trained K-means model and vocabulary")
    print("  - *_bow_histogram.txt: Individual BoW histograms")
    print("  - bow_histograms.png: Histogram visualizations")
    print("  - bow_distance_matrix.png: Distance matrix visualization")
    print("  - bow_histograms.npz: All data in NumPy format")


if __name__ == '__main__':
    main()