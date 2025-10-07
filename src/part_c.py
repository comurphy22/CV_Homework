#!/usr/bin/env python3
"""Part C: Canny Edge Detector

Usage example:
    python src/part_c.py --data-dir data --out-dir out --sigma 1.0 --low-thresh 0.1 --high-thresh 0.2
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from utils import load_and_preprocess_image, ensure_dir

EXPECTED_IMAGES = [
    ("cardinal1", "cardinal1.jpg"),
    ("cardinal2", "cardinal2.jpg"),
    ("leopard1", "leopard1.jpg"),
    ("leopard2", "leopard2.jpg"),
    ("panda1", "panda1.jpg"),
    ("panda2", "panda2.jpg"),
]


def gaussian_smooth(image, sigma):
    """Apply Gaussian smoothing to image."""
    return gaussian_filter(image, sigma=sigma)


def compute_gradients(image):
    """Compute gradient magnitude and direction using Sobel operators."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    
    grad_x = ndimage.convolve(image, sobel_x)
    grad_y = ndimage.convolve(image, sobel_y)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    
    return magnitude, direction, grad_x, grad_y


def non_maximum_suppression(magnitude, direction):
    """Apply non-maximum suppression to thin edges."""
    height, width = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    
    angle = np.rad2deg(direction) % 180
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 22.5 <= angle[i, j] < 67.5:
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            elif 67.5 <= angle[i, j] < 112.5:
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            
            if magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = magnitude[i, j]
    
    return suppressed


def double_threshold(image, low_thresh, high_thresh):
    """Apply double thresholding to classify edge pixels."""
    strong_edges = image > high_thresh
    weak_edges = (image >= low_thresh) & (image <= high_thresh)
    
    thresholded = np.zeros_like(image, dtype=np.uint8)
    thresholded[weak_edges] = 1
    thresholded[strong_edges] = 2
    
    return thresholded


def edge_tracking_by_hysteresis(thresholded):
    """Track edges by hysteresis - connect weak edges to strong edges."""
    height, width = thresholded.shape
    edges = np.zeros_like(thresholded, dtype=bool)
    
    strong_edges = (thresholded == 2)
    edges[strong_edges] = True
    
    changed = True
    while changed:
        changed = False
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if thresholded[i, j] == 1 and not edges[i, j]:
                    neighborhood = edges[i-1:i+2, j-1:j+2]
                    if np.any(neighborhood):
                        edges[i, j] = True
                        changed = True
    
    return edges.astype(np.uint8)


def canny_edge_detection(image, sigma=1.0, low_thresh=0.1, high_thresh=0.2):
    """Complete Canny edge detection pipeline."""
    smoothed = gaussian_smooth(image, sigma)
    
    magnitude, direction, grad_x, grad_y = compute_gradients(smoothed)
    
    suppressed = non_maximum_suppression(magnitude, direction)
    
    thresholded = double_threshold(suppressed, low_thresh, high_thresh)
    
    edges = edge_tracking_by_hysteresis(thresholded)
    
    return {
        'smoothed': smoothed,
        'magnitude': magnitude,
        'direction': direction,
        'grad_x': grad_x,
        'grad_y': grad_y,
        'suppressed': suppressed,
        'thresholded': thresholded,
        'edges': edges
    }


def visualize_canny_steps(image_name, original, results, save_dir):
    """Visualize all steps of Canny edge detection."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(results['smoothed'], cmap='gray')
    axes[1].set_title(f'Gaussian Smoothed')
    axes[1].axis('off')
    
    axes[2].imshow(results['magnitude'], cmap='gray')
    axes[2].set_title('Gradient Magnitude')
    axes[2].axis('off')
    
    axes[3].imshow(results['direction'], cmap='hsv')
    axes[3].set_title('Gradient Direction')
    axes[3].axis('off')
    
    axes[4].imshow(results['suppressed'], cmap='gray')
    axes[4].set_title('Non-Max Suppression')
    axes[4].axis('off')
    
    axes[5].imshow(results['thresholded'], cmap='gray')
    axes[5].set_title('Double Threshold')
    axes[5].axis('off')
    
    axes[6].imshow(results['edges'], cmap='gray')
    axes[6].set_title('Final Edges')
    axes[6].axis('off')
    
    axes[7].axis('off')
    
    plt.suptitle(f'Canny Edge Detection Steps - {image_name}')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{image_name}_canny_steps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Canny steps visualization: {save_path}")


def save_individual_outputs(image_name, results, save_dir):
    """Save individual gradient and edge images."""
    plt.imsave(os.path.join(save_dir, f'{image_name}_gradient_magnitude.png'), 
               results['magnitude'], cmap='gray')
    
    plt.imsave(os.path.join(save_dir, f'{image_name}_gradient_direction.png'), 
               results['direction'], cmap='hsv')
    
    plt.imsave(os.path.join(save_dir, f'{image_name}_edges.png'), 
               results['edges'], cmap='gray')
    
    print(f"Saved individual outputs for {image_name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='out')
    p.add_argument('--size', type=int, default=100)
    p.add_argument('--sigma', type=float, default=1.0, help='Gaussian smoothing sigma')
    p.add_argument('--low-thresh', type=float, default=0.1, help='Low threshold for double thresholding')
    p.add_argument('--high-thresh', type=float, default=0.2, help='High threshold for double thresholding')
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, 'part_c'))
    
    save_dir = os.path.join(out_dir, 'part_c')

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

    print(f"Processing {len(images)} images with Canny edge detection")
    print(f"Parameters: sigma={args.sigma}, low_thresh={args.low_thresh}, high_thresh={args.high_thresh}")

    for name, image in images.items():
        print(f"\nProcessing {name}...")
        
        results = canny_edge_detection(image, args.sigma, args.low_thresh, args.high_thresh)
        
        visualize_canny_steps(name, image, results, save_dir)
        
        save_individual_outputs(name, results, save_dir)

    print(f"\nCanny edge detection completed!")
    print(f"Results saved to: {save_dir}")
    print("Generated files:")
    print("  - *_canny_steps.png: Complete pipeline visualization")
    print("  - *_gradient_magnitude.png: Gradient magnitude images")
    print("  - *_gradient_direction.png: Gradient direction images") 
    print("  - *_edges.png: Final edge maps")


if __name__ == '__main__':
    main()