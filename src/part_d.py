#!/usr/bin/env python3
"""Part D: Harris Feature Detection

Usage example:
    python src/part_d.py --data-dir data --out-dir out --sigma 1.0 --k 0.04 --threshold 0.01 --max-features 100
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from utils import load_and_preprocess_image, ensure_dir

EXPECTED_IMAGES = [
    ("cardinal1", "cardinal1.jpg"),
    ("cardinal2", "cardinal2.jpg"),
    ("leopard1", "leopard1.jpg"),
    ("leopard2", "leopard2.jpg"),
    ("panda1", "panda1.jpg"),
    ("panda2", "panda2.jpg"),
]


def compute_harris_response(image, sigma=1.0, k=0.04):
    """Compute Harris corner response function.
    
    Args:
        image: Input grayscale image
        sigma: Gaussian window parameter
        k: Harris detector free parameter (typically 0.04-0.06)
    
    Returns:
        R: Harris response map
        grad_x, grad_y: Image gradients
    """
    # Compute image gradients using Sobel operators
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    
    grad_x = ndimage.convolve(image, sobel_x)
    grad_y = ndimage.convolve(image, sobel_y)
    
    # Compute components of the structure tensor
    Ixx = grad_x * grad_x
    Iyy = grad_y * grad_y
    Ixy = grad_x * grad_y
    
    # Apply Gaussian window
    Sxx = gaussian_filter(Ixx, sigma=sigma)
    Syy = gaussian_filter(Iyy, sigma=sigma)
    Sxy = gaussian_filter(Ixy, sigma=sigma)
    
    # Compute Harris response: R = det(M) - k * trace(M)^2
    # where M is the structure tensor [[Sxx, Sxy], [Sxy, Syy]]
    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M ** 2)
    
    return R, grad_x, grad_y


def find_harris_keypoints(R, threshold=0.01, min_distance=5, max_features=100):
    """Find Harris keypoints from response map.
    
    Args:
        R: Harris response map
        threshold: Minimum response value for keypoint detection
        min_distance: Minimum distance between keypoints
        max_features: Maximum number of keypoints to return
    
    Returns:
        keypoints: List of (y, x) coordinates
        scores: List of Harris response values
    """
    # Threshold the response map
    R_thresh = R.copy()
    R_thresh[R < threshold] = 0
    
    # Find local maxima using peak_local_max
    coords = peak_local_max(R_thresh, min_distance=min_distance, 
                           threshold_abs=threshold, num_peaks=max_features)
    
    if len(coords) == 0:
        return [], []
    
    # Convert to list of (y, x) coordinates and get scores
    keypoints = [(coord[0], coord[1]) for coord in coords]
    scores = [R[coord[0], coord[1]] for coord in coords]
    
    # Sort by score (highest first) and limit to max_features
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keypoints = [keypoints[i] for i in sorted_indices[:max_features]]
    scores = [scores[i] for i in sorted_indices[:max_features]]
    
    return keypoints, scores


def compute_keypoint_orientations(grad_x, grad_y, keypoints, window_size=11):
    """Compute dominant orientations for keypoints.
    
    Args:
        grad_x, grad_y: Image gradients
        keypoints: List of (y, x) keypoint coordinates
        window_size: Size of window around keypoint for orientation computation
    
    Returns:
        orientations: List of dominant orientations in radians
    """
    orientations = []
    half_window = window_size // 2
    height, width = grad_x.shape
    
    for y, x in keypoints:
        # Define window bounds
        y_min = max(0, y - half_window)
        y_max = min(height, y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(width, x + half_window + 1)
        
        # Extract gradients in window
        gx_window = grad_x[y_min:y_max, x_min:x_max]
        gy_window = grad_y[y_min:y_max, x_min:x_max]
        
        # Compute gradient magnitudes and angles
        magnitudes = np.sqrt(gx_window**2 + gy_window**2)
        angles = np.arctan2(gy_window, gx_window)
        
        # Create orientation histogram (36 bins, 10 degrees each)
        hist_bins = 36
        hist, bin_edges = np.histogram(angles, bins=hist_bins, range=(-np.pi, np.pi), 
                                     weights=magnitudes)
        
        # Find dominant orientation (peak in histogram)
        peak_idx = np.argmax(hist)
        dominant_angle = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
        
        orientations.append(dominant_angle)
    
    return orientations


def visualize_harris_keypoints(image, keypoints, scores, orientations, title, save_path):
    """Visualize Harris keypoints on image."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image with keypoints
    axes[0].imshow(image, cmap='gray')
    if keypoints:
        y_coords, x_coords = zip(*keypoints)
        axes[0].scatter(x_coords, y_coords, c='red', s=20, marker='+')
        
        # Draw orientation lines
        for (y, x), orientation in zip(keypoints, orientations):
            length = 10
            dx = length * np.cos(orientation)
            dy = length * np.sin(orientation)
            axes[0].arrow(x, y, dx, dy, head_width=2, head_length=2, fc='yellow', ec='yellow')
    
    axes[0].set_title(f'{title} - Keypoints ({len(keypoints)} found)')
    axes[0].axis('off')
    
    # Harris response map
    axes[1].imshow(image, cmap='gray', alpha=0.7)
    if keypoints:
        y_coords, x_coords = zip(*keypoints)
        # Scale scores for visualization
        scaled_scores = np.array(scores) * 1000  # Scale up for visibility
        scatter = axes[1].scatter(x_coords, y_coords, c=scaled_scores, s=50, 
                                cmap='hot', alpha=0.8, marker='o')
        plt.colorbar(scatter, ax=axes[1], label='Harris Response (×1000)')
    
    axes[1].set_title(f'{title} - Response Strength')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Harris keypoints visualization: {save_path}")


def save_keypoints_data(keypoints, scores, orientations, save_path):
    """Save keypoint data to file."""
    if not keypoints:
        print(f"No keypoints to save for {save_path}")
        return
    
    data = []
    for (y, x), score, orientation in zip(keypoints, scores, orientations):
        data.append([x, y, score, orientation])
    
    np.savetxt(save_path, data, fmt='%.6f', 
               header='x y score orientation', 
               comments='# Harris keypoints: ')
    print(f"Saved keypoints data: {save_path}")


def harris_feature_detection(image, sigma=1.0, k=0.04, threshold=0.01, 
                           min_distance=5, max_features=100, window_size=11):
    """Complete Harris feature detection pipeline.
    
    Returns:
        keypoints: List of (y, x) coordinates
        scores: List of Harris response values
        orientations: List of dominant orientations in radians
        R: Harris response map
    """
    # Compute Harris response
    R, grad_x, grad_y = compute_harris_response(image, sigma, k)
    
    # Find keypoints
    keypoints, scores = find_harris_keypoints(R, threshold, min_distance, max_features)
    
    # Compute orientations
    orientations = compute_keypoint_orientations(grad_x, grad_y, keypoints, window_size)
    
    return keypoints, scores, orientations, R


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='out')
    p.add_argument('--size', type=int, default=100)
    p.add_argument('--sigma', type=float, default=1.0, help='Gaussian window parameter')
    p.add_argument('--k', type=float, default=0.04, help='Harris detector free parameter')
    p.add_argument('--threshold', type=float, default=0.01, help='Harris response threshold')
    p.add_argument('--min-distance', type=int, default=5, help='Minimum distance between keypoints')
    p.add_argument('--max-features', type=int, default=100, help='Maximum number of keypoints')
    p.add_argument('--window-size', type=int, default=11, help='Window size for orientation computation')
    args = p.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, 'part_d'))
    
    save_dir = os.path.join(out_dir, 'part_d')

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

    print(f"Processing {len(images)} images with Harris feature detection")
    print(f"Parameters: sigma={args.sigma}, k={args.k}, threshold={args.threshold}")
    print(f"            min_distance={args.min_distance}, max_features={args.max_features}")

    # Process each image
    all_results = {}
    for name, image in images.items():
        print(f"\nProcessing {name}...")
        
        # Apply Harris feature detection
        keypoints, scores, orientations, R = harris_feature_detection(
            image, args.sigma, args.k, args.threshold, 
            args.min_distance, args.max_features, args.window_size)
        
        print(f"  Found {len(keypoints)} Harris keypoints")
        if keypoints:
            print(f"  Score range: [{min(scores):.6f}, {max(scores):.6f}]")
            print(f"  Orientation range: [{min(orientations):.3f}, {max(orientations):.3f}] radians")
        
        # Save results
        all_results[name] = {
            'keypoints': keypoints,
            'scores': scores,
            'orientations': orientations,
            'response_map': R
        }
        
        # Visualize keypoints
        visualize_harris_keypoints(image, keypoints, scores, orientations, name,
                                 os.path.join(save_dir, f'{name}_harris_keypoints.png'))
        
        # Save keypoints data
        save_keypoints_data(keypoints, scores, orientations,
                          os.path.join(save_dir, f'{name}_harris_keypoints.txt'))
        
        # Save response map
        plt.imsave(os.path.join(save_dir, f'{name}_harris_response.png'), R, cmap='hot')

    # Summary statistics
    print(f"\n=== Harris Feature Detection Summary ===")
    total_keypoints = sum(len(results['keypoints']) for results in all_results.values())
    print(f"Total keypoints detected: {total_keypoints}")
    
    for name, results in all_results.items():
        n_kpts = len(results['keypoints'])
        if n_kpts > 0:
            avg_score = np.mean(results['scores'])
            print(f"  {name}: {n_kpts} keypoints, avg score: {avg_score:.6f}")
        else:
            print(f"  {name}: {n_kpts} keypoints")

    print(f"\nHarris feature detection completed!")
    print(f"Results saved to: {save_dir}")
    print("Generated files:")
    print("  - *_harris_keypoints.png: Keypoints visualization with orientations")
    print("  - *_harris_keypoints.txt: Keypoint coordinates, scores, and orientations")
    print("  - *_harris_response.png: Harris response maps")


if __name__ == '__main__':
    main()