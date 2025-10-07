#!/usr/bin/env python3
"""Part F: Comparison of Image Descriptions and Report

Usage example:
    python src/part_f.py --out-dir out
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from datetime import datetime

from utils import ensure_dir

EXPECTED_IMAGES = [
    "cardinal1", "cardinal2", "leopard1", "leopard2", "panda1", "panda2"
]

# Define animal categories
CATEGORIES = {
    'cardinal': ['cardinal1', 'cardinal2'],
    'leopard': ['leopard1', 'leopard2'],
    'panda': ['panda1', 'panda2']
}


def load_texture_representations(part_b_dir):
    """Load texture representations from Part B."""
    representations = {}
    
    npz_path = os.path.join(part_b_dir, 'texture_representations.npz')
    if not os.path.exists(npz_path):
        print(f"Warning: {npz_path} not found. Skipping texture representations.")
        return representations
    
    data = np.load(npz_path)
    
    for img_name in EXPECTED_IMAGES:
        if img_name in data:
            representations[f'{img_name}_individual'] = data[img_name]
    
    for category in ['cardinal', 'leopard', 'panda']:
        key = f'{category}_mean'
        if key in data:
            representations[f'{category}_mean'] = data[key]
    
    for category in ['cardinal', 'leopard', 'panda']:
        key = f'{category}_concat'
        if key in data:
            representations[f'{category}_concat'] = data[key]
    
    return representations


def load_bow_representations(part_e_dir):
    """Load bag-of-words representations from Part E."""
    representations = {}
    
    npz_path = os.path.join(part_e_dir, 'bow_histograms.npz')
    if not os.path.exists(npz_path):
        print(f"Warning: {npz_path} not found. Skipping BoW representations.")
        return representations
    
    data = np.load(npz_path)
    histograms = data['histograms']
    image_names = data['image_names']
    
    for i, img_name in enumerate(image_names):
        representations[f'{img_name}_bow'] = histograms[i]
    
    return representations


def compute_within_between_distances(representations, representation_type):
    """Compute within-class and between-class distances for a representation type."""
    
    type_reprs = {}
    for img_name in EXPECTED_IMAGES:
        key = f'{img_name}_{representation_type}'
        if key in representations:
            type_reprs[img_name] = representations[key]
    
    if len(type_reprs) < 2:
        return [], [], 0.0, 0.0, 0.0
    
    img_names = list(type_reprs.keys())
    repr_matrix = np.array([type_reprs[name] for name in img_names])
    
    distance_matrix = pairwise_distances(repr_matrix, metric='euclidean')
    
    within_class_distances = []
    between_class_distances = []
    
    for category, category_images in CATEGORIES.items():
        category_indices = [i for i, name in enumerate(img_names) if name in category_images]
        
        for i in range(len(category_indices)):
            for j in range(i+1, len(category_indices)):
                idx_i, idx_j = category_indices[i], category_indices[j]
                within_class_distances.append(distance_matrix[idx_i, idx_j])
    
    for i in range(len(img_names)):
        for j in range(i+1, len(img_names)):
            name_i, name_j = img_names[i], img_names[j]
            
            category_i = None
            category_j = None
            for cat, images in CATEGORIES.items():
                if name_i in images:
                    category_i = cat
                if name_j in images:
                    category_j = cat
            
            if category_i != category_j and category_i is not None and category_j is not None:
                between_class_distances.append(distance_matrix[i, j])
    
    avg_within = np.mean(within_class_distances) if within_class_distances else 0.0
    avg_between = np.mean(between_class_distances) if between_class_distances else 0.0
    ratio = avg_between / avg_within if avg_within > 0 else float('inf')
    
    return within_class_distances, between_class_distances, avg_within, avg_between, ratio


def analyze_special_representations(representations):
    """Analyze the special texture representations (mean and concatenated)."""
    results = {}
    
    mean_reprs = {}
    for category in ['cardinal', 'leopard', 'panda']:
        key = f'{category}_mean'
        if key in representations:
            mean_reprs[category] = representations[key]
    
    if len(mean_reprs) >= 2:
        categories = list(mean_reprs.keys())
        repr_matrix = np.array([mean_reprs[cat] for cat in categories])
        distance_matrix = pairwise_distances(repr_matrix, metric='euclidean')
        
        results['mean_distances'] = {}
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                pair = f'{categories[i]}-{categories[j]}'
                results['mean_distances'][pair] = distance_matrix[i, j]
    
    concat_reprs = {}
    for category in ['cardinal', 'leopard', 'panda']:
        key = f'{category}_concat'
        if key in representations:
            concat_reprs[category] = representations[key]
    
    if len(concat_reprs) >= 2:
        categories = list(concat_reprs.keys())
        repr_matrix = np.array([concat_reprs[cat] for cat in categories])
        distance_matrix = pairwise_distances(repr_matrix, metric='euclidean')
        
        results['concat_distances'] = {}
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                pair = f'{categories[i]}-{categories[j]}'
                results['concat_distances'][pair] = distance_matrix[i, j]
    
    return results


def visualize_comparison_results(results, save_path):
    """Create a comprehensive visualization of all results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    representation_types = []
    avg_within_list = []
    avg_between_list = []
    ratios = []
    
    for repr_type, data in results.items():
        if 'avg_within' in data:
            representation_types.append(repr_type.replace('_', ' ').title())
            avg_within_list.append(data['avg_within'])
            avg_between_list.append(data['avg_between'])
            ratios.append(data['ratio'])
    
    x_pos = np.arange(len(representation_types))
    
    axes[0, 0].bar(x_pos - 0.2, avg_within_list, 0.4, label='Within-class', alpha=0.7)
    axes[0, 0].bar(x_pos + 0.2, avg_between_list, 0.4, label='Between-class', alpha=0.7)
    axes[0, 0].set_xlabel('Representation Type')
    axes[0, 0].set_ylabel('Average Distance')
    axes[0, 0].set_title('Within-class vs Between-class Distances')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(representation_types, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    bars = axes[0, 1].bar(x_pos, ratios, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Representation Type')
    axes[0, 1].set_ylabel('Between/Within Ratio')
    axes[0, 1].set_title('Discriminative Power (Higher = Better)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(representation_types, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{ratio:.3f}', ha='center', va='bottom')
    
    all_within = []
    all_between = []
    labels = []
    
    for repr_type, data in results.items():
        if 'within_distances' in data and 'between_distances' in data:
            all_within.extend(data['within_distances'])
            all_between.extend(data['between_distances'])
            labels.extend([repr_type.replace('_', ' ').title()] * 
                         (len(data['within_distances']) + len(data['between_distances'])))
    
    if all_within and all_between:
        axes[1, 0].hist([all_within, all_between], bins=20, alpha=0.7, 
                       label=['Within-class', 'Between-class'])
        axes[1, 0].set_xlabel('Distance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distance Distribution (All Representations)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    summary_text = "Summary Statistics:\n\n"
    
    best_ratio = 0
    best_repr = ""
    for repr_type, data in results.items():
        if 'ratio' in data:
            summary_text += f"{repr_type.replace('_', ' ').title()}:\n"
            summary_text += f"  Ratio: {data['ratio']:.3f}\n"
            summary_text += f"  Within: {data['avg_within']:.3f}\n"
            summary_text += f"  Between: {data['avg_between']:.3f}\n\n"
            
            if data['ratio'] > best_ratio:
                best_ratio = data['ratio']
                best_repr = repr_type.replace('_', ' ').title()
    
    summary_text += f"Best discriminative power:\n{best_repr} (ratio: {best_ratio:.3f})"
    
    axes[1, 1].text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                   fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison visualization: {save_path}")


def generate_report(results, special_results, save_path):
    """Generate a comprehensive text report."""
    
    with open(save_path, 'w') as f:
        f.write("Computer Vision Homework 1 - Final Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("COMPARISON OF IMAGE DESCRIPTIONS\n")
        f.write("-" * 35 + "\n\n")
        
        f.write("This report compares different image representation methods:\n")
        f.write("1. Individual Texture Representations (Part B)\n")
        f.write("2. SIFT Bag-of-Words Representations (Part E)\n")
        f.write("3. Mean Texture Representations (Part B)\n")
        f.write("4. Concatenated Texture Representations (Part B)\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 12 + "\n")
        f.write("- Dataset: 6 animal images (2 cardinals, 2 leopards, 2 pandas)\n")
        f.write("- Distance metric: Euclidean distance\n")
        f.write("- Within-class: distances between images of same animal\n")
        f.write("- Between-class: distances between images of different animals\n")
        f.write("- Discriminative power: between-class / within-class ratio\n\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 15 + "\n")
        
        # Sort results by ratio for ranking
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('ratio', 0), reverse=True)
        
        f.write(f"{'Representation':<25} {'Within':<10} {'Between':<10} {'Ratio':<8} {'Rank':<4}\n")
        f.write("-" * 65 + "\n")
        
        for rank, (repr_type, data) in enumerate(sorted_results, 1):
            if 'avg_within' in data:
                name = repr_type.replace('_', ' ').title()
                f.write(f"{name:<25} {data['avg_within']:<10.3f} {data['avg_between']:<10.3f} "
                       f"{data['ratio']:<8.3f} {rank:<4}\n")
        
        f.write("\nDETAILED ANALYSIS\n")
        f.write("-" * 17 + "\n\n")
        
        for repr_type, data in sorted_results:
            if 'avg_within' not in data:
                continue
                
            f.write(f"{repr_type.replace('_', ' ').title()}\n")
            f.write("  " + "=" * (len(repr_type) + 5) + "\n")
            f.write(f"  Average within-class distance: {data['avg_within']:.6f}\n")
            f.write(f"  Average between-class distance: {data['avg_between']:.6f}\n")
            f.write(f"  Between/within ratio: {data['ratio']:.6f}\n")
            f.write(f"  Number of within-class pairs: {len(data['within_distances'])}\n")
            f.write(f"  Number of between-class pairs: {len(data['between_distances'])}\n")
            
            if data['within_distances']:
                f.write(f"  Within-class distance range: [{min(data['within_distances']):.3f}, "
                       f"{max(data['within_distances']):.3f}]\n")
            if data['between_distances']:
                f.write(f"  Between-class distance range: [{min(data['between_distances']):.3f}, "
                       f"{max(data['between_distances']):.3f}]\n")
            f.write("\n")
        
        # Special representations analysis
        if special_results:
            f.write("SPECIAL TEXTURE REPRESENTATIONS\n")
            f.write("-" * 31 + "\n\n")
            
            if 'mean_distances' in special_results:
                f.write("Mean Texture Representation Distances:\n")
                for pair, distance in special_results['mean_distances'].items():
                    f.write(f"  {pair}: {distance:.6f}\n")
                f.write("\n")
            
            if 'concat_distances' in special_results:
                f.write("Concatenated Texture Representation Distances:\n")
                for pair, distance in special_results['concat_distances'].items():
                    f.write(f"  {pair}: {distance:.6f}\n")
                f.write("\n")
        
        f.write("CONCLUSIONS\n")
        f.write("-" * 11 + "\n")
        
        if sorted_results:
            best_repr = sorted_results[0][0].replace('_', ' ').title()
            best_ratio = sorted_results[0][1]['ratio']
            f.write(f"1. Best performing representation: {best_repr}\n")
            f.write(f"   - Achieved highest discriminative ratio: {best_ratio:.3f}\n")
            f.write(f"   - This means between-class distances are {best_ratio:.1f}x larger than within-class\n\n")
        
        f.write("2. General observations:\n")
        f.write("   - Ratios > 1.0 indicate good class separation\n")
        f.write("   - Higher ratios suggest better discriminative power\n")
        f.write("   - Different representations capture different aspects of image content\n\n")
        
        f.write("3. Representation characteristics:\n")
        f.write("   - Texture representations: Capture global texture statistics\n")
        f.write("   - SIFT BoW: Captures local feature patterns and spatial relationships\n")
        f.write("   - Both approaches provide complementary information\n\n")
        
        f.write("This analysis demonstrates the effectiveness of different computer vision\n")
        f.write("techniques for image representation and classification tasks.\n")

    print(f"Generated comprehensive report: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='out')
    args = p.parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, 'part_f'))
    
    save_dir = os.path.join(out_dir, 'part_f')
    part_b_dir = os.path.join(out_dir, 'part_b')
    part_e_dir = os.path.join(out_dir, 'part_e')

    print("Loading representations from previous parts...")
    
    all_representations = {}
    
    texture_reprs = load_texture_representations(part_b_dir)
    all_representations.update(texture_reprs)
    print(f"Loaded {len(texture_reprs)} texture representations")
    
    bow_reprs = load_bow_representations(part_e_dir)
    all_representations.update(bow_reprs)
    print(f"Loaded {len(bow_reprs)} BoW representations")
    
    if len(all_representations) == 0:
        print("No representations found. Make sure Parts B and E have been completed.")
        return

    print(f"\nAnalyzing {len(all_representations)} total representations...")
    
    results = {}
    
    print("Analyzing individual texture representations...")
    within, between, avg_within, avg_between, ratio = compute_within_between_distances(
        all_representations, 'individual')
    if within or between:
        results['individual_texture'] = {
            'within_distances': within,
            'between_distances': between,
            'avg_within': avg_within,
            'avg_between': avg_between,
            'ratio': ratio
        }
    
    print("Analyzing SIFT BoW representations...")
    within, between, avg_within, avg_between, ratio = compute_within_between_distances(
        all_representations, 'bow')
    if within or between:
        results['sift_bow'] = {
            'within_distances': within,
            'between_distances': between,
            'avg_within': avg_within,
            'avg_between': avg_between,
            'ratio': ratio
        }
    
    print("Analyzing special texture representations...")
    special_results = analyze_special_representations(all_representations)
    
    print(f"\n=== COMPARISON RESULTS ===")
    for repr_type, data in results.items():
        name = repr_type.replace('_', ' ').title()
        print(f"{name}:")
        print(f"  Average within-class distance: {data['avg_within']:.6f}")
        print(f"  Average between-class distance: {data['avg_between']:.6f}")
        print(f"  Between/within ratio: {data['ratio']:.6f}")
        print()
    
    best_ratio = 0
    best_repr = ""
    for repr_type, data in results.items():
        if data['ratio'] > best_ratio:
            best_ratio = data['ratio']
            best_repr = repr_type.replace('_', ' ').title()
    
    if best_repr:
        print(f"Best discriminative power: {best_repr} (ratio: {best_ratio:.3f})")
    
    print(f"\nGenerating visualizations...")
    visualize_comparison_results(results, os.path.join(save_dir, 'comparison_results.png'))
    
    print(f"Generating comprehensive report...")
    generate_report(results, special_results, os.path.join(save_dir, 'results.txt'))
    
    np.savez(os.path.join(save_dir, 'comparison_data.npz'),
             results=results, special_results=special_results)
    
    print(f"\nPart F completed!")
    print(f"Results saved to: {save_dir}")
    print("Generated files:")
    print("  - results.txt: Comprehensive analysis report")
    print("  - comparison_results.png: Visual comparison of all methods")
    print("  - comparison_data.npz: Raw analysis data")


if __name__ == '__main__':
    main()