"""Question 5 - Part 1: High-Dimensional Dimensionality Reduction Comparison

Compare t-SNE, LLE, UMAP, and Isomap on a high-dimensional dataset.
Visualize resulting 2D embeddings from each method.

Usage:
    python -m q5.part1_manifold_comparison
    python -m q5.part1_manifold_comparison --dataset mnist --n-samples 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.datasets import fetch_openml, load_digits
from sklearn.preprocessing import StandardScaler
import time
import warnings

warnings.filterwarnings('ignore')

# Try to import UMAP, provide fallback if not available
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")


def load_dataset(dataset_name: str = 'digits', n_samples: int = None, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Load high-dimensional dataset."""
    
    if dataset_name == 'digits':
        # sklearn digits: 1797 samples, 64 features (8x8 images)
        data = load_digits()
        X, y = data.data, data.target
        print(f"Loaded sklearn digits dataset: {X.shape}")
        
    elif dataset_name == 'mnist':
        # MNIST: 70000 samples, 784 features (28x28 images)
        print("Downloading MNIST dataset (this may take a moment)...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        print(f"Loaded MNIST dataset: {X.shape}")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Subsample if requested
    if n_samples and n_samples < len(X):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=n_samples, replace=False)
        X, y = X[indices], y[indices]
        print(f"Subsampled to {n_samples} samples")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y


def apply_tsne(X: np.ndarray, random_state: int = 0, verbose: bool = True) -> Tuple[np.ndarray, float]:
    """Apply t-SNE dimensionality reduction."""
    start = time.time()
    
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        random_state=random_state,
        verbose=0
    )
    X_embedded = tsne.fit_transform(X)
    
    elapsed = time.time() - start
    
    if verbose:
        print(f"t-SNE completed in {elapsed:.2f}s")
    
    return X_embedded, elapsed


def apply_lle(X: np.ndarray, n_neighbors: int = 10, random_state: int = 0, verbose: bool = True) -> Tuple[np.ndarray, float]:
    """Apply Locally Linear Embedding (LLE)."""
    start = time.time()
    
    lle = LocallyLinearEmbedding(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_jobs=-1
    )
    X_embedded = lle.fit_transform(X)
    
    elapsed = time.time() - start
    
    if verbose:
        print(f"LLE completed in {elapsed:.2f}s")
    
    return X_embedded, elapsed


def apply_umap(X: np.ndarray, n_neighbors: int = 15, random_state: int = 0, verbose: bool = True) -> Tuple[np.ndarray, float]:
    """Apply UMAP dimensionality reduction."""
    if not UMAP_AVAILABLE:
        print("UMAP not available, skipping...")
        return None, 0.0
    
    start = time.time()
    
    umap = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=random_state,
        verbose=False
    )
    X_embedded = umap.fit_transform(X)
    
    elapsed = time.time() - start
    
    if verbose:
        print(f"UMAP completed in {elapsed:.2f}s")
    
    return X_embedded, elapsed


def apply_isomap(X: np.ndarray, n_neighbors: int = 10, verbose: bool = True) -> Tuple[np.ndarray, float]:
    """Apply Isomap dimensionality reduction."""
    start = time.time()
    
    isomap = Isomap(
        n_components=2,
        n_neighbors=n_neighbors,
        n_jobs=-1
    )
    X_embedded = isomap.fit_transform(X)
    
    elapsed = time.time() - start
    
    if verbose:
        print(f"Isomap completed in {elapsed:.2f}s")
    
    return X_embedded, elapsed


def plot_embeddings(embeddings: dict, y: np.ndarray, output_path: Path, dataset_name: str):
    """Plot all embeddings side by side."""
    
    # Filter out None embeddings
    valid_embeddings = {k: v for k, v in embeddings.items() if v is not None}
    n_methods = len(valid_embeddings)
    
    if n_methods == 0:
        print("No embeddings to plot!")
        return
    
    # Create subplots
    n_cols = 2
    n_rows = (n_methods + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Get unique classes and color map
    classes = np.unique(y)
    n_classes = len(classes)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # Plot each method
    for idx, (method_name, X_emb) in enumerate(valid_embeddings.items()):
        ax = axes_flat[idx]
        
        # Plot each class with different color
        for class_idx, color in zip(classes, colors):
            mask = y == class_idx
            ax.scatter(
                X_emb[mask, 0],
                X_emb[mask, 1],
                c=[color],
                label=str(class_idx),
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        
        ax.set_title(method_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)
        
        # Add legend for first plot only (to avoid clutter)
        if idx == 0:
            ax.legend(loc='best', ncol=2, fontsize=8, title='Class')
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    fig.suptitle(f'Manifold Learning Comparison - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved embeddings plot: {output_path}")
    plt.close()


def run_comparison(dataset: str = 'digits', n_samples: int = None, 
                  output_dir: str = 'outputs/q5_manifold', 
                  random_state: int = 0, verbose: bool = True):
    """Run complete manifold learning comparison."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Q5 Part 1: High-Dimensional Dimensionality Reduction Comparison")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading {dataset} dataset...")
    X, y = load_dataset(dataset, n_samples, random_state)
    print(f"Dataset shape: {X.shape}, Classes: {len(np.unique(y))}")
    
    # Apply all methods
    print("\n" + "="*70)
    print("Applying Dimensionality Reduction Methods")
    print("="*70)
    
    embeddings = {}
    timings = {}
    
    print("\n1. t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    X_tsne, time_tsne = apply_tsne(X, random_state, verbose)
    embeddings['t-SNE'] = X_tsne
    timings['t-SNE'] = time_tsne
    
    print("\n2. LLE (Locally Linear Embedding)")
    X_lle, time_lle = apply_lle(X, random_state=random_state, verbose=verbose)
    embeddings['LLE'] = X_lle
    timings['LLE'] = time_lle
    
    print("\n3. UMAP (Uniform Manifold Approximation and Projection)")
    if UMAP_AVAILABLE:
        X_umap, time_umap = apply_umap(X, random_state=random_state, verbose=verbose)
        embeddings['UMAP'] = X_umap
        timings['UMAP'] = time_umap
    else:
        print("   UMAP not available (install with: pip install umap-learn)")
    
    print("\n4. Isomap (Isometric Mapping)")
    X_isomap, time_isomap = apply_isomap(X, verbose=verbose)
    embeddings['Isomap'] = X_isomap
    timings['Isomap'] = time_isomap
    
    # Plot results
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)
    
    plot_embeddings(embeddings, y, output_path / f'{dataset}_embeddings.png', dataset)
    
    # Save results
    results_file = output_path / f'{dataset}_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Q5 Part 1: Manifold Learning Comparison - {dataset.upper()}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Samples: {len(X)}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Classes: {len(np.unique(y))}\n\n")
        
        f.write("Computation Times:\n")
        f.write("-"*70 + "\n")
        for method, elapsed in timings.items():
            f.write(f"{method:15s}: {elapsed:8.2f}s\n")
        
        f.write("\n\nMethod Characteristics:\n")
        f.write("-"*70 + "\n")
        f.write("t-SNE:\n")
        f.write("  - Preserves local structure very well\n")
        f.write("  - Good for visualization and cluster discovery\n")
        f.write("  - Non-deterministic (different runs give different results)\n")
        f.write("  - Slow for large datasets\n\n")
        
        f.write("LLE (Locally Linear Embedding):\n")
        f.write("  - Assumes data lies on locally linear patches\n")
        f.write("  - Preserves local neighborhood structure\n")
        f.write("  - Faster than t-SNE\n")
        f.write("  - Can struggle with noisy data\n\n")
        
        f.write("UMAP:\n")
        f.write("  - Preserves both local and global structure\n")
        f.write("  - Faster than t-SNE\n")
        f.write("  - Better scalability to large datasets\n")
        f.write("  - Often produces tighter clusters than t-SNE\n\n")
        
        f.write("Isomap:\n")
        f.write("  - Preserves geodesic distances\n")
        f.write("  - Good for datasets on curved manifolds\n")
        f.write("  - Sensitive to neighborhood size parameter\n")
        f.write("  - Can handle non-linear structures\n")
    
    print(f"\nResults saved: {results_file}")
    print(f"\nComparison complete! Outputs in: {output_path.resolve()}")
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"{'Method':<15} {'Time (s)':<12} {'Status'}")
    print("-"*70)
    for method in ['t-SNE', 'LLE', 'UMAP', 'Isomap']:
        if method in timings:
            print(f"{method:<15} {timings[method]:<12.2f} ✓")
        else:
            print(f"{method:<15} {'N/A':<12} ✗ (not available)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare t-SNE, LLE, UMAP, and Isomap for dimensionality reduction"
    )
    parser.add_argument(
        '--dataset', type=str, default='digits', choices=['digits', 'mnist'],
        help='Dataset to use (digits=1797x64, mnist=70000x784)'
    )
    parser.add_argument(
        '--n-samples', type=int, default=None,
        help='Subsample dataset to this many samples (for speed)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs/q5_manifold',
        help='Output directory'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='Verbose output'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    run_comparison(
        dataset=args.dataset,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        random_state=args.seed,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()