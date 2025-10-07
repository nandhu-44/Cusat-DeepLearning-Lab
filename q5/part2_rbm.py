"""Question 5 - Part 2: Restricted Boltzmann Machine (RBM)

Implement a simple RBM with binary visible and hidden units.
Train on binarized MNIST or toy dataset.
Visualize learned weights (filters) after training.
Track reconstruction error during training.

Usage:
    python -m q5.part2_rbm --dataset toy --epochs 100
    python -m q5.part2_rbm --dataset mnist --n-samples 5000 --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_digits
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class RBM:
    """Restricted Boltzmann Machine with binary visible and hidden units."""
    
    def __init__(self, n_visible: int, n_hidden: int, learning_rate: float = 0.1, 
                 momentum: float = 0.5, random_state: int = 0):
        """
        Initialize RBM.
        
        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units
            learning_rate: Learning rate for weight updates
            momentum: Momentum coefficient for gradient updates
            random_state: Random seed
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize weights and biases
        rng = np.random.default_rng(random_state)
        self.W = rng.normal(0, 0.01, size=(n_visible, n_hidden))
        self.vbias = np.zeros(n_visible)
        self.hbias = np.zeros(n_hidden)
        
        # Momentum terms
        self.W_momentum = np.zeros_like(self.W)
        self.vbias_momentum = np.zeros_like(self.vbias)
        self.hbias_momentum = np.zeros_like(self.hbias)
        
        # Training history
        self.reconstruction_errors = []
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def sample_hidden(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample hidden units given visible units.
        
        Returns:
            h_prob: Probability of hidden units being 1
            h_sample: Binary sample from h_prob
        """
        h_prob = self.sigmoid(v @ self.W + self.hbias)
        h_sample = (np.random.random(h_prob.shape) < h_prob).astype(float)
        return h_prob, h_sample
    
    def sample_visible(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample visible units given hidden units.
        
        Returns:
            v_prob: Probability of visible units being 1
            v_sample: Binary sample from v_prob
        """
        v_prob = self.sigmoid(h @ self.W.T + self.vbias)
        v_sample = (np.random.random(v_prob.shape) < v_prob).astype(float)
        return v_prob, v_sample
    
    def contrastive_divergence(self, v_data: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform k-step contrastive divergence.
        
        Args:
            v_data: Input visible data (batch_size, n_visible)
            k: Number of Gibbs sampling steps
            
        Returns:
            positive_grad: Positive phase gradient
            negative_grad: Negative phase gradient
            v_recon_prob: Reconstructed visible probabilities
        """
        batch_size = v_data.shape[0]
        
        # Positive phase
        h_prob_pos, h_sample_pos = self.sample_hidden(v_data)
        positive_grad = v_data.T @ h_prob_pos / batch_size
        
        # Negative phase (k-step Gibbs sampling)
        v_sample = v_data
        for _ in range(k):
            h_prob, h_sample = self.sample_hidden(v_sample)
            v_prob, v_sample = self.sample_visible(h_sample)
        
        h_prob_neg, _ = self.sample_hidden(v_sample)
        negative_grad = v_sample.T @ h_prob_neg / batch_size
        
        return positive_grad, negative_grad, v_prob
    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32, 
            k: int = 1, verbose: bool = True) -> 'RBM':
        """
        Train RBM using contrastive divergence.
        
        Args:
            X: Training data (n_samples, n_visible)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            k: Number of Gibbs sampling steps
            verbose: Print progress
        """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            epoch_error = 0.0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = start + batch_size
                batch = X_shuffled[start:end]
                
                # Contrastive divergence
                pos_grad, neg_grad, v_recon = self.contrastive_divergence(batch, k)
                
                # Compute reconstruction error for this batch
                batch_error = np.mean((batch - v_recon) ** 2)
                epoch_error += batch_error
                
                # Update weights with momentum
                self.W_momentum = self.momentum * self.W_momentum + self.learning_rate * (pos_grad - neg_grad)
                self.W += self.W_momentum
                
                # Update biases
                vbias_grad = np.mean(batch - v_recon, axis=0)
                self.vbias_momentum = self.momentum * self.vbias_momentum + self.learning_rate * vbias_grad
                self.vbias += self.vbias_momentum
                
                h_pos = self.sigmoid(batch @ self.W + self.hbias)
                h_neg = self.sigmoid(v_recon @ self.W + self.hbias)
                hbias_grad = np.mean(h_pos - h_neg, axis=0)
                self.hbias_momentum = self.momentum * self.hbias_momentum + self.learning_rate * hbias_grad
                self.hbias += self.hbias_momentum
            
            # Average error for epoch
            avg_error = epoch_error / n_batches
            self.reconstruction_errors.append(avg_error)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Reconstruction Error: {avg_error:.6f}")
        
        return self
    
    def reconstruct(self, v: np.ndarray) -> np.ndarray:
        """Reconstruct visible units from input."""
        h_prob, _ = self.sample_hidden(v)
        v_prob, _ = self.sample_visible(h_prob)
        return v_prob
    
    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """Compute reconstruction error on dataset."""
        v_recon = self.reconstruct(X)
        return np.mean((X - v_recon) ** 2)


def load_dataset(dataset_name: str = 'toy', n_samples: int = None, random_state: int = 0) -> Tuple[np.ndarray, tuple]:
    """Load and binarize dataset."""
    
    if dataset_name == 'toy':
        # Create synthetic binary patterns
        rng = np.random.default_rng(random_state)
        n = n_samples if n_samples else 1000
        X = rng.binomial(1, 0.3, size=(n, 64))
        shape = (8, 8)
        print(f"Generated toy dataset: {X.shape}")
        
    elif dataset_name == 'digits':
        # sklearn digits
        from sklearn.datasets import load_digits
        data = load_digits()
        X = data.data
        
        # Binarize
        threshold = np.median(X)
        X = (X > threshold).astype(float)
        
        shape = (8, 8)
        print(f"Loaded digits dataset: {X.shape}")
        
    elif dataset_name == 'mnist':
        # MNIST
        print("Downloading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto', as_frame=False)
        X = mnist.data
        
        # Binarize (threshold at 0.5 after normalizing to [0,1])
        X = X / 255.0
        X = (X > 0.5).astype(float)
        
        shape = (28, 28)
        print(f"Loaded MNIST dataset: {X.shape}")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Subsample if requested
    if n_samples and n_samples < len(X):
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=n_samples, replace=False)
        X = X[indices]
        print(f"Subsampled to {n_samples} samples")
    
    return X, shape


def visualize_weights(rbm: RBM, shape: tuple, output_path: Path, n_show: int = 100):
    """Visualize learned weight filters."""
    
    n_hidden = min(n_show, rbm.n_hidden)
    n_cols = 10
    n_rows = (n_hidden + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.2))
    axes = axes.flatten()
    
    for i in range(n_hidden):
        # Get weight vector for this hidden unit
        w = rbm.W[:, i].reshape(shape)
        
        ax = axes[i]
        ax.imshow(w, cmap='RdBu', aspect='auto')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_hidden, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'RBM Learned Filters (First {n_hidden} Hidden Units)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved weight visualization: {output_path}")
    plt.close()


def plot_reconstruction_error(reconstruction_errors: list, output_path: Path):
    """Plot reconstruction error over training epochs."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = range(1, len(reconstruction_errors) + 1)
    ax.plot(epochs, reconstruction_errors, linewidth=2, color='darkblue')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('RBM Training: Reconstruction Error Over Time', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    final_error = reconstruction_errors[-1]
    initial_error = reconstruction_errors[0]
    improvement = ((initial_error - final_error) / initial_error) * 100
    
    ax.text(0.98, 0.97, 
            f"Initial Error: {initial_error:.6f}\n"
            f"Final Error: {final_error:.6f}\n"
            f"Improvement: {improvement:.1f}%",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstruction error plot: {output_path}")
    plt.close()


def visualize_reconstructions(rbm: RBM, X: np.ndarray, shape: tuple, 
                             output_path: Path, n_examples: int = 10):
    """Visualize original and reconstructed samples."""
    
    # Select random samples
    indices = np.random.choice(len(X), n_examples, replace=False)
    X_samples = X[indices]
    
    # Reconstruct
    X_recon = rbm.reconstruct(X_samples)
    
    # Plot
    fig, axes = plt.subplots(2, n_examples, figsize=(n_examples * 1.2, 2.5))
    
    for i in range(n_examples):
        # Original
        axes[0, i].imshow(X_samples[i].reshape(shape), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10)
        
        # Reconstruction
        axes[1, i].imshow(X_recon[i].reshape(shape), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=10)
    
    fig.suptitle('RBM Input Reconstruction Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstruction examples: {output_path}")
    plt.close()


def run_rbm_experiment(dataset: str = 'toy', n_samples: int = None,
                      n_hidden: int = 64, epochs: int = 100, batch_size: int = 32,
                      learning_rate: float = 0.1, output_dir: str = 'outputs/q5_rbm',
                      random_state: int = 0, verbose: bool = True):
    """Run complete RBM experiment."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Q5 Part 2: Restricted Boltzmann Machine (RBM)")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading {dataset} dataset...")
    X, shape = load_dataset(dataset, n_samples, random_state)
    n_visible = X.shape[1]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Image shape: {shape}")
    print(f"Binary values: {np.unique(X)}")
    
    # Initialize RBM
    print("\n" + "="*70)
    print("Initializing RBM")
    print("="*70)
    print(f"Visible units: {n_visible}")
    print(f"Hidden units: {n_hidden}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    
    rbm = RBM(
        n_visible=n_visible,
        n_hidden=n_hidden,
        learning_rate=learning_rate,
        momentum=0.5,
        random_state=random_state
    )
    
    # Train RBM
    print("\n" + "="*70)
    print("Training RBM")
    print("="*70)
    
    rbm.fit(X, epochs=epochs, batch_size=batch_size, k=1, verbose=verbose)
    
    # Visualize results
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)
    
    # 1. Learned weights (filters)
    visualize_weights(rbm, shape, output_path / f'{dataset}_weights.png', n_show=100)
    
    # 2. Reconstruction error curve
    plot_reconstruction_error(rbm.reconstruction_errors, 
                             output_path / f'{dataset}_reconstruction_error.png')
    
    # 3. Reconstruction examples
    visualize_reconstructions(rbm, X, shape, 
                            output_path / f'{dataset}_reconstructions.png', 
                            n_examples=10)
    
    # Final evaluation
    final_error = rbm.get_reconstruction_error(X)
    print(f"\nFinal reconstruction error on full dataset: {final_error:.6f}")
    
    # Save results
    results_file = output_path / f'{dataset}_results.txt'
    with open(results_file, 'w') as f:
        f.write("Q5 Part 2: Restricted Boltzmann Machine - Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Samples: {len(X)}\n")
        f.write(f"Visible units: {n_visible}\n")
        f.write(f"Hidden units: {n_hidden}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n\n")
        
        f.write("Training Results:\n")
        f.write("-"*70 + "\n")
        f.write(f"Initial reconstruction error: {rbm.reconstruction_errors[0]:.6f}\n")
        f.write(f"Final reconstruction error: {rbm.reconstruction_errors[-1]:.6f}\n")
        f.write(f"Error on full dataset: {final_error:.6f}\n\n")
        
        improvement = ((rbm.reconstruction_errors[0] - rbm.reconstruction_errors[-1]) / 
                      rbm.reconstruction_errors[0]) * 100
        f.write(f"Improvement: {improvement:.2f}%\n\n")
        
        f.write("\nHow does reconstruction error change during training?\n")
        f.write("-"*70 + "\n")
        f.write("1. Initially high: Random weights produce poor reconstructions\n")
        f.write("2. Rapid decrease: RBM learns major patterns in first epochs\n")
        f.write("3. Gradual convergence: Fine-tuning to capture details\n")
        f.write("4. Plateau: Model reaches capacity for given architecture\n\n")
        f.write("The error typically follows a logarithmic decay pattern,\n")
        f.write("with fast initial learning followed by diminishing returns.\n")
    
    print(f"\nResults saved: {results_file}")
    print(f"\nRBM experiment complete! Outputs in: {output_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RBM and visualize results")
    parser.add_argument(
        '--dataset', type=str, default='toy', 
        choices=['toy', 'digits', 'mnist'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--n-samples', type=int, default=None,
        help='Subsample dataset to this many samples'
    )
    parser.add_argument(
        '--n-hidden', type=int, default=64,
        help='Number of hidden units'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Mini-batch size'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.1,
        help='Learning rate'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs/q5_rbm',
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
    
    run_rbm_experiment(
        dataset=args.dataset,
        n_samples=args.n_samples,
        n_hidden=args.n_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        random_state=args.seed,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()