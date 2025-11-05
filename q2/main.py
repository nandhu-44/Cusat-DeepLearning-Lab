"""
Probabilistic PCA vs Classical PCA Analysis
============================================
1. PPCA vs Classical PCA on MNIST
   - Reconstruction quality comparison (MSE & PSNR)
   - Multiple latent dimensions
   - Visualization of reconstructions

2. PPCA for Missing Data Imputation
   - Synthetic low-rank Gaussian data
   - 10% missing entries
   - Comparison with mean imputation baseline
   
Implementation: NumPy + SciPy only (no sklearn/PyTorch)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os
import urllib.request
import gzip

# Create output directory
os.makedirs('./outputs/images', exist_ok=True)

print("=" * 80)
print("PROBABILISTIC PCA (PPCA) vs CLASSICAL PCA")
print("=" * 80)

# ============================================================================
# Helper Functions
# ============================================================================

def download_mnist():
    """Download MNIST dataset if not present"""
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    os.makedirs('./data/mnist', exist_ok=True)
    
    for key, filename in files.items():
        filepath = f'./data/mnist/{filename}'
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
    
    return files

def load_mnist_images(filepath):
    """Load MNIST images from gz file"""
    with gzip.open(filepath, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        
    return data.astype(np.float64) / 255.0  # Normalize to [0, 1]

def load_mnist_labels(filepath):
    """Load MNIST labels from gz file"""
    with gzip.open(filepath, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def calculate_psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Since normalized to [0, 1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# ============================================================================
# Classical PCA (via SVD)
# ============================================================================

class ClassicalPCA:
    """Classical PCA using SVD"""
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        
    def fit(self, X):
        """Fit PCA model
        
        Args:
            X: (n_samples, n_features) data matrix
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Perform SVD
        U, S, Vt = linalg.svd(X_centered, full_matrices=False)
        
        # Keep top n_components
        self.components = Vt[:self.n_components]
        self.explained_variance = (S[:self.n_components] ** 2) / (X.shape[0] - 1)
        
        return self
    
    def transform(self, X):
        """Project data to latent space"""
        X_centered = X - self.mean
        return X_centered @ self.components.T
    
    def inverse_transform(self, Z):
        """Reconstruct data from latent representation"""
        return Z @ self.components + self.mean
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)

# ============================================================================
# Probabilistic PCA (EM Algorithm)
# ============================================================================

class ProbabilisticPCA:
    """Probabilistic PCA using EM algorithm
    
    Model: x = W*z + mu + epsilon
    where z ~ N(0, I), epsilon ~ N(0, sigma^2 * I)
    """
    
    def __init__(self, n_components, max_iter=100, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.W = None  # Loading matrix
        self.mu = None  # Mean
        self.sigma2 = None  # Noise variance
        
    def fit(self, X, missing_mask=None):
        """Fit PPCA model using EM algorithm
        
        Args:
            X: (n_samples, n_features) data matrix
            missing_mask: Boolean mask where True indicates missing values
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        q = self.n_components
        
        # Handle missing data
        if missing_mask is None:
            X_work = X.copy()
            self.mu = np.mean(X, axis=0)
        else:
            X_work = X.copy()
            # Initialize missing values with column means
            for j in range(n_features):
                col_mean = np.mean(X_work[~missing_mask[:, j], j])
                X_work[missing_mask[:, j], j] = col_mean
            self.mu = np.mean(X_work, axis=0)
        
        # Center data
        X_centered = X_work - self.mu
        
        # Initialize parameters
        self.W = np.random.randn(n_features, q) * 0.01
        self.sigma2 = 1.0
        
        prev_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # ========== E-step ==========
            M = self.W.T @ self.W + self.sigma2 * np.eye(q)
            M_inv = linalg.inv(M)
            
            # Posterior mean: E[z|x]
            EZ = X_centered @ self.W @ M_inv  # (n_samples, q)
            
            # Posterior covariance: Cov[z|x]
            EZZt = self.sigma2 * M_inv  # (q, q)
            
            # ========== M-step ==========
            # Update W
            sum_EZZt = n_samples * EZZt + EZ.T @ EZ
            self.W = (X_centered.T @ EZ) @ linalg.inv(sum_EZZt)
            
            # Update sigma^2
            if missing_mask is None:
                residual = X_centered - EZ @ self.W.T
                self.sigma2 = np.sum(residual ** 2) / (n_samples * n_features)
                self.sigma2 += np.trace(EZZt @ self.W.T @ self.W) / n_features
            else:
                # Handle missing data in M-step
                total_error = 0
                total_count = 0
                for i in range(n_samples):
                    obs_idx = ~missing_mask[i]
                    if np.any(obs_idx):
                        residual = X_centered[i, obs_idx] - EZ[i] @ self.W[obs_idx].T
                        total_error += np.sum(residual ** 2)
                        total_count += np.sum(obs_idx)
                
                self.sigma2 = total_error / total_count
                self.sigma2 += np.trace(EZZt @ self.W.T @ self.W) / n_features
            
            # Compute log-likelihood for convergence check
            log_likelihood = self._compute_log_likelihood(X_work, EZ, M_inv)
            
            # Check convergence
            if abs(log_likelihood - prev_likelihood) < self.tol:
                print(f"  Converged at iteration {iteration + 1}")
                break
            
            prev_likelihood = log_likelihood
            
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration + 1}/{self.max_iter}, "
                      f"Log-likelihood: {log_likelihood:.4f}, sigma^2: {self.sigma2:.6f}")
        
        return self
    
    def _compute_log_likelihood(self, X, EZ, M_inv):
        """Compute log-likelihood"""
        X_centered = X - self.mu
        n_samples, n_features = X.shape
        
        C = self.W @ self.W.T + self.sigma2 * np.eye(n_features)
        
        try:
            sign, logdet = linalg.slogdet(C)
            if sign <= 0:
                return -np.inf
            
            C_inv = linalg.inv(C)
            ll = -0.5 * n_samples * (n_features * np.log(2 * np.pi) + logdet)
            ll -= 0.5 * np.sum(X_centered * (X_centered @ C_inv))
        except:
            return -np.inf
        
        return ll
    
    def transform(self, X):
        """Project data to latent space"""
        X_centered = X - self.mu
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.n_components)
        M_inv = linalg.inv(M)
        return X_centered @ self.W @ M_inv
    
    def inverse_transform(self, Z):
        """Reconstruct data from latent representation"""
        return Z @ self.W.T + self.mu
    
    def fit_transform(self, X, missing_mask=None):
        """Fit and transform in one step"""
        self.fit(X, missing_mask)
        return self.transform(X)
    
    def impute(self, X, missing_mask):
        """Impute missing values using posterior mean"""
        X_imputed = X.copy()
        Z = self.transform(X)
        X_reconstructed = self.inverse_transform(Z)
        X_imputed[missing_mask] = X_reconstructed[missing_mask]
        return X_imputed

# ============================================================================
# PART 1: MNIST Comparison
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: Classical PCA vs PPCA on MNIST")
print("=" * 80)

# Download and load MNIST
print("\nLoading MNIST dataset...")
files = download_mnist()
X_train = load_mnist_images('./data/mnist/train-images-idx3-ubyte.gz')
y_train = load_mnist_labels('./data/mnist/train-labels-idx1-ubyte.gz')
X_test = load_mnist_images('./data/mnist/t10k-images-idx3-ubyte.gz')
y_test = load_mnist_labels('./data/mnist/t10k-labels-idx1-ubyte.gz')

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Use subset for faster computation
n_train_samples = 10000
X_train_subset = X_train[:n_train_samples]
y_train_subset = y_train[:n_train_samples]

print(f"\nUsing {n_train_samples} training samples for analysis")

# Test different latent dimensions
latent_dims = [10, 20, 50, 100, 150]
results = {
    'dims': latent_dims,
    'pca_mse': [],
    'pca_psnr': [],
    'ppca_mse': [],
    'ppca_psnr': []
}

print("\n" + "-" * 80)
print("Comparing reconstruction quality across latent dimensions...")
print("-" * 80)

for n_comp in latent_dims:
    print(f"\n📊 Latent dimensions: {n_comp}")
    
    # Classical PCA
    print("  Classical PCA...")
    pca = ClassicalPCA(n_components=n_comp)
    Z_pca = pca.fit_transform(X_train_subset)
    X_recon_pca = pca.inverse_transform(Z_pca)
    
    mse_pca = np.mean((X_train_subset - X_recon_pca) ** 2)
    psnr_pca = calculate_psnr(X_train_subset, X_recon_pca)
    
    results['pca_mse'].append(mse_pca)
    results['pca_psnr'].append(psnr_pca)
    
    print(f"    MSE: {mse_pca:.6f}, PSNR: {psnr_pca:.2f} dB")
    
    # Probabilistic PCA
    print("  Probabilistic PCA (EM)...")
    ppca = ProbabilisticPCA(n_components=n_comp, max_iter=100, random_state=42)
    Z_ppca = ppca.fit_transform(X_train_subset)
    X_recon_ppca = ppca.inverse_transform(Z_ppca)
    
    mse_ppca = np.mean((X_train_subset - X_recon_ppca) ** 2)
    psnr_ppca = calculate_psnr(X_train_subset, X_recon_ppca)
    
    results['ppca_mse'].append(mse_ppca)
    results['ppca_psnr'].append(psnr_ppca)
    
    print(f"    MSE: {mse_ppca:.6f}, PSNR: {psnr_ppca:.2f} dB")
    print(f"    Noise variance (σ²): {ppca.sigma2:.6f}")

# Plot reconstruction quality comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MSE comparison
axes[0].plot(latent_dims, results['pca_mse'], 'o-', linewidth=2, 
             markersize=8, label='Classical PCA', color='#3498db')
axes[0].plot(latent_dims, results['ppca_mse'], 's-', linewidth=2, 
             markersize=8, label='Probabilistic PCA', color='#e74c3c')
axes[0].set_xlabel('Latent Dimensions', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
axes[0].set_title('Reconstruction Error vs Latent Dimensions', fontsize=14, fontweight='bold')
axes[0].legend(frameon=True, shadow=True)
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# PSNR comparison
axes[1].plot(latent_dims, results['pca_psnr'], 'o-', linewidth=2, 
             markersize=8, label='Classical PCA', color='#3498db')
axes[1].plot(latent_dims, results['ppca_psnr'], 's-', linewidth=2, 
             markersize=8, label='Probabilistic PCA', color='#e74c3c')
axes[1].set_xlabel('Latent Dimensions', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Peak Signal-to-Noise Ratio (PSNR) [dB]', fontsize=12, fontweight='bold')
axes[1].set_title('Reconstruction Quality vs Latent Dimensions', fontsize=14, fontweight='bold')
axes[1].legend(frameon=True, shadow=True)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/images/mnist_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Comparison plot saved to ./outputs/images/mnist_comparison.png")

# Visualize sample reconstructions
print("\nGenerating reconstruction visualizations...")
n_comp_viz = 50
pca_viz = ClassicalPCA(n_components=n_comp_viz)
ppca_viz = ProbabilisticPCA(n_components=n_comp_viz, max_iter=100, random_state=42)

Z_pca_viz = pca_viz.fit_transform(X_train_subset)
Z_ppca_viz = ppca_viz.fit_transform(X_train_subset)

X_recon_pca_viz = pca_viz.inverse_transform(Z_pca_viz)
X_recon_ppca_viz = ppca_viz.inverse_transform(Z_ppca_viz)

# Plot sample reconstructions
n_samples_viz = 10
fig, axes = plt.subplots(3, n_samples_viz, figsize=(20, 6))

for i in range(n_samples_viz):
    # Original
    axes[0, i].imshow(X_train_subset[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontsize=12, fontweight='bold')
    else:
        axes[0, i].set_title(f'{y_train_subset[i]}', fontsize=10)
    
    # Classical PCA reconstruction
    axes[1, i].imshow(X_recon_pca_viz[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Classical PCA', fontsize=12, fontweight='bold')
    
    # PPCA reconstruction
    axes[2, i].imshow(X_recon_ppca_viz[i].reshape(28, 28), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('Probabilistic PCA', fontsize=12, fontweight='bold')

plt.suptitle(f'MNIST Reconstructions (Latent Dimensions: {n_comp_viz})', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('./outputs/images/mnist_reconstructions.png', dpi=300, bbox_inches='tight')
print(f"✓ Reconstruction samples saved to ./outputs/images/mnist_reconstructions.png")

# ============================================================================
# Conceptual Diagram: PCA vs PPCA
# ============================================================================

print("\nGenerating conceptual diagram of PCA vs PPCA...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Classical PCA Diagram
axes[0].set_xlim(0, 10)
axes[0].set_ylim(0, 6)
axes[0].axis('off')
axes[0].set_title('Classical PCA', fontsize=16, fontweight='bold', pad=20)

# Data point
axes[0].text(1, 5, 'Data Point x', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue'))
axes[0].arrow(1.5, 5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Mean
axes[0].text(3, 5, 'Subtract Mean μ', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))
axes[0].arrow(4.2, 5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Projection
axes[0].text(6, 5, 'Project: z = W^T(x - μ)', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
axes[0].arrow(7.8, 5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Reconstruction
axes[0].text(9, 5, 'Reconstruct: x̂ = Wz + μ', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral'))

# Model equation
axes[0].text(5, 3, 'x = Wz + μ', fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))

# Assumptions
axes[0].text(5, 1, 'Assumptions:\n• Deterministic mapping\n• No noise model\n• Exact reconstruction\n  (if enough components)', 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray'))

# Probabilistic PCA Diagram
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 6)
axes[1].axis('off')
axes[1].set_title('Probabilistic PCA (PPCA)', fontsize=16, fontweight='bold', pad=20)

# Latent variable
axes[1].text(0.5, 5, 'Latent z ~ N(0, I)', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan'))
axes[1].arrow(2.2, 5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Linear mapping
axes[1].text(4, 5, 'Linear Mapping: Wz + μ', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))
axes[1].arrow(5.8, 5, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Noise
axes[1].text(7.5, 5, 'Add Noise ε ~ N(0, σ²I)', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightsalmon'))
axes[1].arrow(9.2, 5, 0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Data point
axes[1].text(9.5, 5, 'x', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue'))

# Generative process arrow
axes[1].arrow(5, 4, 0, -1, head_width=0.1, head_length=0.1, fc='red', ec='red')
axes[1].text(5.2, 3.5, 'Generative Process', fontsize=10, color='red', rotation=90)

# Inference arrow
axes[1].arrow(5, 2, 0, -1, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
axes[1].text(5.2, 1.5, 'Inference (EM)', fontsize=10, color='blue', rotation=90)

# Model equation
axes[1].text(5, 3.8, 'x = Wz + μ + ε', fontsize=14, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))

# Assumptions
axes[1].text(5, 1, 'Assumptions:\n• Probabilistic latent variables\n• Gaussian noise model\n• Approximate reconstruction\n  (handles uncertainty)', 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray'))

plt.tight_layout()
plt.savefig('./outputs/images/pca_vs_ppca_diagram.png', dpi=300, bbox_inches='tight')
print(f"✓ Conceptual diagram saved to ./outputs/images/pca_vs_ppca_diagram.png")

# ============================================================================
# PART 2: Missing Data Imputation
# ============================================================================

print("\n\n" + "=" * 80)
print("PART 2: PPCA for Missing Data Imputation")
print("=" * 80)

# Generate synthetic low-rank Gaussian data
print("\nGenerating synthetic low-rank Gaussian data...")
np.random.seed(42)

n_samples_syn = 500
n_features_syn = 50
true_rank = 5

# Generate low-rank data: X = Z @ W^T + noise
Z_true = np.random.randn(n_samples_syn, true_rank)
W_true = np.random.randn(n_features_syn, true_rank)
noise = np.random.randn(n_samples_syn, n_features_syn) * 0.1

X_complete = Z_true @ W_true.T + noise
X_complete = X_complete - np.mean(X_complete, axis=0)  # Center

print(f"Data shape: {X_complete.shape}")
print(f"True rank: {true_rank}")

# Randomly remove 10% of entries
missing_rate = 0.10
n_missing = int(n_samples_syn * n_features_syn * missing_rate)
missing_indices = np.random.choice(n_samples_syn * n_features_syn, 
                                   size=n_missing, replace=False)

# Create missing mask
missing_mask = np.zeros((n_samples_syn, n_features_syn), dtype=bool)
missing_mask.flat[missing_indices] = True

# Create data with missing values
X_missing = X_complete.copy()
X_missing[missing_mask] = np.nan

print(f"Missing entries: {n_missing} ({missing_rate*100:.1f}%)")

# Baseline: Mean imputation
print("\n" + "-" * 80)
print("Baseline: Mean Imputation")
print("-" * 80)

X_mean_imputed = X_missing.copy()
for j in range(n_features_syn):
    col_mean = np.nanmean(X_missing[:, j])
    X_mean_imputed[np.isnan(X_mean_imputed[:, j]), j] = col_mean

# Calculate errors for mean imputation
mean_imp_errors = np.abs(X_complete[missing_mask] - X_mean_imputed[missing_mask])
mean_imp_rmse = np.sqrt(np.mean((X_complete[missing_mask] - X_mean_imputed[missing_mask]) ** 2))
mean_imp_mae = np.mean(mean_imp_errors)

print(f"RMSE: {mean_imp_rmse:.6f}")
print(f"MAE:  {mean_imp_mae:.6f}")

# PPCA imputation
print("\n" + "-" * 80)
print("PPCA Imputation (EM Algorithm)")
print("-" * 80)

# Replace NaN with column means for initialization
X_init = X_missing.copy()
for j in range(n_features_syn):
    col_mean = np.nanmean(X_missing[:, j])
    X_init[np.isnan(X_init[:, j]), j] = col_mean

# Fit PPCA with missing data handling
n_comp_impute = true_rank
ppca_impute = ProbabilisticPCA(n_components=n_comp_impute, max_iter=200, 
                               random_state=42)
ppca_impute.fit(X_init, missing_mask=missing_mask)

# Impute missing values
X_ppca_imputed = ppca_impute.impute(X_init, missing_mask)

# Calculate errors for PPCA imputation
ppca_errors = np.abs(X_complete[missing_mask] - X_ppca_imputed[missing_mask])
ppca_rmse = np.sqrt(np.mean((X_complete[missing_mask] - X_ppca_imputed[missing_mask]) ** 2))
ppca_mae = np.mean(ppca_errors)

print(f"RMSE: {ppca_rmse:.6f}")
print(f"MAE:  {ppca_mae:.6f}")
print(f"Noise variance (σ²): {ppca_impute.sigma2:.6f}")

# Comparison summary
print("\n" + "=" * 80)
print("IMPUTATION RESULTS SUMMARY")
print("=" * 80)

print(f"\nMissing Data: {missing_rate*100:.1f}% ({n_missing} entries)")
print(f"True Rank: {true_rank}")
print(f"PPCA Latent Dimensions: {n_comp_impute}")

print("\n" + "-" * 80)
print("Method              RMSE        MAE      Improvement")
print("-" * 80)
print(f"Mean Imputation    {mean_imp_rmse:.6f}  {mean_imp_mae:.6f}    (baseline)")
print(f"PPCA Imputation    {ppca_rmse:.6f}  {ppca_mae:.6f}    "
      f"{(1 - ppca_rmse/mean_imp_rmse)*100:.1f}% better")
print("-" * 80)

# Visualize imputation results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error distribution
axes[0, 0].hist(mean_imp_errors, bins=50, alpha=0.7, label='Mean Imputation', 
                color='#95a5a6', edgecolor='black')
axes[0, 0].hist(ppca_errors, bins=50, alpha=0.7, label='PPCA Imputation', 
                color='#e74c3c', edgecolor='black')
axes[0, 0].set_xlabel('Absolute Error', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Imputation Error Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend(frameon=True, shadow=True)
axes[0, 0].grid(True, alpha=0.3)

# Scatter: True vs Imputed (Mean)
axes[0, 1].scatter(X_complete[missing_mask], X_mean_imputed[missing_mask], 
                  alpha=0.5, s=10, color='#95a5a6')
axes[0, 1].plot([X_complete[missing_mask].min(), X_complete[missing_mask].max()], 
               [X_complete[missing_mask].min(), X_complete[missing_mask].max()], 
               'r--', linewidth=2, label='Perfect Imputation')
axes[0, 1].set_xlabel('True Values', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Imputed Values', fontsize=12, fontweight='bold')
axes[0, 1].set_title(f'Mean Imputation (RMSE: {mean_imp_rmse:.4f})', 
                    fontsize=14, fontweight='bold')
axes[0, 1].legend(frameon=True, shadow=True)
axes[0, 1].grid(True, alpha=0.3)

# Scatter: True vs Imputed (PPCA)
axes[1, 0].scatter(X_complete[missing_mask], X_ppca_imputed[missing_mask], 
                  alpha=0.5, s=10, color='#e74c3c')
axes[1, 0].plot([X_complete[missing_mask].min(), X_complete[missing_mask].max()], 
               [X_complete[missing_mask].min(), X_complete[missing_mask].max()], 
               'r--', linewidth=2, label='Perfect Imputation')
axes[1, 0].set_xlabel('True Values', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Imputed Values', fontsize=12, fontweight='bold')
axes[1, 0].set_title(f'PPCA Imputation (RMSE: {ppca_rmse:.4f})', 
                    fontsize=14, fontweight='bold')
axes[1, 0].legend(frameon=True, shadow=True)
axes[1, 0].grid(True, alpha=0.3)

# RMSE Comparison bar chart
methods = ['Mean\nImputation', 'PPCA\nImputation']
rmse_values = [mean_imp_rmse, ppca_rmse]
colors_bar = ['#95a5a6', '#e74c3c']

bars = axes[1, 1].bar(methods, rmse_values, color=colors_bar, edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('RMSE', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Imputation Quality Comparison', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, rmse_values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement annotation
improvement = (1 - ppca_rmse/mean_imp_rmse) * 100
axes[1, 1].text(0.5, max(rmse_values) * 0.8, 
               f'{improvement:.1f}% improvement',
               ha='center', fontsize=14, fontweight='bold', 
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('./outputs/images/imputation_results.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Imputation results saved to ./outputs/images/imputation_results.png")