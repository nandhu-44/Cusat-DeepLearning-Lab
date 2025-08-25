"""Question 2: PPCA with Missing Data

Implements:
1. Synthetic low-rank dataset generation
2. Random 10% masking to simulate missing values
3. PPCA-EM that handles missing data per sample
4. Imputation of missing entries and RMSE computation

Usage:
    python -m q2.main --n-samples 1000 --dim 50 --latent 8 --missing-frac 0.1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional
import numpy as np
from pathlib import Path


def generate_synthetic_data(n_samples: int, dim: int, latent_dim: int, noise_std: float = 0.1, seed: int = 0):
    """Generate synthetic low-rank data X = Z W^T + mu + noise."""
    rng = np.random.default_rng(seed)
    
    # True parameters
    W_true = rng.normal(scale=0.5, size=(dim, latent_dim))
    mu_true = rng.normal(scale=0.2, size=dim)
    
    # Latent variables and observations
    Z = rng.normal(size=(n_samples, latent_dim))
    X = Z @ W_true.T + mu_true + noise_std * rng.normal(size=(n_samples, dim))
    
    return X, W_true, mu_true


def mask_randomly(X: np.ndarray, missing_fraction: float = 0.1, seed: int = 0):
    """Randomly mask entries in X to simulate missing data."""
    rng = np.random.default_rng(seed)
    mask = np.ones_like(X, dtype=bool)
    n_total = X.size
    n_missing = int(missing_fraction * n_total)
    
    flat_indices = rng.choice(n_total, size=n_missing, replace=False)
    mask.flat[flat_indices] = False
    
    X_masked = X.copy()
    X_masked[~mask] = np.nan
    
    return X_masked, mask


@dataclass
class PPCAMissing:
    """PPCA with missing data support using EM algorithm."""
    n_components: int
    max_iter: int = 300
    tol: float = 1e-4
    random_state: Optional[int] = None
    verbose: bool = False
    
    # Learned parameters
    W: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    sigma2: Optional[float] = None
    n_iter_: int = 0
    converged_: bool = False
    
    def _init_params(self, X: np.ndarray, mask: np.ndarray):
        """Initialize parameters using observed data only."""
        rng = np.random.default_rng(self.random_state)
        n_samples, dim = X.shape
        
        # Initialize mu as mean of observed values per feature
        self.mu = np.array([
            X[mask[:, d], d].mean() if mask[:, d].any() else 0.0
            for d in range(dim)
        ])
        
        # Small random initialization for W
        self.W = 0.01 * rng.standard_normal((dim, self.n_components))
        
        # Initialize sigma2 using observed variance
        obs_data = X[mask]
        if len(obs_data) > 0:
            self.sigma2 = np.var(obs_data) / 10.0
        else:
            self.sigma2 = 0.1
    
    def fit(self, X: np.ndarray, mask: Optional[np.ndarray] = None):
        """Fit PPCA model to data with missing values."""
        X = np.asarray(X, dtype=float)
        
        if mask is None:
            mask = ~np.isnan(X)
        else:
            mask = mask.astype(bool)
        
        n_samples, dim = X.shape
        q = self.n_components
        
        if q >= dim:
            raise ValueError("n_components must be < data dimensionality")
        
        self._init_params(X, mask)
        W, mu, sigma2 = self.W, self.mu, self.sigma2
        
        for iteration in range(1, self.max_iter + 1):
            # E-step: compute sufficient statistics
            S_zz = np.zeros((q, q))
            S_xz = np.zeros((dim, q))
            sigma_residual_sum = 0.0
            total_observed = 0
            
            for n in range(n_samples):
                obs_indices = np.where(mask[n])[0]
                if len(obs_indices) == 0:
                    continue
                
                # Observed data for this sample
                x_obs = X[n, obs_indices]
                mu_obs = mu[obs_indices]
                W_obs = W[obs_indices, :]  # (|O|, q)
                
                # Posterior moments
                M_n = W_obs.T @ W_obs + sigma2 * np.eye(q)
                M_n_inv = np.linalg.inv(M_n)
                
                x_centered = x_obs - mu_obs
                Ez_n = M_n_inv @ W_obs.T @ x_centered
                Ezz_n = sigma2 * M_n_inv + np.outer(Ez_n, Ez_n)
                
                # Accumulate sufficient statistics
                S_zz += Ezz_n
                S_xz[obs_indices, :] += np.outer(x_centered, Ez_n)
                
                # Residual for sigma2 update
                reconstruction = W_obs @ Ez_n
                residual = x_centered - reconstruction
                sigma_residual_sum += np.sum(residual**2)
                sigma_residual_sum += sigma2 * np.trace(W_obs @ M_n_inv @ W_obs.T)
                total_observed += len(obs_indices)
            
            # M-step: update parameters
            if np.linalg.det(S_zz) > 1e-10:
                S_zz_inv = np.linalg.inv(S_zz)
                W_new = S_xz @ S_zz_inv
            else:
                W_new = W  # Keep previous if singular
            
            sigma2_new = sigma_residual_sum / total_observed if total_observed > 0 else sigma2
            
            # Update mu as mean of observed values per feature
            mu_new = np.array([
                X[mask[:, d], d].mean() if mask[:, d].any() else mu[d]
                for d in range(dim)
            ])
            
            # Check convergence
            W_change = np.max(np.abs(W_new - W)) if W is not None else np.inf
            
            W, mu, sigma2 = W_new, mu_new, sigma2_new
            
            if self.verbose and iteration % 20 == 0:
                print(f"[PPCAMissing] iter={iteration} max|dW|={W_change:.3e} sigma2={sigma2:.3e}")
            
            if W_change < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration
                break
        else:
            self.n_iter_ = self.max_iter
        
        self.W, self.mu, self.sigma2 = W, mu, sigma2
        return self
    
    def impute(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Impute missing values using the fitted model."""
        if self.W is None:
            raise RuntimeError("Model not fitted")
        
        X = np.asarray(X, dtype=float)
        if mask is None:
            mask = ~np.isnan(X)
        else:
            mask = mask.astype(bool)
        
        n_samples, dim = X.shape
        q = self.n_components
        W, mu, sigma2 = self.W, self.mu, self.sigma2
        
        X_imputed = X.copy()
        
        for n in range(n_samples):
            obs_indices = np.where(mask[n])[0]
            miss_indices = np.where(~mask[n])[0]
            
            if len(obs_indices) == 0 or len(miss_indices) == 0:
                # If no observed or no missing, skip
                continue
            
            # Compute posterior mean of latent variables
            W_obs = W[obs_indices, :]
            x_obs = X[n, obs_indices]
            mu_obs = mu[obs_indices]
            
            M_n = W_obs.T @ W_obs + sigma2 * np.eye(q)
            M_n_inv = np.linalg.inv(M_n)
            Ez_n = M_n_inv @ W_obs.T @ (x_obs - mu_obs)
            
            # Impute missing values
            X_imputed[n, miss_indices] = mu[miss_indices] + W[miss_indices, :] @ Ez_n
        
        return X_imputed


def compute_imputation_rmse(X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray) -> float:
    """Compute RMSE on imputed (missing) entries only."""
    missing_mask = ~mask
    if not missing_mask.any():
        return 0.0
    
    true_missing = X_true[missing_mask]
    imputed_missing = X_imputed[missing_mask]
    
    return float(np.sqrt(np.mean((true_missing - imputed_missing)**2)))


def run_experiment(n_samples: int, dim: int, latent_dim: int, missing_frac: float, 
                  noise_std: float, seed: int, verbose: bool = False):
    """Run the complete missing data experiment."""
    print(f"Generating synthetic data: {n_samples} samples, {dim}D, latent={latent_dim}")
    X_true, W_true, mu_true = generate_synthetic_data(
        n_samples, dim, latent_dim, noise_std, seed
    )
    
    print(f"Masking {missing_frac*100:.1f}% of entries randomly")
    X_masked, mask = mask_randomly(X_true, missing_frac, seed)
    
    print("Fitting PPCA with missing data...")
    model = PPCAMissing(
        n_components=latent_dim,
        max_iter=200,
        tol=1e-4,
        random_state=seed,
        verbose=verbose
    )
    model.fit(X_masked, mask)
    
    print("Imputing missing values...")
    X_imputed = model.impute(X_masked, mask)
    
    # Compute RMSE on missing entries
    rmse = compute_imputation_rmse(X_true, X_imputed, mask)
    
    print(f"\nResults:")
    print(f"Converged: {model.converged_} (iterations: {model.n_iter_})")
    print(f"Imputation RMSE on missing entries: {rmse:.4f}")
    print(f"Final sigma2: {model.sigma2:.4f}")
    
    # Compare learned vs true parameters (up to rotation)
    if W_true.shape == model.W.shape:
        # Simple correlation-based similarity (not accounting for rotation)
        W_corr = np.abs(np.corrcoef(W_true.flatten(), model.W.flatten())[0, 1])
        print(f"W correlation with true (|corr|): {W_corr:.3f}")
    
    return rmse


def parse_args():
    parser = argparse.ArgumentParser(description="PPCA with Missing Data Experiment")
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--dim', type=int, default=50, help='Data dimensionality')
    parser.add_argument('--latent', type=int, default=8, help='Latent dimensionality')
    parser.add_argument('--missing-frac', type=float, default=0.1, help='Fraction of missing data')
    parser.add_argument('--noise-std', type=float, default=0.1, help='Noise standard deviation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    return parser.parse_args()


def main():
    args = parse_args()
    
    rmse = run_experiment(
        n_samples=args.n_samples,
        dim=args.dim,
        latent_dim=args.latent,
        missing_frac=args.missing_frac,
        noise_std=args.noise_std,
        seed=args.seed,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
