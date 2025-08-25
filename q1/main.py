"""Question 1: PPCA for Image Compression on CIFAR-10.

Downloads CIFAR-10, samples a subset, and compares standard PCA with
Probabilistic PCA (EM implementation) for reconstruction quality.

Example:
    python -m q1.main --sample-size 4000 --latents 32 64 128
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import torch  # noqa: F401 (ensures torchvision backend)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def load_cifar10(sample_size: int, random_state: int = 0):
    tfm = transforms.ToTensor()
    ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
    N = len(ds)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(N, size=min(sample_size, N), replace=False)
    X = []
    for i in idx:
        x, _ = ds[i]
        X.append(x.numpy())  # (3,32,32)
    X = np.stack(X)
    X = X.reshape(X.shape[0], -1)
    return X


@dataclass
class PPCA:
    n_components: int
    max_iter: int = 200
    tol: float = 1e-4
    random_state: Optional[int] = None
    verbose: bool = False

    W: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    sigma2: Optional[float] = None
    n_iter_: int = 0
    converged_: bool = False

    def _init(self, X):
        rng = np.random.default_rng(self.random_state)
        self.mu = X.mean(axis=0)
        self.W = 0.01 * rng.standard_normal((X.shape[1], self.n_components))
        self.sigma2 = np.var(X) / 100

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        N, D = X.shape
        q = self.n_components
        if q >= D:
            raise ValueError("n_components must be < D")
        self._init(X)
        W, mu, sigma2 = self.W, self.mu, self.sigma2
        Iq = np.eye(q)
        prev_ll = -np.inf
        Xc = X - mu
        for it in range(1, self.max_iter + 1):
            M = W.T @ W + sigma2 * Iq
            M_inv = np.linalg.inv(M)
            Ez = (M_inv @ W.T @ Xc.T).T
            Ezz_sum = X.shape[0] * sigma2 * M_inv + Ez.T @ Ez
            W = (Xc.T @ Ez) @ np.linalg.inv(Ezz_sum)
            X_hat_c = Ez @ W.T
            residual = Xc - X_hat_c
            sigma2 = np.sum(residual**2) / (N * D)
            C = W @ W.T + sigma2 * np.eye(D)
            sign, logdetC = np.linalg.slogdet(C)
            if sign <= 0:
                ll = -np.inf
            else:
                invC = np.linalg.inv(C)
                ll = -0.5 * (N * (logdetC + np.trace(invC @ (Xc.T @ Xc) / N)))
            if self.verbose and it % 10 == 0:
                print(f"PPCA iter={it} ll={ll:.2f} sigma2={sigma2:.4f}")
            if ll - prev_ll < self.tol:
                self.converged_ = True
                self.n_iter_ = it
                break
            prev_ll = ll
        else:
            self.n_iter_ = self.max_iter
        self.W, self.mu, self.sigma2 = W, mu, sigma2
        return self

    def transform(self, X):
        W, mu, sigma2 = self.W, self.mu, self.sigma2
        M_inv = np.linalg.inv(W.T @ W + sigma2 * np.eye(self.n_components))
        return (M_inv @ W.T @ (X - mu).T).T

    def reconstruct(self, X):
        Z = self.transform(X)
        return Z @ self.W.T + self.mu


def visualize_samples(X_orig, X_pca, X_ppca, path: Path, n_show=8):
    side = 32
    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 1.3, 3.6))
    sel = np.random.choice(X_orig.shape[0], n_show, replace=False)
    for row, Xv, title in zip(range(3), [X_orig, X_pca, X_ppca], ["Orig", "PCA", "PPCA"]):
        for j, idx in enumerate(sel):
            img = Xv[idx].reshape(3, side, side).transpose(1, 2, 0)
            axes[row, j].imshow(np.clip(img, 0, 1))
            axes[row, j].axis('off')
            if j == 0:
                axes[row, j].set_ylabel(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(latent_dims: Sequence[int], sample_size: int, out_dir: str, ppca_iters: int, seed: int):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print("Loading CIFAR-10 subset ...")
    X = load_cifar10(sample_size=sample_size, random_state=seed)
    print(f"Data shape: {X.shape}")
    results = []
    for q in latent_dims:
        print(f"Latent dim {q}")
        pca = PCA(n_components=q, svd_solver='randomized', random_state=seed)
        X_pca_rec = pca.inverse_transform(pca.fit_transform(X))
        pca_mse = mse(X, X_pca_rec)
        ppca = PPCA(n_components=q, max_iter=ppca_iters, tol=1e-4, random_state=seed, verbose=False)
        ppca.fit(X)
        X_ppca_rec = ppca.reconstruct(X)
        ppca_mse = mse(X, X_ppca_rec)
        results.append((q, pca_mse, ppca_mse))
        visualize_samples(X, X_pca_rec, X_ppca_rec, out_path / f"recon_q{q}.png")
        print(f"  MSE PCA={pca_mse:.5f} PPCA={ppca_mse:.5f}")
    import matplotlib
    matplotlib.use('Agg')
    qs = [r[0] for r in results]
    p_mse = [r[1] for r in results]
    pp_mse = [r[2] for r in results]
    plt.figure(figsize=(5, 3))
    plt.plot(qs, p_mse, '-o', label='PCA')
    plt.plot(qs, pp_mse, '-o', label='PPCA')
    plt.xlabel('Latent dim')
    plt.ylabel('Reconstruction MSE')
    plt.title('CIFAR-10 PCA vs PPCA')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / 'mse_curve.png')
    print('Results: (q, PCA_MSE, PPCA_MSE)')
    for r in results:
        print(r)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latents', type=int, nargs='+', default=[32, 64, 128])
    ap.add_argument('--sample-size', type=int, default=4000)
    ap.add_argument('--ppca-iters', type=int, default=120)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', type=str, default='outputs/q1_cifar10')
    return ap.parse_args()


def main():
    args = parse_args()
    run(args.latents, args.sample_size, args.out_dir, args.ppca_iters, args.seed)


if __name__ == '__main__':
    main()
