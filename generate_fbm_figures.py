#!/usr/bin/env python3
"""
Generate figures illustrating fBM and mixed fBM for different Hurst parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def simulate_fbm(n, H, T=1.0, seed=None):
    """
    Simulate fractional Brownian motion using Cholesky decomposition.

    Parameters
    ----------
    n : int
        Number of time steps
    H : float
        Hurst parameter in (0, 1)
    T : float
        Terminal time
    seed : int, optional
        Random seed

    Returns
    -------
    t : ndarray
        Time points (length n+1)
    B : ndarray
        fBM path (length n+1), starting at 0
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n
    t = np.linspace(0, T, n + 1)

    # Build covariance matrix of fBM at times t[1], ..., t[n]
    # Cov(B^H_s, B^H_t) = 0.5 * (s^{2H} + t^{2H} - |t-s|^{2H})
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ti, tj = t[i+1], t[j+1]
            cov[i, j] = 0.5 * (ti**(2*H) + tj**(2*H) - abs(ti - tj)**(2*H))

    # Cholesky decomposition
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))  # small regularization

    # Generate fBM
    Z = np.random.randn(n)
    B = np.zeros(n + 1)
    B[1:] = L @ Z

    return t, B


def simulate_bm(n, T=1.0, seed=None):
    """
    Simulate standard Brownian motion.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n
    t = np.linspace(0, T, n + 1)
    dW = np.sqrt(dt) * np.random.randn(n)
    W = np.zeros(n + 1)
    W[1:] = np.cumsum(dW)

    return t, W


def plot_fbm_comparison(save_path='fbm_comparison.png'):
    """
    Plot fBM paths for H = 0.25, 0.5, 0.75 on the same figure.
    """
    n = 1000
    T = 1.0
    seed_base = 42

    fig, ax = plt.subplots(figsize=(10, 5))

    H_values = [0.25, 0.5, 0.75]
    colors = ['C0', 'C1', 'C2']
    labels = [r'$\mathcal{H} = 0.25$ (rough)',
              r'$\mathcal{H} = 0.5$ (Brownian motion)',
              r'$\mathcal{H} = 0.75$ (smooth)']

    for H, color, label in zip(H_values, colors, labels):
        t, B = simulate_fbm(n, H, T, seed=seed_base)
        ax.plot(t, B, color=color, label=label, linewidth=0.8, alpha=0.9)

    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(r'$B^{\mathcal{H}}_t$', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_mixed_fbm(save_path='mixed_fbm_example.png'):
    """
    Plot mixed fBM: S_t = W_t + alpha * B^H_t for different alpha values.
    """
    n = 1000
    T = 1.0
    H = 0.75
    seed_W = 42
    seed_B = 123

    # Simulate W and B^H
    t, W = simulate_bm(n, T, seed=seed_W)
    _, B = simulate_fbm(n, H, T, seed=seed_B)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: Individual components
    ax = axes[0]
    ax.plot(t, W, 'C0', label=r'$W_t$ (BM)', linewidth=0.8)
    ax.plot(t, B, 'C2', label=r'$B^{0.75}_t$ (fBM)', linewidth=0.8)
    ax.set_xlabel(r'$t$', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Components', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T])

    # Plot 2: Mixed fBM with alpha = 1
    ax = axes[1]
    alpha = 1.0
    S = W + alpha * B
    ax.plot(t, S, 'C1', linewidth=0.8)
    ax.set_xlabel(r'$t$', fontsize=11)
    ax.set_title(r'$S_t = W_t + B^{0.75}_t$ ($\alpha=1$)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T])

    # Plot 3: Mixed fBM with alpha = 5
    ax = axes[2]
    alpha = 5.0
    S = W + alpha * B
    ax.plot(t, S, 'C3', linewidth=0.8)
    ax.set_xlabel(r'$t$', fontsize=11)
    ax.set_title(r'$S_t = W_t + 5 B^{0.75}_t$ ($\alpha=5$)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, T])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_fbm_increments(save_path='fbm_increments.png'):
    """
    Plot fBM increments to show correlation structure.
    """
    n = 200
    T = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    H_values = [0.25, 0.5, 0.75]
    titles = [r'$\mathcal{H} = 0.25$ (negatively correlated)',
              r'$\mathcal{H} = 0.5$ (independent)',
              r'$\mathcal{H} = 0.75$ (positively correlated)']
    colors = ['C0', 'C1', 'C2']

    for ax, H, title, color in zip(axes, H_values, titles, colors):
        t, B = simulate_fbm(n, H, T, seed=42)
        increments = np.diff(B)
        ax.bar(range(len(increments)), increments, color=color, alpha=0.7, width=1.0)
        ax.set_xlabel('Time step', fontsize=10)
        ax.set_ylabel('Increment', fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlim([0, n])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Generate all figures
    plot_fbm_comparison('fbm_comparison.png')
    plot_mixed_fbm('mixed_fbm_example.png')
    plot_fbm_increments('fbm_increments.png')

    print("\nAll figures generated!")
