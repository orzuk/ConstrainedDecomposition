"""
Two additional numerical figures for the utility_quadratic paper:

  A) Certainty-equivalent value as a function of Hurst parameter H,
     for several fixed cost magnitudes lambda. (Transpose of the existing
     "CE vs lambda" plot — emphasises the autocorrelation dependence.)

  B) Optimal trading strategy gamma_i = -(U^T X)_i vs time-step index i,
     on one sample path of X, for several lambda. Shows the smoothing
     effect of quadratic costs on the optimal strategy.

Both are produced in the centered case mu = 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from triangular_decomposition import backward_triangular_elimination_efficient
from constrained_decomposition_matrices import spd_fractional_BM


def almgren_chriss_tridiagonal(n: int, lam: float = 1.0) -> np.ndarray:
    A = 2.0 * lam * np.eye(n)
    off = -lam
    i = np.arange(n - 1)
    A[i, i + 1] = off
    A[i + 1, i] = off
    return A


def ce_value(A: np.ndarray, Sigma: np.ndarray) -> float:
    Lam = np.linalg.inv(Sigma)
    U, _, _ = backward_triangular_elimination_efficient(A, Lam)
    if U is None:
        return float("nan")
    S_U = Lam - U - U.T - U @ A @ U.T
    sign_S, logdet_S = np.linalg.slogdet(S_U)
    sign_Sig, logdet_Sig = np.linalg.slogdet(Sigma)
    if sign_S <= 0 or sign_Sig <= 0:
        return float("nan")
    return 0.5 * (logdet_Sig + logdet_S)


def optimal_U(A: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """For A SPD, solve Lambda = L + U + UAL.
    For A = 0 this degenerates to U = strict-upper(Lambda)."""
    Lam = np.linalg.inv(Sigma)
    if np.allclose(A, 0):
        U = np.triu(Lam, k=1)
        return U
    U, _, _ = backward_triangular_elimination_efficient(A, Lam)
    return U


def ce_value_general(A: np.ndarray, Sigma: np.ndarray) -> float:
    """CE = 0.5*(log|Sigma| + log|S_U|), with S_U = Lambda - U - U^T - U A U^T."""
    Lam = np.linalg.inv(Sigma)
    U = optimal_U(A, Sigma)
    S_U = Lam - U - U.T - U @ A @ U.T
    sign_S, logdet_S = np.linalg.slogdet(S_U)
    sign_Sig, logdet_Sig = np.linalg.slogdet(Sigma)
    if sign_S <= 0 or sign_Sig <= 0:
        return float("nan")
    return 0.5 * (logdet_Sig + logdet_S)


def plot_ce_vs_H(output_path: Path, n: int = 100,
                 lambda_values=(0.01, 0.1, 1.0, 10.0),
                 H_values=None) -> None:
    if H_values is None:
        H_values = np.linspace(0.05, 0.95, 19)
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    cmap = plt.get_cmap("coolwarm")
    lam_min, lam_max = min(lambda_values), max(lambda_values)
    log_span = max(np.log10(lam_max) - np.log10(lam_min), 1e-12)

    for lam in lambda_values:
        ces = []
        for H in H_values:
            Sigma = spd_fractional_BM(n, H=H, T=1.0, diff_flag=True)
            A = almgren_chriss_tridiagonal(n, lam=lam)
            ces.append(ce_value(A, Sigma))
        t = (np.log10(lam) - np.log10(lam_min)) / log_span
        color = cmap(t)
        ax.plot(H_values, ces, color=color, marker="o", markersize=3.5,
                linewidth=1.4, label=fr"$\lambda={lam:g}$")

    ax.axvline(0.5, color="0.5", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$H$")
    ax.set_ylabel(r"$\frac{1}{2}(\log|\Sigma|+\log|S_U|)$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_strategy_trajectory(output_path: Path, n: int = 200,
                             H_values=(0.3, 0.7),
                             lambda_values=(0.0, 0.001, 0.01, 0.1, 1.0, 10.0),
                             seed: int = 0) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    ncols = len(H_values)
    fig, axes = plt.subplots(2, ncols, figsize=(6.0 * ncols, 6.0),
                             sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})
    if ncols == 1:
        axes = axes.reshape(2, 1)

    cmap = plt.get_cmap("coolwarm")
    pos_lams = [lam for lam in lambda_values if lam > 0]
    log_span = max(np.log10(max(pos_lams)) - np.log10(min(pos_lams)), 1e-12) \
        if len(pos_lams) > 1 else 1.0
    lam_log_min = np.log10(min(pos_lams)) if pos_lams else 0.0
    idx = np.arange(1, n + 1)

    for col, H in enumerate(H_values):
        rng = np.random.default_rng(seed)
        Sigma = spd_fractional_BM(n, H=H, T=1.0, diff_flag=True)
        L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(n))
        X = L @ rng.standard_normal(n)

        ax_top = axes[0, col]
        ax_top.plot(idx, X, color="k", linewidth=0.6)
        ax_top.axhline(0, color="0.5", linewidth=0.5)
        ax_top.set_title(fr"$H={H:g}$", fontsize=13)
        ax_top.grid(True, alpha=0.3)
        if col == 0:
            ax_top.set_ylabel(r"$X_i$")

        # Precompute CE values so each legend column can be right-aligned to
        # its own max width (so wider numbers don't force padding on shorter
        # ones in the OTHER column).
        half = (len(lambda_values) + 1) // 2
        ces_all = []
        for lam in lambda_values:
            A_tmp = almgren_chriss_tridiagonal(n, lam=lam) if lam > 0 \
                else np.zeros((n, n))
            ces_all.append(ce_value_general(A_tmp, Sigma))
        max_lam_len_left = max(len(f"{lam:g}") for lam in lambda_values[:half])
        max_lam_len_right = max(len(f"{lam:g}") for lam in lambda_values[half:])
        max_ce_len_left = max(len(f"{ce:.2f}") for ce in ces_all[:half])
        max_ce_len_right = max(len(f"{ce:.2f}") for ce in ces_all[half:])
        # All solid; lambda=0 drawn thinner so an overlapping lambda=0.001
        # on top of it still shows clearly.
        linestyles = ["-"] * len(lambda_values)
        linewidths = [0.6 if lam == 0 else 1.0 for lam in lambda_values]
        # Skip the pale middle of coolwarm so all 6 colors stay saturated.
        n_lam_local = len(lambda_values)
        half = n_lam_local // 2
        color_pos = (list(np.linspace(0.0, 0.30, half))
                     + list(np.linspace(0.70, 1.0, n_lam_local - half)))

        ax_bot = axes[1, col]
        curves = []
        for j, lam in enumerate(lambda_values):
            A = almgren_chriss_tridiagonal(n, lam=lam) if lam > 0 \
                else np.zeros((n, n))
            U = optimal_U(A, Sigma)
            gamma = -U.T @ X
            ce = ce_value_general(A, Sigma)
            color = cmap(color_pos[j])
            ls = linestyles[j]
            lw = linewidths[j]
            if j < half:
                lam_str = f"{lam:g}".rjust(max_lam_len_left)
                ce_str = f"{ce:.2f}".rjust(max_ce_len_left)
            else:
                lam_str = f"{lam:g}".rjust(max_lam_len_right)
                ce_str = f"{ce:.2f}".rjust(max_ce_len_right)
            label = rf"$\lambda$={lam_str} (CE={ce_str})"
            curves.append((gamma, color, ls, lw, label))

        # Plot all curves, keep handles to control legend ordering
        handles = []
        labels = []
        for gamma, color, ls, lw, label in curves:
            line, = ax_bot.plot(idx, gamma, color=color, linestyle=ls,
                                linewidth=lw, label=label)
            handles.append(line)
            labels.append(label)
        ax_bot.axhline(0, color="0.5", linewidth=0.5)
        ax_bot.set_xlabel(r"$i$")
        ax_bot.grid(True, alpha=0.3)

        # Place legend just below the lowest curve over the FULL i-range
        # (legend now spans most of figure width, so we no longer restrict
        # to the right third). The legend's pixel height is fixed by
        # font/entry-count; in data units it scales with the y-axis range,
        # so we solve self-consistently for new y_min so the legend bottom
        # sits exactly 1mm above the axis bottom.
        all_min = min(g.min() for g, *_ in curves)
        all_max = max(g.max() for g, *_ in curves)
        data_range = all_max - all_min
        mm = 0.01 * data_range             # ~1mm in data units
        gap = -mm                          # legend top 1mm ABOVE global min
                                           # (lifted ~2mm vs previous +1mm)
        bottom_pad = 4 * mm                # ~4mm below legend (legend was
                                           # clipping with 1mm padding)
        top_margin = 0.03 * data_range
        new_y_max = all_max + top_margin
        # 2-column legend with 3 rows of monospace fontsize 9:
        # 3 lines ~ 0.5 in in a ~4 in panel.
        r = 0.16
        new_y_min_legend = (all_min - gap - bottom_pad - r * new_y_max) \
            / (1.0 - r)
        new_y_min = min(all_min - top_margin, new_y_min_legend)
        ax_bot.set_ylim(new_y_min, new_y_max)

        # matplotlib's ncol fills column-major by default, so natural order
        # gives col1 = [0, 0.001, 0.01], col2 = [0.1, 1, 10].
        ax_bot.legend(handles, labels, ncol=2, frameon=False,
                      loc="upper right",
                      bbox_to_anchor=(n, all_min - gap),
                      bbox_transform=ax_bot.transData,
                      prop={"family": "monospace", "size": 9},
                      columnspacing=1.5, handlelength=2.0)
        if col == 0:
            ax_bot.set_ylabel(r"$\hat\gamma_i$")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_decay_alignment(output_path: Path, n: int = 100,
                         H_values=(0.25, 0.5, 0.75),
                         lam: float = 0.1,
                         alpha_values=None,
                         eps: float = 0.01) -> None:
    """CE as a function of the cost kernel's decay rate alpha.

    Physical cost family: graph Laplacian with exponentially decaying
    edge weights. Define W_alpha[i,j] = exp(-alpha*|i-j|) for i != j,
    D_alpha = diag(W_alpha @ 1) + eps*I, and
        A_alpha = ( D_alpha - W_alpha ) * (2*n*lambda / trace).

    The quadratic form is
        gamma^T A_alpha gamma = (scale/2) * sum_{i != j} W_alpha[i,j]
                                * (gamma_i - gamma_j)^2 + eps*scale*||gamma||^2,
    so the cost penalises DIFFERENCES between trading positions (smooth
    strategies cheap), as Almgren-Chriss does. The decay rate alpha
    controls how far in time the smoothing penalty reaches:

    - alpha large : only nearest-neighbour differences matter -> AC-like.
    - alpha small : long-range differences penalised -> kernel-propagator-like.

    Trace is held fixed at 2*n*lambda so what varies is structure, not
    magnitude.
    """
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    if alpha_values is None:
        alpha_values = np.logspace(-2, 1.5, 30)  # 0.01 .. ~31

    cmap = plt.get_cmap("coolwarm")
    H_min, H_max = min(H_values), max(H_values)
    H_span = max(H_max - H_min, 1e-12)

    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    I_n = np.eye(n)
    for H in H_values:
        Sigma = spd_fractional_BM(n, H=H, T=1.0, diff_flag=True)
        ces = []
        for alpha in alpha_values:
            W = np.exp(-alpha * d)
            np.fill_diagonal(W, 0.0)
            D = np.diag(W.sum(axis=1)) + eps * I_n
            A_un = D - W
            scale = (2.0 * n * lam) / np.trace(A_un)
            A = scale * A_un
            ces.append(ce_value_general(A, Sigma))
        color = cmap((H - H_min) / H_span)
        ax.plot(alpha_values, ces, color=color, marker="o", markersize=3.5,
                linewidth=1.4, label=fr"$H={H:g}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$ (cost decay rate)")
    ax.set_ylabel(r"$\frac{1}{2}(\log|\Sigma|+\log|S_U|)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_alignment_effect(output_path: Path, n: int = 100,
                          H_values=(0.25, 0.75),
                          lambda_values=(0.01, 0.1, 1.0),
                          n_rho: int = 21) -> None:
    """CE as a function of off-diagonal weight rho, with trace(A) held fixed.

    A_rho := 2*lambda*I_n - lambda*rho*T,  where T_{i,i+1}=T_{i+1,i}=1.

    - rho = 0: pure diagonal cost (penalises |gamma|^2 only).
    - rho = 1: full Almgren-Chriss tridiagonal (penalises position size
      AND time-changes in position).

    The trace of A is 2*n*lambda independent of rho, so this isolates the
    *structural* alignment of A with Sigma from its overall magnitude.
    """
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    rhos = np.linspace(0.0, 1.0, n_rho)
    I_n = np.eye(n)
    T_offdiag = np.zeros((n, n))
    idx = np.arange(n - 1)
    T_offdiag[idx, idx + 1] = 1.0
    T_offdiag[idx + 1, idx] = 1.0

    cmap_blue = plt.get_cmap("Blues")
    cmap_red = plt.get_cmap("Reds")
    pos_lams = sorted(lambda_values)
    n_lam = len(pos_lams)

    linestyles = ["-", "--", ":"]
    for j, lam in enumerate(pos_lams):
        for H in H_values:
            Sigma = spd_fractional_BM(n, H=H, T=1.0, diff_flag=True)
            ces = []
            for rho in rhos:
                A = 2.0 * lam * I_n - lam * rho * T_offdiag
                ces.append(ce_value_general(A, Sigma))
            shade = 0.45 + 0.45 * j / max(1, n_lam - 1)
            color = cmap_blue(shade) if H < 0.5 else cmap_red(shade)
            ls = linestyles[j % len(linestyles)]
            ax.plot(rhos, ces, color=color, linestyle=ls,
                    marker="o", markersize=3.5, linewidth=1.4,
                    label=fr"$H={H:g}$, $\lambda={lam:g}$")

    ax.set_xlabel(r"$\rho$ (off-diagonal weight)")
    ax.set_ylabel(r"$\frac{1}{2}(\log|\Sigma|+\log|S_U|)$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="best",
              prop={"family": "monospace", "size": 8})
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    out_dir = Path("figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_ce_vs_H(out_dir / "fbm_utility_ce_vs_H.pdf")
    plot_strategy_trajectory(out_dir / "fbm_utility_strategy_trajectory.pdf")
    plot_alignment_effect(out_dir / "fbm_utility_alignment.pdf")
    plot_decay_alignment(out_dir / "fbm_utility_decay_alignment.pdf")
