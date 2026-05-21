"""
Numerical example for the companion paper "Exponential Utility Maximization
with Quadratic Costs in a Discrete-Time Gaussian Framework" (Dolinsky-Zuk).

Setup: discretized fractional Brownian motion (fBm) increments as the asset
price covariance, with the Almgren-Chriss tridiagonal permanent-impact
quadratic cost matrix
    A_{ii}   =  lambda,
    A_{i,i+/-1} = -lambda/2.

For mu = 0 the optimal exponential-utility value is
    V* = -1 / sqrt( |Sigma| * |S_U(lambda)| ),
where S_U is the SPD matrix of Theorem 1(i). The certainty-equivalent
log-value is therefore
    CE(lambda, H) = 0.5 * ( log|Sigma| + log|S_U(lambda)| ).

This script sweeps lambda on a log scale for several Hurst parameters and
plots CE vs lambda, illustrating the trade-off between profiting from fBm
autocorrelation (H != 1/2) and quadratic transaction costs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from triangular_decomposition import backward_triangular_elimination_efficient
from constrained_decomposition_matrices import spd_fractional_BM


def almgren_chriss_tridiagonal(n: int, lam: float = 1.0) -> np.ndarray:
    """
    Almgren-Chriss tridiagonal permanent-impact cost matrix.

    A_{ii} = 2*lambda for all i, A_{i, i+/-1} = -lambda, zero elsewhere.
    This corresponds to the cost (1/2) * gamma^T A gamma
    = (lambda/2) * sum_{i=1}^{n+1} (gamma_i - gamma_{i-1})^2
    with gamma_0 = gamma_{n+1} = 0.
    Positive definite for every lambda > 0 and n >= 1 (eigenvalues lie in
    (0, 4*lambda)).
    """
    A = 2.0 * lam * np.eye(n)
    off = -lam
    i = np.arange(n - 1)
    A[i, i + 1] = off
    A[i + 1, i] = off
    return A


def certainty_equivalent(A: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Centered (mu=0) certainty-equivalent value
        CE = 0.5 * ( log|Sigma| + log|S_U| ).

    Returns NaN if the algorithm cannot return a valid SPD S_U.
    """
    Lam = np.linalg.inv(Sigma)
    U, _, info = backward_triangular_elimination_efficient(A, Lam)
    if U is None:
        return float("nan")
    S_U = Lam - U - U.T - U @ A @ U.T
    sign_S, logdet_S = np.linalg.slogdet(S_U)
    sign_Sig, logdet_Sig = np.linalg.slogdet(Sigma)
    if sign_S <= 0 or sign_Sig <= 0:
        return float("nan")
    return 0.5 * (logdet_Sig + logdet_S)


def sweep(n: int, H_values: Iterable[float],
          lambda_values: Iterable[float]) -> pd.DataFrame:
    rows = []
    for H in H_values:
        Sigma = spd_fractional_BM(n, H=H, T=1.0, diff_flag=True)
        for lam in lambda_values:
            A = almgren_chriss_tridiagonal(n, lam=lam)
            CE = certainty_equivalent(A, Sigma)
            rows.append({"H": H, "lambda": lam, "CE": CE})
    return pd.DataFrame(rows)


def plot_value_vs_lambda(df: pd.DataFrame, n: int, output_path: Path) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    H_values = sorted(df["H"].unique())
    cmap = plt.get_cmap("coolwarm")
    H_min = min(H_values)
    H_max = max(H_values)
    H_span = max(H_max - H_min, 1e-12)
    for H in H_values:
        sub = df[df["H"] == H].sort_values("lambda")
        if abs(H - 0.5) < 1e-9:
            color = "0.25"  # dark neutral grey for the Brownian curve
        else:
            color = cmap((H - H_min) / H_span)
        ax.plot(sub["lambda"], sub["CE"], color=color,
                marker="o", markersize=3.5, linewidth=1.4,
                label=fr"$H={H:g}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\frac{1}{2}(\log|\Sigma|+\log|S_U|)$")
    ax.set_title(rf"Optimal value vs. quadratic cost magnitude ($n={n}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--H", type=float, nargs="+",
                    default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--lambda-min", type=float, default=1e-3)
    ap.add_argument("--lambda-max", type=float, default=1e2)
    ap.add_argument("--n-lambda", type=int, default=50)
    ap.add_argument("--output-dir", type=Path, default=Path("figs"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    lambda_values = np.logspace(
        np.log10(args.lambda_min), np.log10(args.lambda_max), args.n_lambda
    )

    print(f"Sweeping n={args.n}, H={args.H}, "
          f"lambda in [{args.lambda_min:g}, {args.lambda_max:g}] ({args.n_lambda} pts)")
    df = sweep(args.n, args.H, lambda_values)

    csv_path = args.output_dir / f"fbm_utility_n{args.n}.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    pdf_path = args.output_dir / f"fbm_utility_n{args.n}.pdf"
    plot_value_vs_lambda(df, args.n, pdf_path)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
