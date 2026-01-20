"""
Visualization utilities for constrained SPD decompositions.

Kept separate so you can add more visualizations without importing matplotlib
into the solver/core modules.
"""


import numpy as np
import matplotlib.pyplot as plt
from constrained_decomposition_core import spd_inverse


def plot_decomposition_heatmaps(A, B, C, basis, add_title=True, out_file=None, show=False, residual_on="B"):
    """
    Plot 2x2 heatmaps:
        top-left:  A
        top-right: C
        bottom-left:  B
        bottom-right: B^{-1}

    Title includes:
        - Reconstruction error
        - Sum of (Bkk - Bk,k+1)
        - Max |Bkk - Bk,k+1|
        - Phi = -log det(A - C)

    Everything printed with 3 significant digits.
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)

    # Compute inverse and reconstruction error
    Binv = spd_inverse(B)
    A_reconstructed = Binv + C
    err = np.linalg.norm(A - A_reconstructed, ord="fro")

    # Compute gradient diagnostic g_k = Bkk - Bk,k+1
    n = B.shape[0]

    g = None
    if basis is not None and hasattr(basis, "trace_with"):
        try:
            if str(residual_on).upper().startswith("C"):
                g = basis.trace_with(C)
            else:
                g = basis.trace_with(B)
        except Exception:
            g = None

    if g is not None:
        sum_g = np.sum(g)
        max_g = np.max(np.abs(g))
    else:
        sum_g = np.nan
        max_g = np.nan

    # Compute phi = -log det(A - C)
    M = A - C
    try:
        L = np.linalg.cholesky(M)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        phi = -logdet
    except np.linalg.LinAlgError:
        phi = np.nan  # If not SPD, logdet undefined

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    matrices = [[A, C],
                [B, Binv]]
    titles = [["A", "C"],
              ["B", r"$B^{-1}$"]]

    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            im = ax.imshow(matrices[r][c], aspect="equal")
            ax.set_title(titles[r][c])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Title: small font and multiple diagnostics ---
    if add_title:
        fig.suptitle(
            rf"Decomposition heatmaps"
            rf"\n$\|A-(B^{{-1}}+C)\|_F = {err:.3g}$"
            rf",  $\sum g_k = {sum_g:.3g}$"
            rf",  $\max |g_k| = {max_g:.3g}$"
            rf",  $\phi = {phi:.3g}$",
            fontsize=9
        )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if out_file is not None:
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig, axes
