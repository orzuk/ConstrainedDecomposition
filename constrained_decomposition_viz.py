"""
Visualization utilities for constrained SPD decompositions.

Kept separate so you can add more visualizations without importing matplotlib
into the solver/core modules.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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


def plot_block_decomposition(A, B, C, blocks, active_blocks=None,
                              out_file=None, show=False, figsize=(12, 8)):
    """
    Plot block-structured decomposition with clear visualization:
    - Shows A, C, B, B^{-1} in 2x2 grid
    - Draws block boundaries
    - Highlights active vs inactive blocks in C
    - Uses appropriate colormaps

    Parameters
    ----------
    A, B, C : np.ndarray
        The decomposition: A = B^{-1} + C
    blocks : list of list of int
        Block partition
    active_blocks : list of (i, j) tuples, optional
        Which block pairs are active (C can be nonzero there)
    out_file : str, optional
        Path to save figure
    show : bool
        Whether to display the figure
    figsize : tuple
        Figure size
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    n = A.shape[0]
    r = len(blocks)

    # Compute B^{-1} and reconstruction error
    Binv = spd_inverse(B)
    err = np.linalg.norm(A - (Binv + C), ord="fro")

    # Block boundaries (cumulative sizes)
    block_bounds = [0]
    for blk in blocks:
        block_bounds.append(block_bounds[-1] + len(blk))

    def add_block_lines(ax, color='white', linewidth=1.5, linestyle='-'):
        """Add block boundary lines to an axis."""
        for b in block_bounds[1:-1]:  # Skip first (0) and last (n)
            ax.axhline(y=b - 0.5, color=color, linewidth=linewidth, linestyle=linestyle)
            ax.axvline(x=b - 0.5, color=color, linewidth=linewidth, linestyle=linestyle)

    def highlight_inactive_blocks(ax, alpha=0.3):
        """Overlay hatching on inactive blocks in C."""
        if active_blocks is None:
            return
        active_set = set(active_blocks)
        for bi in range(r):
            for bj in range(r):
                # Normalize to (min, max) order for comparison
                block_pair = (min(bi, bj), max(bi, bj))
                if block_pair not in active_set:
                    # This block is inactive - add hatching
                    i_start = block_bounds[bi]
                    i_end = block_bounds[bi + 1]
                    j_start = block_bounds[bj]
                    j_end = block_bounds[bj + 1]
                    rect = Rectangle(
                        (j_start - 0.5, i_start - 0.5),
                        j_end - j_start, i_end - i_start,
                        linewidth=0, edgecolor='none',
                        facecolor='gray', alpha=alpha,
                        hatch='///'
                    )
                    ax.add_patch(rect)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # SHARED colormap for A, C, B^{-1} (they represent the same values)
    # Use vmin=0 so that 0 appears as white (inactive blocks in C)
    all_vals = np.concatenate([A.flatten(), C.flatten(), Binv.flatten()])
    vmin_shared = 0.0  # Force 0 to be white
    vmax_shared = np.max(all_vals)

    # B has its own colormap (different scale - precision matrix)
    vmin_B, vmax_B = np.min(B), np.max(B)

    # Grayscale colormap: 0=white, higher=darker (more visible differences)
    cmap_shared = 'gray_r'  # Reversed gray: white=0, black=max
    line_color = 'red'      # Red lines visible on grayscale

    # Top-left: A (shared colormap)
    ax = axes[0, 0]
    im = ax.imshow(A, cmap=cmap_shared, aspect='equal', vmin=vmin_shared, vmax=vmax_shared)
    ax.set_title(r'$A$ (input)', fontsize=12)
    add_block_lines(ax, color=line_color, linewidth=1.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Top-right: C (shared colormap, with inactive block highlighting)
    ax = axes[0, 1]
    im = ax.imshow(C, cmap=cmap_shared, aspect='equal', vmin=vmin_shared, vmax=vmax_shared)
    ax.set_title(r'$C \in S$ (block-constant, zero diagonal)', fontsize=12)
    add_block_lines(ax, color=line_color, linewidth=1.5)
    highlight_inactive_blocks(ax, alpha=0.3)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom-left: B (own colormap - precision matrix has different scale)
    ax = axes[1, 0]
    im = ax.imshow(B, cmap='plasma', aspect='equal', vmin=vmin_B, vmax=vmax_B)
    ax.set_title(r'$B \in S^\perp$ (precision)', fontsize=12)
    add_block_lines(ax, color='white', linewidth=1.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom-right: B^{-1} with colormap focused on OFF-DIAGONAL range
    # This makes the active vs inactive cross-block difference visible
    ax = axes[1, 1]
    # Get off-diagonal values only (exclude diagonal)
    Binv_offdiag = Binv.copy()
    np.fill_diagonal(Binv_offdiag, np.nan)
    offdiag_min = np.nanmin(Binv_offdiag)
    offdiag_max = np.nanmax(Binv_offdiag)
    # Use focused range for better contrast in cross-blocks
    im = ax.imshow(Binv, cmap=cmap_shared, aspect='equal',
                   vmin=offdiag_min - 0.05, vmax=offdiag_max + 0.05)
    ax.set_title(r'$B^{-1}$ (colormap: off-diag range)', fontsize=12)
    add_block_lines(ax, color=line_color, linewidth=1.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Suptitle with key info
    n_active = len(active_blocks) if active_blocks else '?'
    n_total = r + r * (r - 1) // 2  # diagonal + upper-triangular cross-blocks
    fig.suptitle(
        f'Block Decomposition: $A = B^{{-1}} + C$\n'
        f'$n={n}$, $r={r}$ blocks, '
        f'active blocks: {n_active}/{n_total}, '
        f'$\\|A-(B^{{-1}}+C)\\|_F = {err:.2e}$',
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if out_file is not None:
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return fig, axes
