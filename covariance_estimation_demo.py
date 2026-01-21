"""
Covariance Estimation Demo: Combining Tree + Block Structure

This demo compares different approaches for estimating a covariance matrix
with additive structure: Sigma = B^{-1} + C where:
  - B^{-1} has tree-structured precision (sparse graphical model)
  - C has block-constant structure (group effects)

Methods compared:
  1. Decomposition-based (ours): Decompose -> Chow-Liu on B' -> Block-avg on C'
  2. EM-type: Iterate between tree fitting and block averaging
  3. Naive: Just use empirical covariance (no structure)
  4. Tree-only: Chow-Liu on full precision (ignores blocks)
  5. Block-only: Block average (ignores tree)

Run:
  python covariance_estimation_demo.py --n 50 --r 5 --samples 200 --seed 42
"""

import argparse
import time
import numpy as np
from scipy import linalg as sp_linalg
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import List, Tuple, Dict, Optional

# Import from existing codebase
from constrained_decomposition_core import (
    constrained_decomposition,
    SymBasis,
    spd_inverse,
    is_spd,
)
from constrained_decomposition_matrices import (
    make_blocks_variable,
    make_block_fixed_basis_offdiag,  # Block-CONSTANT basis (one element per block pair)
)


# =============================================================================
# Tree-structured precision matrix generation
# =============================================================================

def make_random_tree(n: int, seed: int = 0) -> List[Tuple[int, int]]:
    """
    Generate a random tree on n nodes using random edge weights + MST.

    Returns list of edges (i, j) with i < j.
    """
    rng = np.random.default_rng(seed)

    # Create random complete graph weights
    W = rng.uniform(0.1, 1.0, size=(n, n))
    W = (W + W.T) / 2  # symmetrize
    np.fill_diagonal(W, 0)

    # Find minimum spanning tree (gives us a random tree)
    mst = minimum_spanning_tree(W)
    mst = mst.toarray()

    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if mst[i, j] > 0 or mst[j, i] > 0:
                edges.append((i, j))

    return edges


def make_tree_precision(n: int, edges: List[Tuple[int, int]],
                        diag_range: Tuple[float, float] = (2.0, 4.0),
                        edge_range: Tuple[float, float] = (0.3, 0.8),
                        seed: int = 0) -> np.ndarray:
    """
    Create a tree-structured precision matrix (SPD).

    The precision matrix B has:
      - B[i,i] > 0 (diagonal)
      - B[i,j] != 0 only if (i,j) is an edge in the tree

    To ensure SPD, we use diagonal dominance.

    Parameters
    ----------
    n : int
        Matrix dimension
    edges : list of (i, j) tuples
        Tree edges (n-1 edges for n nodes)
    diag_range : tuple
        Range for diagonal entries (before dominance adjustment)
    edge_range : tuple
        Range for |off-diagonal| entries on tree edges
    seed : int
        Random seed

    Returns
    -------
    B : np.ndarray
        n x n tree-structured SPD precision matrix
    """
    rng = np.random.default_rng(seed)

    B = np.zeros((n, n), dtype=float)

    # Set diagonal entries
    B[np.diag_indices(n)] = rng.uniform(*diag_range, size=n)

    # Set edge entries (with random signs)
    for (i, j) in edges:
        val = rng.uniform(*edge_range)
        sign = rng.choice([-1, 1])
        B[i, j] = sign * val
        B[j, i] = sign * val

    # Ensure SPD via diagonal dominance
    for i in range(n):
        off_diag_sum = np.sum(np.abs(B[i, :])) - np.abs(B[i, i])
        B[i, i] = off_diag_sum + rng.uniform(0.5, 1.5)  # margin

    return B


def make_block_constant_C(n: int, blocks: List[List[int]],
                          within_range: Tuple[float, float] = (0.1, 0.3),
                          cross_range: Tuple[float, float] = (0.05, 0.15),
                          seed: int = 0) -> np.ndarray:
    """
    Create a block-constant matrix C with zero diagonal.

    C has:
      - C[i,i] = 0 (zero diagonal, required for S cap SPSD = {0})
      - C[i,j] = constant within each block (off-diagonal)
      - C[i,j] = constant for each cross-block pair

    Parameters
    ----------
    n : int
        Matrix dimension
    blocks : list of list of int
        Partition of {0,...,n-1} into blocks
    within_range : tuple
        Range for within-block off-diagonal constants
    cross_range : tuple
        Range for cross-block constants
    seed : int
        Random seed

    Returns
    -------
    C : np.ndarray
        n x n block-constant symmetric matrix with zero diagonal
    """
    rng = np.random.default_rng(seed)
    r = len(blocks)

    C = np.zeros((n, n), dtype=float)

    # Within-block constants (off-diagonal only)
    for blk in blocks:
        if len(blk) > 1:
            val = rng.uniform(*within_range)
            for i in blk:
                for j in blk:
                    if i != j:
                        C[i, j] = val

    # Cross-block constants
    for bi in range(r):
        for bj in range(bi + 1, r):
            val = rng.uniform(*cross_range)
            for i in blocks[bi]:
                for j in blocks[bj]:
                    C[i, j] = val
                    C[j, i] = val

    return C


def make_true_covariance(n: int, blocks: List[List[int]],
                         tree_edges: List[Tuple[int, int]],
                         seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create true covariance Sigma = B^{-1} + C.

    Returns (Sigma, B, C) where:
      - B is tree-structured precision
      - C is block-constant with zero diagonal
      - Sigma = B^{-1} + C is guaranteed SPD
    """
    rng = np.random.default_rng(seed)

    # Create tree-structured precision B
    B = make_tree_precision(n, tree_edges, seed=seed)

    # Create block-constant C
    C = make_block_constant_C(n, blocks, seed=seed + 100)

    # Compute B^{-1}
    B_inv = spd_inverse(B)

    # Sigma = B^{-1} + C
    Sigma = B_inv + C

    # Ensure Sigma is SPD (should be if C is small enough)
    if not is_spd(Sigma):
        # Scale down C if needed
        for scale in [0.5, 0.25, 0.1, 0.05]:
            Sigma_try = B_inv + scale * C
            if is_spd(Sigma_try):
                C = scale * C
                Sigma = Sigma_try
                print(f"  Scaled C by {scale} to ensure Sigma SPD")
                break
        else:
            raise ValueError("Cannot make Sigma SPD even with scaled C")

    return Sigma, B, C


# =============================================================================
# Chow-Liu algorithm for tree structure estimation
# =============================================================================

def compute_mutual_information_matrix(Sigma: np.ndarray) -> np.ndarray:
    """
    Compute mutual information I(X_i; X_j) for all pairs from covariance.

    For Gaussian: I(X_i; X_j) = -0.5 * log(1 - rho_ij^2)
    where rho_ij is the correlation.
    """
    n = Sigma.shape[0]

    # Compute correlation matrix
    D = np.sqrt(np.diag(Sigma))
    D_inv = 1.0 / D
    Rho = Sigma * np.outer(D_inv, D_inv)

    # Clip correlations to avoid numerical issues
    Rho = np.clip(Rho, -0.9999, 0.9999)

    # Mutual information (negated for minimum spanning tree = maximum weight tree)
    MI = -0.5 * np.log(1 - Rho**2)
    np.fill_diagonal(MI, 0)

    return MI


def chow_liu_tree(Sigma: np.ndarray) -> List[Tuple[int, int]]:
    """
    Chow-Liu algorithm: find optimal tree structure for Gaussian graphical model.

    This finds the maximum-weight spanning tree where weights are mutual information.

    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix

    Returns
    -------
    edges : list of (i, j) tuples
        Tree edges (i < j)
    """
    n = Sigma.shape[0]

    # Compute mutual information as edge weights
    MI = compute_mutual_information_matrix(Sigma)

    # MST on negative weights = max weight spanning tree
    mst = minimum_spanning_tree(-MI)
    mst = mst.toarray()

    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if mst[i, j] != 0 or mst[j, i] != 0:
                edges.append((i, j))

    return edges


def fit_tree_precision(Sigma: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Fit precision matrix with given tree structure via MLE.

    For a tree-structured Gaussian, the MLE precision has:
      - Precision[i,j] = 0 if (i,j) not an edge
      - Entries determined by local 2x2 submatrices

    This uses the fact that for trees, the precision can be computed
    from pairwise marginals.
    """
    n = Sigma.shape[0]

    # Initialize precision
    Lambda = np.zeros((n, n), dtype=float)

    # For a tree, Lambda[i,j] = -Sigma_ij^{-1} for edges
    # and diagonal entries are computed from local structure

    # Build adjacency structure
    adj = {i: [] for i in range(n)}
    for (i, j) in edges:
        adj[i].append(j)
        adj[j].append(i)

    # Compute precision entries for each edge
    for (i, j) in edges:
        # 2x2 marginal covariance
        idx = [i, j]
        Sigma_ij = Sigma[np.ix_(idx, idx)]

        # Inverse of 2x2
        det = Sigma_ij[0, 0] * Sigma_ij[1, 1] - Sigma_ij[0, 1]**2
        if det > 1e-12:
            Lambda[i, j] = -Sigma_ij[0, 1] / det
            Lambda[j, i] = Lambda[i, j]

    # Compute diagonal entries
    for i in range(n):
        # Diagonal is 1/conditional_variance
        neighbors = adj[i]
        if len(neighbors) == 0:
            Lambda[i, i] = 1.0 / Sigma[i, i]
        else:
            # Use formula: Lambda[i,i] = 1/Sigma[i,i] + sum_{j~i} Lambda[i,j]^2 * Sigma[j,j]
            # Actually simpler: Lambda[i,i] = 1/Var(X_i | X_{N(i)})
            # For tree, this equals (Sigma^{-1})[i,i] restricted to tree

            # Approximate: ensure diagonal dominance
            off_sum = sum(abs(Lambda[i, j]) for j in neighbors)
            Lambda[i, i] = max(1.0 / Sigma[i, i], off_sum + 0.1)

    # Ensure SPD
    if not is_spd(Lambda):
        # Boost diagonal
        eigvals = np.linalg.eigvalsh(Lambda)
        if eigvals[0] < 0.01:
            Lambda += (0.1 - eigvals[0]) * np.eye(n)

    return Lambda


# =============================================================================
# Block averaging
# =============================================================================

def block_average(M: np.ndarray, blocks: List[List[int]],
                  zero_diagonal: bool = True) -> np.ndarray:
    """
    Project matrix M onto block-constant structure.

    Parameters
    ----------
    M : np.ndarray
        Input matrix
    blocks : list of list of int
        Block partition
    zero_diagonal : bool
        If True, set diagonal to zero (for C estimation)

    Returns
    -------
    M_avg : np.ndarray
        Block-averaged matrix
    """
    n = M.shape[0]
    M_avg = np.zeros((n, n), dtype=float)
    r = len(blocks)

    for bi in range(r):
        I = blocks[bi]
        for bj in range(bi, r):
            J = blocks[bj]

            if bi == bj:
                # Within-block: separate diagonal and off-diagonal
                sub = M[np.ix_(I, I)]
                if len(I) > 1:
                    # Off-diagonal average
                    mask = ~np.eye(len(I), dtype=bool)
                    off_avg = np.mean(sub[mask])

                    for ii, i in enumerate(I):
                        for jj, j in enumerate(I):
                            if i != j:
                                M_avg[i, j] = off_avg

                # Diagonal
                if not zero_diagonal:
                    diag_avg = np.mean(np.diag(sub))
                    for i in I:
                        M_avg[i, i] = diag_avg
            else:
                # Cross-block: all entries get same average
                sub = M[np.ix_(I, J)]
                avg = np.mean(sub)
                for i in I:
                    for j in J:
                        M_avg[i, j] = avg
                        M_avg[j, i] = avg

    return M_avg


# =============================================================================
# Estimation methods
# =============================================================================

def estimate_decomposition_based(Sigma_hat: np.ndarray,
                                  blocks: List[List[int]],
                                  verbose: bool = False) -> Dict:
    """
    Our decomposition-based method:
    1. Decompose Sigma_hat = B'^{-1} + C' using constrained decomposition
    2. Apply Chow-Liu to B'^{-1} to get tree structure
    3. Fit tree-structured precision to B'^{-1}
    4. Block-average C'

    Returns dict with estimated B, C, Sigma, and timing info.
    """
    n = Sigma_hat.shape[0]
    t0 = time.perf_counter()

    # Step 1: Decompose Sigma_hat
    # Constraint: C must be block-CONSTANT with zero diagonal
    # Use block-fixed basis: one element per block (within) + one per block pair (cross)
    # This ensures C' is block-constant, not just block-supported
    basis = make_block_fixed_basis_offdiag(n, blocks)

    t_basis = time.perf_counter()

    # Run decomposition: Sigma_hat = B'^{-1} + C'
    B_prime, C_prime, x, info = constrained_decomposition(
        A=Sigma_hat,
        basis=basis,
        method="newton",
        tol=1e-8,
        max_iter=200,
        return_info=True,
        verbose=verbose,
    )

    t_decomp = time.perf_counter()

    # Step 2: Chow-Liu on B'^{-1}
    B_prime_inv = spd_inverse(B_prime)
    tree_edges = chow_liu_tree(B_prime_inv)

    t_chowliu = time.perf_counter()

    # Step 3: Fit tree-structured precision
    B_tree = fit_tree_precision(B_prime_inv, tree_edges)

    t_treefit = time.perf_counter()

    # Step 4: Block-average C'
    C_avg = block_average(C_prime, blocks, zero_diagonal=True)

    t_blockavg = time.perf_counter()

    # Final estimate
    B_est = B_tree
    C_est = C_avg

    # Ensure B_est is SPD
    if not is_spd(B_est):
        eigvals = np.linalg.eigvalsh(B_est)
        B_est = B_est + (0.1 - eigvals[0]) * np.eye(n)

    Sigma_est = spd_inverse(B_est) + C_est

    # Ensure Sigma_est is SPD
    if not is_spd(Sigma_est):
        eigvals = np.linalg.eigvalsh(Sigma_est)
        Sigma_est = Sigma_est + (0.1 - eigvals[0]) * np.eye(n)

    t_total = time.perf_counter() - t0

    return {
        "B": B_est,
        "C": C_est,
        "Sigma": Sigma_est,
        "B_prime": B_prime,
        "C_prime": C_prime,
        "tree_edges": tree_edges,
        "time_total": t_total,
        "time_decomp": t_decomp - t_basis,
        "time_chowliu": t_chowliu - t_decomp,
        "time_treefit": t_treefit - t_chowliu,
        "time_blockavg": t_blockavg - t_treefit,
        "decomp_iters": info.get("iters", -1),
    }


def estimate_em(Sigma_hat: np.ndarray,
                blocks: List[List[int]],
                max_iter: int = 50,
                tol: float = 1e-6,
                verbose: bool = False,
                init_C: Optional[np.ndarray] = None) -> Dict:
    """
    Alternating projections algorithm (NOT true EM):
    1. Initialize C = 0 (or provided init_C for warm start)
    2. Repeat:
       a. Given C, estimate tree-structured B from (Sigma_hat - C)
       b. Given B, estimate block-constant C from Sigma_hat - B^{-1}
    Until convergence.

    WARNING: This is NOT a true EM algorithm. Log-likelihood is NOT guaranteed
    to improve monotonically. The alternating projections can oscillate.
    For guaranteed results, use the decomposition-based method.

    Parameters
    ----------
    init_C : ndarray, optional
        Initial value for C. If None, starts from C=0.
    """
    n = Sigma_hat.shape[0]
    t0 = time.perf_counter()

    # Initialize
    if init_C is not None:
        C = init_C.copy()
    else:
        C = np.zeros((n, n), dtype=float)
    B = np.eye(n)

    for iteration in range(max_iter):
        # Store old for convergence check
        C_old = C.copy()

        # Step a: Given C, estimate tree-structured B
        # Model: Sigma = B^{-1} + C, so Sigma - C = B^{-1}
        # We want tree-structured B, so fit tree to (Sigma_hat - C)^{-1}
        B_inv_approx = Sigma_hat - C

        # Ensure SPD before inverting
        if not is_spd(B_inv_approx):
            eigvals = np.linalg.eigvalsh(B_inv_approx)
            B_inv_approx = B_inv_approx + (0.1 - eigvals[0]) * np.eye(n)

        # B_approx = (Sigma_hat - C)^{-1} should have tree structure
        B_approx = spd_inverse(B_inv_approx)

        # Fit tree structure to B_approx (the precision, not the covariance!)
        # Chow-Liu finds optimal tree for precision matrix
        tree_edges = chow_liu_tree(B_inv_approx)  # Tree based on covariance B^{-1}
        B = fit_tree_precision(B_inv_approx, tree_edges)  # Fit tree precision

        # Step b: Given B, estimate block-constant C
        B_inv = spd_inverse(B)
        C_raw = Sigma_hat - B_inv
        C = block_average(C_raw, blocks, zero_diagonal=True)

        # Convergence check
        diff = np.linalg.norm(C - C_old, 'fro')
        if verbose and iteration % 10 == 0:
            print(f"  EM iter {iteration}: ||C_new - C_old||_F = {diff:.6e}")

        if diff < tol:
            break

    t_total = time.perf_counter() - t0

    # Final estimate
    Sigma_est = spd_inverse(B) + C
    if not is_spd(Sigma_est):
        eigvals = np.linalg.eigvalsh(Sigma_est)
        Sigma_est = Sigma_est + (0.1 - eigvals[0]) * np.eye(n)

    return {
        "B": B,
        "C": C,
        "Sigma": Sigma_est,
        "tree_edges": tree_edges,
        "time_total": t_total,
        "em_iters": iteration + 1,
    }


def estimate_naive(Sigma_hat: np.ndarray) -> Dict:
    """
    Naive: just use empirical covariance (no structure).
    """
    return {
        "Sigma": Sigma_hat.copy(),
        "B": None,
        "C": None,
        "time_total": 0.0,
    }


def estimate_tree_only(Sigma_hat: np.ndarray) -> Dict:
    """
    Tree-only: Chow-Liu on full Sigma_hat^{-1} (ignores block structure).
    """
    n = Sigma_hat.shape[0]
    t0 = time.perf_counter()

    tree_edges = chow_liu_tree(Sigma_hat)
    B = fit_tree_precision(Sigma_hat, tree_edges)

    t_total = time.perf_counter() - t0

    Sigma_est = spd_inverse(B)

    return {
        "B": B,
        "C": None,
        "Sigma": Sigma_est,
        "tree_edges": tree_edges,
        "time_total": t_total,
    }


def estimate_block_only(Sigma_hat: np.ndarray,
                        blocks: List[List[int]]) -> Dict:
    """
    Block-only: block-average Sigma_hat (ignores tree structure).
    """
    t0 = time.perf_counter()

    Sigma_est = block_average(Sigma_hat, blocks, zero_diagonal=False)

    # Ensure SPD
    if not is_spd(Sigma_est):
        eigvals = np.linalg.eigvalsh(Sigma_est)
        Sigma_est = Sigma_est + (0.1 - eigvals[0]) * np.eye(Sigma_est.shape[0])

    t_total = time.perf_counter() - t0

    return {
        "B": None,
        "C": None,
        "Sigma": Sigma_est,
        "time_total": t_total,
    }


def estimate_decomp_then_em(Sigma_hat: np.ndarray,
                            blocks: List[List[int]],
                            em_max_iter: int = 20,
                            verbose: bool = False) -> Dict:
    """
    Combined method: Decomposition followed by alternating projections.

    WARNING: This does NOT improve on decomposition alone because the
    "EM" is actually alternating projections that don't monotonically
    improve log-likelihood. The decomposition-based method alone is
    recommended for this problem.

    Kept for comparison/experimentation only.
    """
    n = Sigma_hat.shape[0]
    t0 = time.perf_counter()

    # Step 1: Decomposition for initialization
    res_decomp = estimate_decomposition_based(Sigma_hat, blocks, verbose=verbose)

    t_decomp = time.perf_counter()

    # Step 2: Use decomposition's C' as warm start for EM
    # Use block-averaged C' as initialization
    C_init = block_average(res_decomp["C_prime"], blocks, zero_diagonal=True)

    res_em = estimate_em(
        Sigma_hat, blocks,
        max_iter=em_max_iter,
        init_C=C_init,
        verbose=verbose
    )

    t_total = time.perf_counter() - t0

    return {
        "B": res_em["B"],
        "C": res_em["C"],
        "Sigma": res_em["Sigma"],
        "tree_edges": res_em.get("tree_edges"),
        "time_total": t_total,
        "time_decomp": t_decomp - t0,
        "time_em": res_em["time_total"],
        "decomp_iters": res_decomp.get("decomp_iters", -1),
        "em_iters": res_em.get("em_iters", -1),
    }


# =============================================================================
# Evaluation metrics
# =============================================================================

def frobenius_error(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius norm of difference."""
    return np.linalg.norm(A - B, 'fro')


def relative_frobenius_error(A: np.ndarray, B: np.ndarray) -> float:
    """Relative Frobenius error: ||A - B||_F / ||A||_F."""
    return np.linalg.norm(A - B, 'fro') / np.linalg.norm(A, 'fro')


def log_det_divergence(Sigma_true: np.ndarray, Sigma_est: np.ndarray) -> float:
    """
    Log-det Bregman divergence (Stein's loss):
    D(Sigma_true || Sigma_est) = tr(Sigma_true @ Sigma_est^{-1}) - log|Sigma_true @ Sigma_est^{-1}| - n
    """
    n = Sigma_true.shape[0]
    try:
        Sigma_est_inv = spd_inverse(Sigma_est)
        product = Sigma_true @ Sigma_est_inv
        trace = np.trace(product)
        _, logdet = np.linalg.slogdet(product)
        return trace - logdet - n
    except:
        return float('inf')


def kl_divergence_gaussian(Sigma_true: np.ndarray, Sigma_est: np.ndarray) -> float:
    """
    KL divergence between N(0, Sigma_true) and N(0, Sigma_est).
    KL = 0.5 * (tr(Sigma_est^{-1} Sigma_true) - n + log|Sigma_est|/|Sigma_true|)
    """
    n = Sigma_true.shape[0]
    try:
        Sigma_est_inv = spd_inverse(Sigma_est)
        trace_term = np.trace(Sigma_est_inv @ Sigma_true)
        _, logdet_est = np.linalg.slogdet(Sigma_est)
        _, logdet_true = np.linalg.slogdet(Sigma_true)
        return 0.5 * (trace_term - n + logdet_est - logdet_true)
    except:
        return float('inf')


def gaussian_log_likelihood(Sigma_est: np.ndarray, Sigma_hat: np.ndarray, n_samples: int) -> float:
    """
    Gaussian log-likelihood (up to constant) for estimator Sigma_est given empirical cov Sigma_hat.

    log L(Sigma | data) = -n/2 * (log|Sigma| + tr(Sigma^{-1} Sigma_hat))

    We return the normalized version (per sample, ignoring constants):
    -0.5 * (log|Sigma| + tr(Sigma^{-1} Sigma_hat))

    Higher is better.
    """
    try:
        _, logdet = np.linalg.slogdet(Sigma_est)
        Sigma_inv = spd_inverse(Sigma_est)
        trace_term = np.trace(Sigma_inv @ Sigma_hat)
        return -0.5 * (logdet + trace_term)
    except:
        return float('-inf')


# =============================================================================
# Main simulation
# =============================================================================

def run_simulation(n: int, r: int, n_samples: int, seed: int = 0,
                   verbose: bool = False) -> Dict:
    """
    Run one simulation comparing all methods.

    Parameters
    ----------
    n : int
        Matrix dimension
    r : int
        Number of blocks
    n_samples : int
        Number of samples for empirical covariance
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Results for all methods including errors and timing
    """
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Simulation: n={n}, r={r}, samples={n_samples}, seed={seed}")
        print(f"{'='*60}")

    # Generate true model
    if verbose:
        print("\n1. Generating true covariance structure...")

    blocks = make_blocks_variable(n, r=r, seed=seed)
    tree_edges = make_random_tree(n, seed=seed + 1)

    Sigma_true, B_true, C_true = make_true_covariance(
        n, blocks, tree_edges, seed=seed + 2
    )

    if verbose:
        print(f"   - Tree edges: {len(tree_edges)}")
        print(f"   - Blocks: {[len(b) for b in blocks]}")
        print(f"   - Sigma condition number: {np.linalg.cond(Sigma_true):.2f}")

    # Sample and form empirical covariance
    if verbose:
        print("\n2. Sampling from true distribution...")

    L = np.linalg.cholesky(Sigma_true)
    Z = rng.standard_normal((n_samples, n))
    X = Z @ L.T  # samples from N(0, Sigma_true)

    Sigma_hat = (X.T @ X) / n_samples

    if verbose:
        print(f"   - Empirical cov error: {relative_frobenius_error(Sigma_true, Sigma_hat):.4f}")

    # Run all methods
    results = {
        "n": n,
        "r": r,
        "n_samples": n_samples,
        "seed": seed,
        "Sigma_true": Sigma_true,
        "B_true": B_true,
        "C_true": C_true,
        "Sigma_hat": Sigma_hat,
        "blocks": blocks,
        "tree_edges_true": tree_edges,
    }

    # Method 1: Our decomposition-based
    if verbose:
        print("\n3. Running decomposition-based method (ours)...")

    try:
        res_decomp = estimate_decomposition_based(Sigma_hat, blocks, verbose=verbose)
        results["decomp"] = res_decomp
        if verbose:
            print(f"   - Time: {res_decomp['time_total']:.4f}s")
            print(f"   - Decomp iters: {res_decomp['decomp_iters']}")
    except Exception as e:
        print(f"   - FAILED: {e}")
        results["decomp"] = {"Sigma": Sigma_hat, "time_total": -1, "error": str(e)}

    # Method 2: EM
    if verbose:
        print("\n4. Running EM method...")

    try:
        res_em = estimate_em(Sigma_hat, blocks, verbose=verbose)
        results["em"] = res_em
        if verbose:
            print(f"   - Time: {res_em['time_total']:.4f}s")
            print(f"   - EM iters: {res_em['em_iters']}")
    except Exception as e:
        print(f"   - FAILED: {e}")
        results["em"] = {"Sigma": Sigma_hat, "time_total": -1, "error": str(e)}

    # Method 3: Naive
    if verbose:
        print("\n5. Running naive method...")

    res_naive = estimate_naive(Sigma_hat)
    results["naive"] = res_naive

    # Method 4: Tree-only
    if verbose:
        print("\n6. Running tree-only method...")

    try:
        res_tree = estimate_tree_only(Sigma_hat)
        results["tree_only"] = res_tree
        if verbose:
            print(f"   - Time: {res_tree['time_total']:.4f}s")
    except Exception as e:
        print(f"   - FAILED: {e}")
        results["tree_only"] = {"Sigma": Sigma_hat, "time_total": -1, "error": str(e)}

    # Method 5: Block-only
    if verbose:
        print("\n7. Running block-only method...")

    try:
        res_block = estimate_block_only(Sigma_hat, blocks)
        results["block_only"] = res_block
        if verbose:
            print(f"   - Time: {res_block['time_total']:.4f}s")
    except Exception as e:
        print(f"   - FAILED: {e}")
        results["block_only"] = {"Sigma": Sigma_hat, "time_total": -1, "error": str(e)}

    # Method 6: Decomposition + EM (recommended combined approach)
    if verbose:
        print("\n8. Running decomposition + EM (warm start)...")

    try:
        res_combined = estimate_decomp_then_em(Sigma_hat, blocks, verbose=verbose)
        results["decomp_em"] = res_combined
        if verbose:
            print(f"   - Time: {res_combined['time_total']:.4f}s")
            print(f"   - Decomp iters: {res_combined['decomp_iters']}, EM iters: {res_combined['em_iters']}")
    except Exception as e:
        print(f"   - FAILED: {e}")
        results["decomp_em"] = {"Sigma": Sigma_hat, "time_total": -1, "error": str(e)}

    # Compute errors
    if verbose:
        print("\n9. Computing errors...")

    methods = ["decomp", "em", "naive", "tree_only", "block_only", "decomp_em"]

    for method in methods:
        if method in results and "Sigma" in results[method]:
            Sigma_est = results[method]["Sigma"]
            results[method]["frob_error"] = frobenius_error(Sigma_true, Sigma_est)
            results[method]["rel_frob_error"] = relative_frobenius_error(Sigma_true, Sigma_est)
            results[method]["logdet_div"] = log_det_divergence(Sigma_true, Sigma_est)
            results[method]["kl_div"] = kl_divergence_gaussian(Sigma_true, Sigma_est)
            results[method]["log_likelihood"] = gaussian_log_likelihood(Sigma_est, Sigma_hat, n_samples)

    # Print summary
    if verbose:
        print("\n" + "="*85)
        print("RESULTS SUMMARY")
        print("="*85)
        print(f"{'Method':<15} {'Rel.Frob.Err':<14} {'Log-Lik':<14} {'KL Div':<14} {'Time(s)':<10}")
        print("-"*85)
        for method in methods:
            if method in results:
                r = results[method]
                rel_err = r.get("rel_frob_error", float('nan'))
                loglik = r.get("log_likelihood", float('nan'))
                kl = r.get("kl_div", float('nan'))
                time_s = r.get("time_total", -1)
                print(f"{method:<15} {rel_err:<14.4f} {loglik:<14.4f} {kl:<14.4f} {time_s:<10.4f}")
        print("-"*85)
        print("\nMetrics:")
        print("  Rel.Frob.Err = ||Sigma_est - Sigma_true||_F / ||Sigma_true||_F  (lower is better)")
        print("  Log-Lik      = log p(data | Sigma_est), normalized  (HIGHER is better)")
        print("  KL Div       = KL(N(0,Sigma_true) || N(0,Sigma_est))  (lower is better)")
        print("\nMethods:")
        print("  decomp     = Decomposition only (our method)")
        print("  em         = EM from zero initialization")
        print("  decomp_em  = Decomposition + EM (warm start)")
        print("  naive      = Empirical covariance (no structure)")
        print("  tree_only  = Chow-Liu on full matrix")
        print("  block_only = Block averaging only")

    return results


def run_multiple_simulations(n: int, r: int, n_samples: int,
                              n_reps: int = 10, seed: int = 0,
                              verbose: bool = False) -> Dict:
    """
    Run multiple simulations and aggregate results.
    """
    all_results = []

    for rep in range(n_reps):
        if verbose:
            print(f"\n*** Repetition {rep+1}/{n_reps} ***")

        res = run_simulation(n, r, n_samples, seed=seed + rep * 1000, verbose=verbose)
        all_results.append(res)

    # Aggregate
    methods = ["decomp", "em", "naive", "tree_only", "block_only", "decomp_em"]
    summary = {}

    for method in methods:
        errors = [r[method].get("rel_frob_error", float('nan'))
                  for r in all_results if method in r]
        times = [r[method].get("time_total", -1)
                 for r in all_results if method in r]
        kls = [r[method].get("kl_div", float('nan'))
               for r in all_results if method in r]
        logliks = [r[method].get("log_likelihood", float('nan'))
                   for r in all_results if method in r]

        summary[method] = {
            "rel_frob_mean": np.nanmean(errors),
            "rel_frob_std": np.nanstd(errors),
            "kl_mean": np.nanmean(kls),
            "kl_std": np.nanstd(kls),
            "loglik_mean": np.nanmean(logliks),
            "loglik_std": np.nanstd(logliks),
            "time_mean": np.mean([t for t in times if t >= 0]),
            "time_std": np.std([t for t in times if t >= 0]),
        }

    return {
        "all_results": all_results,
        "summary": summary,
        "n": n,
        "r": r,
        "n_samples": n_samples,
        "n_reps": n_reps,
    }


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Covariance estimation demo: Tree + Block structure"
    )
    parser.add_argument("--n", type=int, default=30, help="Matrix dimension")
    parser.add_argument("--r", type=int, default=4, help="Number of blocks")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--reps", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.reps == 1:
        results = run_simulation(
            n=args.n,
            r=args.r,
            n_samples=args.samples,
            seed=args.seed,
            verbose=True,
        )
    else:
        results = run_multiple_simulations(
            n=args.n,
            r=args.r,
            n_samples=args.samples,
            n_reps=args.reps,
            seed=args.seed,
            verbose=args.verbose,
        )

        # Print aggregated summary
        print("\n" + "="*90)
        print(f"AGGREGATED RESULTS ({args.reps} repetitions)")
        print(f"n={args.n}, r={args.r}, samples={args.samples}")
        print("="*90)
        print(f"{'Method':<15} {'Rel.Err':<12} {'(std)':<10} {'Log-Lik':<12} {'(std)':<10} {'Time':<10}")
        print("-"*90)

        for method in ["decomp", "em", "decomp_em", "naive", "tree_only", "block_only"]:
            s = results["summary"][method]
            print(f"{method:<15} {s['rel_frob_mean']:<12.4f} {s['rel_frob_std']:<10.4f} "
                  f"{s['loglik_mean']:<12.2f} {s['loglik_std']:<10.2f} {s['time_mean']:<10.4f}")

        print("-"*90)
        print("\nRel.Err: lower is better (error vs true Sigma)")
        print("Log-Lik: HIGHER is better (fit to data)")
        print("\nIf decomp_em has higher Log-Lik than decomp but worse Rel.Err,")
        print("it means EM is overfitting to noise in Sigma_hat.")
