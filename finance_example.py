from constrained_decomposition_core import *
from constrained_decomposition_core import make_orthogonal_complement_basis, constrained_decomposition_dual, constrained_decomposition_direct
from constrained_decomposition_matrices import *
from constrained_decomposition_matrices import spd_mixed_fbm_blocked, fgn_cov_toeplitz
from constrained_decomposition_viz import plot_decomposition_heatmaps
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for proper math rendering
from pathlib import Path
import math
import argparse
import time
import multiprocessing as mp
import os
import pandas as pd
import fcntl  # For file locking on Linux


def get_results_file():
    """Get path to the single master results CSV file."""
    here = Path(__file__).resolve().parent
    results_dir = here / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "all_results.csv"


def append_result(H, val_sum, val_markov, val_full, params):
    """Append or update a result row in the master CSV file (thread/process safe).

    If a row with matching (H, model, n, alpha) exists, merge new values into it.
    Otherwise, append a new row.
    """
    filename = get_results_file()
    lockfile = filename.with_suffix('.csv.lock')

    # Define all possible columns in a fixed order
    all_columns = ['H', 'model', 'n', 'N', 'alpha', 'delta_t', 'strategy',
                   'value_sum', 'value_markovian', 'value_full']

    H_rounded = round(H, 6)
    new_row = {
        'H': H_rounded,
        'model': params['model'],
        'n': params['n'],
        'N': params.get('N', params['n']),
        'alpha': params['alpha'],
        'delta_t': round(params.get('delta_t', 1.0), 6),
        'strategy': params['strategy'],
        'value_sum': round(val_sum, 6) if val_sum is not None and not np.isnan(val_sum) else '',
        'value_markovian': round(val_markov, 6) if val_markov is not None and not np.isnan(val_markov) else '',
        'value_full': round(val_full, 6) if val_full is not None and not np.isnan(val_full) else '',
    }

    # Use file locking for safe concurrent writes
    with open(lockfile, 'w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)  # Acquire exclusive lock
        try:
            if filename.exists():
                df = pd.read_csv(filename, dtype={'value_sum': str, 'value_markovian': str, 'value_full': str})

                # Find existing row with same key (H, model, n, alpha)
                mask = (
                    (np.isclose(df['H'], H_rounded)) &
                    (df['model'] == params['model']) &
                    (df['n'] == params['n']) &
                    (np.isclose(df['alpha'], params['alpha']))
                )

                if mask.any():
                    # Update existing row - merge new values into old
                    idx = df[mask].index[0]
                    for col in ['value_sum', 'value_markovian', 'value_full']:
                        new_val = new_row[col]
                        if new_val != '':  # Only update if we have a new value
                            df.at[idx, col] = new_val
                    df.to_csv(filename, index=False)
                else:
                    # Append new row
                    df_row = pd.DataFrame([new_row], columns=all_columns)
                    df_row.to_csv(filename, mode='a', header=False, index=False)
            else:
                df_row = pd.DataFrame([new_row], columns=all_columns)
                df_row.to_csv(filename, index=False)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)  # Release lock


def load_results_for_params(model, n, alpha, strategy='all'):
    """
    Load results from master CSV filtered by parameters.

    Parameters
    ----------
    model : str
    n : int
    alpha : float
    strategy : str
        'all' to merge results from all strategies (default for plotting),
        or 'both', 'markovian', 'full' to filter by specific strategy.

    Returns
    -------
    H_vec : array
    val_markov : array or None
    val_full : array or None
    val_sum : array or None
    """
    filename = get_results_file()
    if not filename.exists():
        return None, None, None, None

    df = pd.read_csv(filename)

    # Filter by parameters (optionally by strategy)
    mask = (
        (df['model'] == model) &
        (df['n'] == n) &
        (np.isclose(df['alpha'], alpha))
    )
    if strategy != 'all':
        mask = mask & (df['strategy'] == strategy)
    df_filtered = df[mask].sort_values('H')

    if len(df_filtered) == 0:
        return None, None, None, None

    H_vec = df_filtered['H'].values
    # Convert to numeric, treating empty strings as NaN
    val_sum = pd.to_numeric(df_filtered['value_sum'], errors='coerce').values if 'value_sum' in df_filtered.columns else None
    val_markov = pd.to_numeric(df_filtered['value_markovian'], errors='coerce').values if 'value_markovian' in df_filtered.columns else None
    val_full = pd.to_numeric(df_filtered['value_full'], errors='coerce').values if 'value_full' in df_filtered.columns else None

    print(f"Loaded {len(H_vec)} results for model={model}, n={n}, alpha={alpha}, strategy={strategy}")
    return H_vec, val_markov, val_full, val_sum


def get_all_param_combinations():
    """Get all unique (model, n, alpha) combinations from the results CSV."""
    filename = get_results_file()
    if not filename.exists():
        return []

    df = pd.read_csv(filename)
    # Get unique combinations, for fbm only keep alpha=1.0
    combos = df[['model', 'n', 'alpha']].drop_duplicates()
    # Filter: for fbm, keep only alpha=1.0 (others are duplicates)
    mask = (combos['model'] != 'fbm') | (np.isclose(combos['alpha'], 1.0))
    combos = combos[mask]
    return combos.values.tolist()


def get_completed_H_values(model, n, alpha, strategy):
    """Get set of H values already computed for given parameters (thread/process safe).

    Only considers a row complete if all relevant strategy columns have values:
    - mixed_fbm: needs value_sum, value_markovian, value_full
    - fbm: needs value_markovian, value_full (no sum)
    """
    filename = get_results_file()
    if not filename.exists():
        return set()

    lockfile = filename.with_suffix('.csv.lock')

    # Use shared lock for reading (allows multiple readers, blocks during writes)
    with open(lockfile, 'w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
        try:
            df = pd.read_csv(filename)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    # Filter by model, n, alpha (but NOT strategy - we check value columns instead)
    mask = (
        (df['model'] == model) &
        (df['n'] == n) &
        (np.isclose(df['alpha'], alpha))
    )

    # Check if the relevant value columns are filled (regardless of strategy column)
    # This allows jobs with --strategy markovian to recognize values from strategy='both' rows
    if strategy == "both":
        has_markov = df['value_markovian'].notna() & (df['value_markovian'] != '')
        has_full = df['value_full'].notna() & (df['value_full'] != '')
        if model == "mixed_fbm":
            has_sum = df['value_sum'].notna() & (df['value_sum'] != '')
            mask = mask & has_markov & has_full & has_sum
        else:  # fbm - no sum strategy
            mask = mask & has_markov & has_full
    elif strategy == "markovian":
        has_markov = df['value_markovian'].notna() & (df['value_markovian'] != '')
        mask = mask & has_markov
    elif strategy == "full":
        has_full = df['value_full'].notna() & (df['value_full'] != '')
        mask = mask & has_full

    # Round to avoid floating point issues
    return set(round(h, 6) for h in df[mask]['H'].values)


# Computing value
# Markovian strategy:
def invest_value_markovian(B, C, log_flag = True):

    (signB, logabsdetB) = np.linalg.slogdet(B)
    (signC, logabsdetC) = np.linalg.slogdet(C)

    if log_flag:
        return (0.5*(logabsdetC-logabsdetB))
    else:
        return -math.exp(0.5*(logabsdetC-logabsdetB))

# General strategy:
def invest_value_general(A, log_flag = True):
    A_inv = spd_inverse(A)
    logabsdetC = -np.sum(np.log(np.diag(A_inv)))

    (signA, logabsdetA) = np.linalg.slogdet(A)

    if log_flag:
        return (0.5*(logabsdetA-logabsdetC))
    else:
        return -math.exp(0.5*(logabsdetA-logabsdetC))


def compute_value_vs_H_fbm(H_vec, n=100):
    n_H = len(H_vec)
    val_vec_markovian = np.zeros(n_H)
    val_vec_general = np.zeros(n_H)

    print(f"fBM: n={n}, matrix size={n}x{n}")
    print(f"Number of H values: {n_H}")
    total_start = time.time()

    for i in range(n_H):
        print(f"\n--- H = {H_vec[i]:.4f} ({i+1}/{n_H}) ---")

        # Build matrix
        A = spd_fractional_BM(n, H=H_vec[i], T=1.0)
        A_inv = spd_inverse(A)

        basis = TridiagC_Basis(n)  # keeps your specialized fast case
        B_newt, C_newt, x_newt = constrained_decomposition(
            A=A_inv,
            basis=basis,
            method="newton",
            tol=1e-6,
            max_iter=500,
            verbose=False
        )

        val_vec_markovian[i] = invest_value_markovian(B_newt, A)

        A_diff = spd_fractional_BM(n, H=H_vec[i], T=1.0, diff_flag=True)
        val_vec_general[i] = invest_value_general(A_diff)

        print(f"  Markovian: {val_vec_markovian[i]:.6f}, General: {val_vec_general[i]:.6f}")

    total_time = time.time() - total_start
    print(f"\n=== Total time: {total_time:.2f} seconds ({total_time/n_H:.2f} sec/H value) ===")

    return val_vec_markovian, val_vec_general


def make_mixed_fbm_full_info_basis(N: int):
    """
    Build the full-information strategy basis for mixed fBM (COO sparse format).

    S is spanned by matrices D^{k,l} in M_{2N}(R), with l = 1,...,N and k <= 2l-2:
        D^{k,l}_{ij} = 1 if i = k and j = 2l - 1
        D^{k,l}_{ij} = 1 if i = 2l - 1 and j = k
        D^{k,l}_{ij} = 0 otherwise

    Each D^{k,l} has only 2 non-zeros, so COO is very efficient.
    Dimension: O(N^2)
    """
    n = 2 * N
    coo_mats = []

    for l in range(1, N + 1):  # l = 1, ..., N
        j_col = 2 * l - 1 - 1  # 0-based: j = 2l-1 in paper -> index 2l-2
        for k in range(1, 2 * l - 1):  # k = 1, ..., 2l-2
            i_row = k - 1  # 0-based
            # Each D has 2 non-zeros: (i_row, j_col) and (j_col, i_row)
            rows = np.array([i_row, j_col], dtype=int)
            cols = np.array([j_col, i_row], dtype=int)
            vals = np.array([1.0, 1.0], dtype=float)
            coo_mats.append((rows, cols, vals))

    return SymBasis(n=n, coo_mats=coo_mats, name=f"mixed_fbm_full_info_N={N}")


# =============================================================================
# BLOCKED ORDERING BASIS FUNCTIONS
# =============================================================================
# These are for the blocked variable ordering Z = (X₁,...,X_N, Y₁,...,Y_N)
# which enables exploitation of Toeplitz structure in the Schur complement.

def make_mixed_fbm_markovian_basis_blocked(N: int):
    """
    Build the Markovian strategy basis for mixed fBM in BLOCKED ordering (COO sparse format).

    For blocked ordering Z = (X₁,...,X_N, Y₁,...,Y_N), the Markovian basis consists of:

    D^Mark_{l,X}: Symmetric matrix with 1s at (r,l) and (l,r) for r = 1,...,l-1
                  (within the X-block, upper-left N×N)

    D^Mark_{l,Y}: Symmetric matrix with 1s at (N+r,l) and (l,N+r) for r = 1,...,l-1
                  (cross-block between Y and X)

    for l = 2,...,N (l=1 gives empty matrices).

    Total dimension: 2*(N-1) = O(N)

    Parameters
    ----------
    N : int
        Number of time steps. Matrix size is 2N x 2N.

    Returns
    -------
    basis : SymBasis
        The Markovian basis in blocked ordering.

    Notes
    -----
    The key advantage of blocked ordering is that C(x) in Λ = B^{-1} + C has
    structure where the Schur complement S(x) = α²Γ_H - C̃(x) is Toeplitz,
    enabling fast O(N²) operations via Toeplitz algorithms.
    """
    n = 2 * N
    coo_mats = []

    for l in range(2, N + 1):  # l = 2, ..., N (l=1 is empty)
        l_idx = l - 1  # 0-based index for column l

        # --- D^Mark_{l,X}: entries in upper-left N×N block ---
        # Positions (r-1, l-1) and (l-1, r-1) for r = 1,...,l-1
        rows_X, cols_X, vals_X = [], [], []
        for r in range(1, l):  # r = 1, ..., l-1
            r_idx = r - 1  # 0-based
            rows_X.extend([r_idx, l_idx])
            cols_X.extend([l_idx, r_idx])
            vals_X.extend([1.0, 1.0])
        if rows_X:
            coo_mats.append((np.array(rows_X, dtype=int),
                             np.array(cols_X, dtype=int),
                             np.array(vals_X, dtype=float)))

        # --- D^Mark_{l,Y}: entries in cross-block (Y rows, X cols) ---
        # Positions (N+r-1, l-1) and (l-1, N+r-1) for r = 1,...,l-1
        rows_Y, cols_Y, vals_Y = [], [], []
        for r in range(1, l):  # r = 1, ..., l-1
            r_idx_Y = N + r - 1  # 0-based, in Y block
            rows_Y.extend([r_idx_Y, l_idx])
            cols_Y.extend([l_idx, r_idx_Y])
            vals_Y.extend([1.0, 1.0])
        if rows_Y:
            coo_mats.append((np.array(rows_Y, dtype=int),
                             np.array(cols_Y, dtype=int),
                             np.array(vals_Y, dtype=float)))

    return SymBasis(n=n, coo_mats=coo_mats, name=f"mixed_fbm_markovian_blocked_N={N}")


def make_mixed_fbm_full_info_basis_blocked(N: int):
    """
    Build the full-information strategy basis for mixed fBM in BLOCKED ordering.

    For blocked ordering Z = (X₁,...,X_N, Y₁,...,Y_N), the full-info basis is:

    For l = 2,...,N:
      - D^{r,l} for r = 1,...,l-1: entries at (r,l) and (l,r) in X-block
      - D^{N+r,l} for r = 1,...,l-1: entries at (N+r,l) and (l,N+r) cross-block

    Total dimension: N*(N-1) = O(N²)

    Parameters
    ----------
    N : int
        Number of time steps. Matrix size is 2N x 2N.

    Returns
    -------
    basis : SymBasis
        The full-info basis in blocked ordering.
    """
    n = 2 * N
    coo_mats = []

    for l in range(2, N + 1):  # l = 2, ..., N
        l_idx = l - 1  # 0-based

        # --- X-block entries: D^{r,l} ---
        for r in range(1, l):  # r = 1, ..., l-1
            r_idx = r - 1  # 0-based
            rows = np.array([r_idx, l_idx], dtype=int)
            cols = np.array([l_idx, r_idx], dtype=int)
            vals = np.array([1.0, 1.0], dtype=float)
            coo_mats.append((rows, cols, vals))

        # --- Cross-block entries: D^{N+r,l} ---
        for r in range(1, l):  # r = 1, ..., l-1
            r_idx_Y = N + r - 1  # 0-based, in Y block
            rows = np.array([r_idx_Y, l_idx], dtype=int)
            cols = np.array([l_idx, r_idx_Y], dtype=int)
            vals = np.array([1.0, 1.0], dtype=float)
            coo_mats.append((rows, cols, vals))

    return SymBasis(n=n, coo_mats=coo_mats, name=f"mixed_fbm_full_info_blocked_N={N}")


def get_interleaved_to_blocked_permutation(N: int) -> np.ndarray:
    """
    Get the permutation matrix P that converts interleaved to blocked ordering.

    Interleaved: (X₁,Y₁,X₂,Y₂,...,X_N,Y_N)
    Blocked:     (X₁,X₂,...,X_N,Y₁,Y₂,...,Y_N)

    If Σ_int is the interleaved covariance and Σ_blk is the blocked covariance:
        Σ_blk = P @ Σ_int @ P.T

    Parameters
    ----------
    N : int
        Number of time steps.

    Returns
    -------
    P : np.ndarray
        The 2N×2N permutation matrix.
    """
    n = 2 * N
    P = np.zeros((n, n), dtype=float)

    # X variables: interleaved positions 0,2,4,...,2N-2 -> blocked positions 0,1,2,...,N-1
    for i in range(N):
        P[i, 2 * i] = 1.0  # X_i in interleaved is at 2i, in blocked at i

    # Y variables: interleaved positions 1,3,5,...,2N-1 -> blocked positions N,N+1,...,2N-1
    for i in range(N):
        P[N + i, 2 * i + 1] = 1.0  # Y_i in interleaved is at 2i+1, in blocked at N+i

    return P


def verify_blocked_ordering(N: int = 10, H: float = 0.6, alpha: float = 1.0, verbose: bool = True):
    """
    Verify that the blocked ordering produces equivalent results to interleaved ordering.

    This checks:
    1. The covariance matrices are related by permutation: Σ_blk = P @ Σ_int @ P.T
    2. The eigenvalues are the same (determinant is the same)
    3. The investment values are the same for both orderings

    Parameters
    ----------
    N : int
        Number of time steps.
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.
    verbose : bool
        Print detailed output.

    Returns
    -------
    success : bool
        True if all checks pass.
    """
    delta_t = 1.0 / N

    if verbose:
        print(f"\n{'='*60}")
        print(f"Verifying blocked vs interleaved ordering")
        print(f"N={N}, H={H:.2f}, alpha={alpha:.2f}, delta_t={delta_t:.6f}")
        print(f"{'='*60}")

    # Build both covariance matrices
    Sigma_int = spd_mixed_fbm(N, H=H, alpha=alpha, delta_t=delta_t)
    Sigma_blk = spd_mixed_fbm_blocked(N, H=H, alpha=alpha, delta_t=delta_t)

    # Get permutation matrix
    P = get_interleaved_to_blocked_permutation(N)

    # Check permutation relationship: Σ_blk = P @ Σ_int @ P.T
    Sigma_int_permuted = P @ Sigma_int @ P.T
    perm_diff = np.max(np.abs(Sigma_blk - Sigma_int_permuted))

    if verbose:
        print(f"\n1. Permutation check: max|Σ_blk - P·Σ_int·P'| = {perm_diff:.2e}")

    perm_ok = perm_diff < 1e-12

    # Check eigenvalues are the same
    eig_int = np.linalg.eigvalsh(Sigma_int)
    eig_blk = np.linalg.eigvalsh(Sigma_blk)
    eig_diff = np.max(np.abs(np.sort(eig_int) - np.sort(eig_blk)))

    if verbose:
        print(f"2. Eigenvalue check: max|λ_int - λ_blk| = {eig_diff:.2e}")

    eig_ok = eig_diff < 1e-12

    # Check log-determinants are the same
    _, logdet_int = np.linalg.slogdet(Sigma_int)
    _, logdet_blk = np.linalg.slogdet(Sigma_blk)
    logdet_diff = np.abs(logdet_int - logdet_blk)

    if verbose:
        print(f"3. Log-det check: |log|Σ_int| - log|Σ_blk|| = {logdet_diff:.2e}")

    logdet_ok = logdet_diff < 1e-12

    # Compute investment values with both orderings
    # For sum strategy, value should be identical since it's the same N×N block
    Lambda_int = spd_inverse(Sigma_int)
    Lambda_blk = spd_inverse(Sigma_blk)

    # Sum strategy (always diagonal B)
    log_B_int = -np.sum(np.log(np.diag(Lambda_int)))
    log_B_blk = -np.sum(np.log(np.diag(Lambda_blk)))
    value_sum_int = 0.5 * (logdet_int - log_B_int)
    value_sum_blk = 0.5 * (logdet_blk - log_B_blk)
    value_sum_diff = np.abs(value_sum_int - value_sum_blk)

    if verbose:
        print(f"\n4. Sum strategy value check:")
        print(f"   Interleaved: {value_sum_int:.10f}")
        print(f"   Blocked:     {value_sum_blk:.10f}")
        print(f"   Difference:  {value_sum_diff:.2e}")

    # Note: Sum values won't be exactly equal because the diagonal of Lambda
    # is different between orderings (even though eigenvalues are same)
    # This is expected - the constraint structure is different!

    # For Markovian strategy, test with blocked basis
    basis_int = make_mixed_fbm_markovian_basis(N)
    basis_blk = make_mixed_fbm_markovian_basis_blocked(N)

    if verbose:
        print(f"\n5. Markovian basis dimensions:")
        print(f"   Interleaved: {basis_int.m}")
        print(f"   Blocked:     {basis_blk.m}")

    # Run decomposition with both
    B_int, C_int, x_int = constrained_decomposition(Lambda_int, basis_int, method="newton", verbose=False)
    B_blk, C_blk, x_blk = constrained_decomposition(Lambda_blk, basis_blk, method="newton", verbose=False)

    _, logdet_B_int = np.linalg.slogdet(B_int)
    _, logdet_B_blk = np.linalg.slogdet(B_blk)

    value_markov_int = 0.5 * (logdet_int - logdet_B_int)
    value_markov_blk = 0.5 * (logdet_blk - logdet_B_blk)
    value_markov_diff = np.abs(value_markov_int - value_markov_blk)

    if verbose:
        print(f"\n6. Markovian strategy value check:")
        print(f"   Interleaved: {value_markov_int:.10f}")
        print(f"   Blocked:     {value_markov_blk:.10f}")
        print(f"   Difference:  {value_markov_diff:.2e}")

    markov_ok = value_markov_diff < 1e-6

    # Summary
    all_ok = perm_ok and eig_ok and logdet_ok and markov_ok

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
        if not perm_ok:
            print(f"  - Permutation check FAILED")
        if not eig_ok:
            print(f"  - Eigenvalue check FAILED")
        if not logdet_ok:
            print(f"  - Log-det check FAILED")
        if not markov_ok:
            print(f"  - Markovian value check FAILED")
        print(f"{'='*60}\n")

    return all_ok


def benchmark_blocked_vs_interleaved(N_values=None, H=0.6, alpha=1.0, strategy="markovian"):
    """
    Benchmark blocked vs interleaved ordering performance.

    Parameters
    ----------
    N_values : list of int
        Values of N to test. Default: [10, 20, 50, 100]
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.
    strategy : str
        Strategy to benchmark.

    Returns
    -------
    results : list of dict
        Benchmark results for each N.
    """
    if N_values is None:
        N_values = [10, 20, 50, 100]

    print(f"\n{'='*70}")
    print(f"Benchmark: Blocked vs Interleaved Ordering")
    print(f"H={H}, alpha={alpha}, strategy={strategy}")
    print(f"{'='*70}")
    print(f"{'N':>6} {'Interleaved':>15} {'Blocked':>15} {'Speedup':>10} {'Match':>8}")
    print(f"{'-'*70}")

    results = []

    for N in N_values:
        delta_t = 1.0 / N

        # Interleaved
        val_int, info_int = invest_value_mixed_fbm(H, N, alpha, delta_t, strategy=strategy)
        t_int = info_int["time"]

        # Blocked
        val_blk, info_blk = invest_value_mixed_fbm_blocked(H, N, alpha, delta_t, strategy=strategy)
        t_blk = info_blk["time"]

        speedup = t_int / t_blk if t_blk > 0 else float('inf')
        match = abs(val_int - val_blk) < 1e-8

        print(f"{N:>6} {t_int:>14.3f}s {t_blk:>14.3f}s {speedup:>9.2f}x {'OK' if match else 'FAIL':>8}")

        results.append({
            'N': N,
            't_interleaved': t_int,
            't_blocked': t_blk,
            'speedup': speedup,
            'match': match,
            'value_interleaved': val_int,
            'value_blocked': val_blk
        })

    print(f"{'='*70}\n")
    return results


def invest_value_mixed_fbm_blocked(H, N, alpha, delta_t, strategy, method="newton",
                                     Sigma=None, Lambda=None, basis=None,
                                     tol=1e-8, max_iter=500, verbose=False, x_init=None):
    """
    Compute investment value for mixed fBM using BLOCKED ordering.

    This is the blocked-ordering equivalent of invest_value_mixed_fbm.
    The blocked ordering Z = (X₁,...,X_N, Y₁,...,Y_N) enables exploitation of
    Toeplitz structure in the upper-left block for faster operations.

    Parameters
    ----------
    H : float
        Hurst parameter.
    N : int
        Number of time steps. Matrix size is 2N×2N.
    alpha : float
        Weight of fBM component.
    delta_t : float
        Time step size.
    strategy : str
        "sum", "markovian", or "full".
    method : str
        Optimization method for decomposition.
    Sigma : np.ndarray, optional
        Pre-computed covariance matrix in blocked ordering.
    Lambda : np.ndarray, optional
        Pre-computed precision matrix.
    basis : SymBasis, optional
        Pre-computed basis for the strategy.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    verbose : bool
        Print solver progress.
    x_init : np.ndarray, optional
        Initial guess for warm starting.

    Returns
    -------
    value : float
        Log investment value.
    info : dict
        Additional info: {"iters": int, "time": float, "method": str, "error": str or None, "x": array}
    """
    info = {"iters": 0, "time": 0.0, "method": strategy, "error": None, "x": None}
    t_start = time.time()

    try:
        # === Step 1: Build covariance matrix ===
        # Sum strategy uses N×N matrix; others use 2N×2N in blocked ordering
        if Sigma is None:
            if strategy == "sum":
                Sigma = spd_sum_fbm(N, H=H, alpha=alpha, delta_t=delta_t)  # N×N
            else:
                Sigma = spd_mixed_fbm_blocked(N, H=H, alpha=alpha, delta_t=delta_t)  # 2N×2N

        if not is_spd(Sigma):
            info["error"] = "Sigma not SPD"
            return np.nan, info

        # === Step 2: Compute precision matrix Lambda ===
        if Lambda is None:
            Lambda = spd_inverse(Sigma)

        if not is_spd(Lambda):
            info["error"] = "Lambda not SPD"
            return np.nan, info

        # === Step 3: Compute log|Σ| ===
        _, log_det_Sigma = np.linalg.slogdet(Sigma)

        # === Step 4: Compute B / log|B| (strategy-specific) ===
        if strategy == "sum":
            # Closed-form: optimal diagonal B has B_ii = 1/Λ_ii
            log_det_B = -float(np.sum(np.log(np.diag(Lambda))))
            value = 0.5 * (log_det_Sigma - log_det_B)
            info["method"] = "diagonal B"
            info["time"] = time.time() - t_start
            return value, info
        else:
            # Decomposition with blocked basis
            if basis is None:
                if strategy == "markovian":
                    basis = make_mixed_fbm_markovian_basis_blocked(N)
                elif strategy == "full":
                    basis = make_mixed_fbm_full_info_basis_blocked(N)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

            # L-BFGS general doesn't work well for this problem (constraint handling issues)
            # Fall back to newton-cg for reliable convergence
            actual_method = "newton-cg" if method == "lbfgs" else method
            if method == "lbfgs" and verbose:
                print(f"  Note: L-BFGS not supported for this solver, using newton-cg")

            B, _, x, decomp_info = constrained_decomposition(
                A=Lambda, basis=basis, method=actual_method,
                tol=tol, max_iter=max_iter, verbose=verbose, return_info=True,
                x_init=x_init, cg_max_iter=cg_max_iter
            )
            info["iters"] = decomp_info["iters"]
            info["method"] = decomp_info.get("used_method", actual_method)
            if method == "lbfgs":
                info["method"] += " (lbfgs fallback)"
            info["x"] = x

        # === Step 5: Compute log|B| and final value ===
        _, log_det_B = np.linalg.slogdet(B)
        value = 0.5 * (log_det_Sigma - log_det_B)

    except Exception as e:
        info["error"] = str(e)
        value = np.nan

    info["time"] = time.time() - t_start
    return value, info


def make_mixed_fbm_markovian_basis(N: int):
    """
    Build the Markovian strategy basis for mixed fBM (COO sparse format).

    S is spanned by matrices D^{l,1} and D^{l,2} for l = 1,...,N:

    D^{l,1}_{ij} = 1 if i < j, j = 2l-1, and i is odd (1-based)
                   1 if i = 2l-1, j < i, and j is odd
                   0 otherwise

    D^{l,2}_{ij} = 1 if i < j, j = 2l-1, and i is even (1-based)
                   1 if i = 2l-1, j < i, and j is even
                   0 otherwise

    Dimension: 2N = O(N)
    """
    n = 2 * N
    coo_mats = []

    for l in range(1, N + 1):  # l = 1, ..., N
        j_col = 2 * l - 1  # 1-based index for j = 2l-1

        # D^{l,1}: collect all odd i < j
        rows1, cols1, vals1 = [], [], []
        for i in range(1, n + 1):  # 1-based
            if i < j_col and (i % 2 == 1):
                # Add both (i-1, j_col-1) and (j_col-1, i-1) for symmetry
                rows1.extend([i - 1, j_col - 1])
                cols1.extend([j_col - 1, i - 1])
                vals1.extend([1.0, 1.0])
        if rows1:  # Only add if non-empty
            coo_mats.append((np.array(rows1, dtype=int), np.array(cols1, dtype=int), np.array(vals1, dtype=float)))

        # D^{l,2}: collect all even i < j
        rows2, cols2, vals2 = [], [], []
        for i in range(1, n + 1):  # 1-based
            if i < j_col and (i % 2 == 0):
                rows2.extend([i - 1, j_col - 1])
                cols2.extend([j_col - 1, i - 1])
                vals2.extend([1.0, 1.0])
        if rows2:  # Only add if non-empty
            coo_mats.append((np.array(rows2, dtype=int), np.array(cols2, dtype=int), np.array(vals2, dtype=float)))

    return SymBasis(n=n, coo_mats=coo_mats, name=f"mixed_fbm_markovian_N={N}")


def invest_value_fbm(H, n, strategy, method="newton", Sigma=None, Lambda=None, basis=None,
                     tol=1e-8, max_iter=500, verbose=False, x_init=None, cg_max_iter=200):
    """
    Compute investment value for pure fBM model with a given strategy.

    Strategies:
      - "markovian": Uses TridiagC_Basis (tridiagonal constraint)
      - "full": General strategy, no constraint (closed-form)

    Parameters
    ----------
    H : float
        Hurst parameter.
    n : int
        Matrix dimension.
    strategy : str
        "markovian" or "full".
    method : str
        Optimization method for decomposition.
    Sigma : np.ndarray, optional
        Pre-computed covariance matrix. Built if not provided.
    Lambda : np.ndarray, optional
        Pre-computed precision matrix. Built if not provided.
    basis : SymBasis, optional
        Pre-computed basis for markovian strategy.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    verbose : bool
        Print solver progress.
    x_init : np.ndarray, optional
        Initial guess for warm starting.

    Returns
    -------
    value : float
        Log investment value.
    info : dict
        Additional info: {"iters": int, "time": float, "method": str, "error": str or None, "x": array}
    """
    info = {"iters": 0, "time": 0.0, "method": strategy, "error": None, "x": None}
    t_start = time.time()

    try:
        # === Step 1: Build covariance matrix Sigma ===
        if Sigma is None:
            Sigma = spd_fractional_BM(n, H=H, T=1.0)

        if not is_spd(Sigma):
            info["error"] = "Sigma not SPD"
            return np.nan, info

        # === Step 2: Compute log|Σ| ===
        _, log_det_Sigma = np.linalg.slogdet(Sigma)

        # === Step 3: Compute B (strategy-specific) ===
        if strategy == "full":
            # General strategy: closed-form from differential covariance
            A_diff = spd_fractional_BM(n, H=H, T=1.0, diff_flag=True)
            value = invest_value_general(A_diff)
            info["method"] = "closed-form"
            info["time"] = time.time() - t_start
            return value, info
        elif strategy == "markovian":
            # Markovian: constrained decomposition with TridiagC_Basis
            # Note: This requires Lambda = Sigma^{-1}. For ill-conditioned Sigma,
            # the results may be less accurate but we try anyway.
            if basis is None:
                basis = TridiagC_Basis(n)

            # Compute Lambda even if ill-conditioned - try our best
            try:
                Lambda = spd_inverse(Sigma)
            except np.linalg.LinAlgError:
                info["error"] = "Cholesky failed for Sigma"
                info["time"] = time.time() - t_start
                return np.nan, info

            actual_method = "newton-cg" if method == "lbfgs" else method

            B, _, x, decomp_info = constrained_decomposition(
                A=Lambda, basis=basis, method=actual_method,
                tol=tol, max_iter=max_iter, verbose=verbose, return_info=True,
                x_init=x_init, cg_max_iter=cg_max_iter
            )
            info["iters"] = decomp_info["iters"]
            info["method"] = decomp_info.get("used_method", actual_method)
            info["x"] = x
        else:
            raise ValueError(f"Unknown strategy for fbm: {strategy}")

        # === Step 4: Compute value ===
        # For markovian: value = 0.5 * (log|Σ| - log|B|)
        _, log_det_B = np.linalg.slogdet(B)
        value = 0.5 * (log_det_Sigma - log_det_B)

    except Exception as e:
        info["error"] = str(e)
        value = np.nan

    info["time"] = time.time() - t_start
    return value, info


def invest_value_mixed_fbm(H, N, alpha, delta_t, strategy, method="newton", solver="primal",
                           Sigma=None, Lambda=None, basis=None, basis_perp=None,
                           tol=1e-8, max_iter=500, verbose=False, x_init=None, cg_max_iter=200):
    """
    Compute investment value for mixed fBM model with a given strategy.

    This is the unified value computation function for all three strategies:
      - "sum": Observes only the mixed index X (N observations, N×N matrix)
      - "markovian": Observes both X and W with Markovian constraints (2N obs, 2N×2N matrix)
      - "full": Observes both X and W with full information (2N obs, 2N×2N matrix)

    Value formula: log(Value) = 0.5 × (log|Σ| - log|B|)

    The only difference between strategies is how B is computed:
      - "sum": B is diagonal with B_ii = 1/Λ_ii (closed-form solution)
      - "markovian"/"full": B comes from constrained decomposition Λ = B^{-1} + C

    Parameters
    ----------
    H : float
        Hurst parameter.
    N : int
        Number of time steps.
    alpha : float
        Weight of fBM component.
    delta_t : float
        Time step size.
    strategy : str
        "sum", "markovian", or "full".
    method : str
        Optimization method for decomposition ("newton", "newton-cg", "quasi-newton").
    solver : str
        "primal" or "dual" (for full strategy only).
    Sigma : np.ndarray, optional
        Pre-computed covariance matrix. Built if not provided.
    Lambda : np.ndarray, optional
        Pre-computed precision matrix. Built if not provided.
    basis : SymBasis, optional
        Pre-computed basis for the strategy. Built if not provided.
    basis_perp : SymBasis, optional
        Pre-computed orthogonal complement basis (for dual solver).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    verbose : bool
        Print solver progress.

    Returns
    -------
    value : float
        Log investment value.
    info : dict
        Additional info: {"iters": int, "time": float, "method": str, "error": str or None}
    """
    info = {"iters": 0, "time": 0.0, "method": strategy, "error": None, "x": None}
    t_start = time.time()
    t_sigma = 0.0
    t_lambda = 0.0
    t_basis = 0.0
    t_solve = 0.0
    t_logdet = 0.0

    try:
        # === Step 1: Build covariance matrix Sigma (N×N for sum, 2N×2N for others) ===
        if Sigma is None:
            t0 = time.time()
            if strategy == "sum":
                Sigma = spd_sum_fbm(N, H=H, alpha=alpha, delta_t=delta_t)
            else:
                Sigma = spd_mixed_fbm(N, H=H, alpha=alpha, delta_t=delta_t)
            t_sigma = time.time() - t0

        if not is_spd(Sigma):
            info["error"] = "Sigma not SPD"
            return np.nan, info

        # === Step 2: Compute precision matrix Lambda (as needed) ===
        if Lambda is None:
            t0 = time.time()
            Lambda = spd_inverse(Sigma)
            t_lambda = time.time() - t0

        if not is_spd(Lambda):
            info["error"] = "Lambda not SPD"
            return np.nan, info

        # === Step 3: Compute log|Σ| (shared across all strategies) ===
        t0 = time.time()
        _, log_det_Sigma = np.linalg.slogdet(Sigma)
        t_logdet += time.time() - t0

        # === Step 4: Compute B / log|B| (strategy-specific) ===
        if strategy == "sum":
            # Closed-form: optimal diagonal B has B_ii = 1/Λ_ii
            t0 = time.time()
            log_det_B = -float(np.sum(np.log(np.diag(Lambda))))
            value = 0.5 * (log_det_Sigma - log_det_B)
            t_logdet += time.time() - t0
            info["method"] = "diagonal B"
            info["time"] = time.time() - t_start
            if verbose:
                print(
                    f"  [timing] sigma={t_sigma:.3f}s lambda={t_lambda:.3f}s "
                    f"logdet={t_logdet:.3f}s solve={t_solve:.3f}s"
                )
            return value, info
        else:
            # Decomposition: solve Λ = B^{-1} + C with C ⊥ S
            if basis is None:
                t0 = time.time()
                if strategy == "markovian":
                    basis = make_mixed_fbm_markovian_basis(N)
                elif strategy == "full":
                    basis = make_mixed_fbm_full_info_basis(N)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                t_basis = time.time() - t0

            if strategy == "full" and solver == "dual":
                if basis_perp is None:
                    t0 = time.time()
                    basis_perp = make_orthogonal_complement_basis(basis)
                    t_basis += time.time() - t0
                t0 = time.time()
                B, _, _, _ = constrained_decomposition_dual(
                    A=Lambda, basis=basis, basis_perp=basis_perp,
                    tol=tol, max_iter=max_iter, verbose=verbose
                )
                t_solve = time.time() - t0
                info["iters"] = "?"
                info["method"] = "dual"
            elif method == "lbfgs" and strategy == "markovian":
                # Use BlockedNewtonSolver with L-BFGS (fastest for large N)
                from toeplitz_solver import BlockedNewtonSolver
                t0 = time.time()
                solver_obj = BlockedNewtonSolver(N, H, alpha, delta_t, verbose=verbose)
                B, _, x, solve_info = solver_obj.solve_lbfgs(tol=tol, max_iter=max_iter)
                t_solve = time.time() - t0
                info["iters"] = solve_info["iters"]
                info["method"] = "lbfgs"
                info["x"] = x
            elif method in ("newton", "newton-cg", "precond-newton-cg") and strategy == "markovian":
                # Use BlockedNewtonSolver for markovian - exploits Toeplitz structure
                from toeplitz_solver import BlockedNewtonSolver
                t0 = time.time()
                solver_obj = BlockedNewtonSolver(N, H, alpha, delta_t, verbose=verbose)
                # Map method: precond-newton-cg uses newton-cg with preconditioning
                solver_method = "newton-cg" if method == "precond-newton-cg" else method
                use_precond = (method == "precond-newton-cg")  # only precond-newton-cg uses preconditioning
                B, _, x, solve_info = solver_obj.solve(tol=tol, max_iter=max_iter, method=solver_method, use_precond=use_precond)
                t_solve = time.time() - t0
                info["iters"] = solve_info["iters"]
                info["method"] = method
                info["x"] = x
                info["timing_detail"] = solve_info.get("timing", {})
            elif method == "lbfgs":
                # L-BFGS only works efficiently for markovian strategy (handled above)
                # For full strategy, Newton-CG is actually faster because:
                # 1. Hessian-vector products are computed efficiently via CG
                # 2. Newton steps are very effective for log-det problems
                # 3. Generic L-BFGS doesn't capture the Hessian structure well
                if verbose:
                    print(f"  Note: Using newton-cg for {strategy} strategy (faster than L-BFGS for this problem)")
                t0 = time.time()
                B, _, x, decomp_info = constrained_decomposition(
                    A=Lambda, basis=basis, method="newton-cg",
                    tol=tol, max_iter=max_iter, verbose=verbose, return_info=True,
                    x_init=x_init, cg_max_iter=cg_max_iter
                )
                t_solve = time.time() - t0
                info["iters"] = decomp_info["iters"]
                info["method"] = "newton-cg"
                info["x"] = x
            else:
                # For full strategy, auto-disable preconditioning (too expensive: m Hv products for m~N²)
                # Preconditioning is only worth it for markovian where m~N is small
                actual_method = method
                if strategy == "full" and method == "precond-newton-cg":
                    actual_method = "newton-cg"
                    if verbose:
                        print(f"  [Auto] Using newton-cg for full strategy (precond too expensive for m={basis.m})")

                t0 = time.time()
                B, _, x, decomp_info = constrained_decomposition(
                    A=Lambda, basis=basis, method=actual_method,
                    tol=tol, max_iter=max_iter, verbose=verbose, return_info=True,
                    x_init=x_init, cg_max_iter=cg_max_iter
                )
                t_solve = time.time() - t0
                info["iters"] = decomp_info["iters"]
                info["method"] = decomp_info.get("used_method", actual_method)
                info["x"] = x  # Return for warm starting next iteration

        # === Step 5: Compute log|B| and final value (shared formula) ===
        t0 = time.time()
        _, log_det_B = np.linalg.slogdet(B)
        value = 0.5 * (log_det_Sigma - log_det_B)
        t_logdet += time.time() - t0

    except Exception as e:
        info["error"] = str(e)
        value = np.nan

    info["time"] = time.time() - t_start
    if verbose:
        timing_detail = info.get("timing_detail", {})
        t_precond = timing_detail.get("precond", 0.0)
        if t_precond > 0:
            print(
                f"  [timing] sigma={t_sigma:.3f}s lambda={t_lambda:.3f}s "
                f"basis={t_basis:.3f}s precond={t_precond:.3f}s solve={t_solve:.3f}s logdet={t_logdet:.3f}s"
            )
        else:
            print(
                f"  [timing] sigma={t_sigma:.3f}s lambda={t_lambda:.3f}s "
                f"basis={t_basis:.3f}s solve={t_solve:.3f}s logdet={t_logdet:.3f}s"
            )
    return value, info


def compute_value_vs_H_mixed_fbm(H_vec, N=50, alpha=1.0, delta_t=None, solver="primal", method="newton", strategy="both"):
    """
    Compute investment value vs Hurst parameter H for mixed fBM model.

    Parameters
    ----------
    H_vec : array-like
        Vector of Hurst parameters to evaluate.
    N : int
        Number of time steps. Matrix size is 2N x 2N.
    alpha : float
        Weight of fBM component in mixed index.
    delta_t : float, optional
        Time step size. Defaults to 1/N.
    solver : str
        "primal" or "dual" (for full strategy).
    method : str
        Optimization method ("newton", "newton-cg", "quasi-newton").
    strategy : str
        "both", "markovian", "full", or "sum".

    Returns
    -------
    val_markovian : np.ndarray or None
    val_full_info : np.ndarray or None
    val_sum_fbm : np.ndarray
    """
    if delta_t is None:
        delta_t = 1.0 / N

    n_H = len(H_vec)
    run_sum = True  # Always compute sum for comparison
    run_markovian = strategy in ("both", "markovian")
    run_full = strategy in ("both", "full")

    val_sum_fbm = np.zeros(n_H)
    val_markovian = np.zeros(n_H) if run_markovian else None
    val_full_info = np.zeros(n_H) if run_full else None

    n = 2 * N

    # Pre-build bases for efficiency (reused across H values)
    basis_markov = make_mixed_fbm_markovian_basis(N) if run_markovian else None
    basis_full = make_mixed_fbm_full_info_basis(N) if run_full else None
    basis_full_perp = None

    print(f"Mixed fBM: N={N}, matrix size={n}x{n}, delta_t={delta_t:.6f}")
    print(f"Strategy: {strategy}")

    if run_markovian:
        print(f"Markovian basis dimension: {basis_markov.m}")
    if run_full:
        print(f"Full-info basis dimension: {basis_full.m}")
        if solver == "dual":
            print("Building orthogonal complement basis...")
            t0 = time.time()
            basis_full_perp = make_orthogonal_complement_basis(basis_full)
            print(f"  Built S⊥ (dim={basis_full_perp.m}) in {time.time()-t0:.2f}s")

    total_start = time.time()

    # Warm start: keep track of previous solutions
    x_markov_prev = None
    x_full_prev = None

    for i, H in enumerate(H_vec):
        print(f"\n--- H = {H:.4f} ({i+1}/{n_H}) ---")

        # Build shared Sigma/Lambda for markovian/full (sum builds its own)
        Sigma = spd_mixed_fbm(N, H=H, alpha=alpha, delta_t=delta_t)
        Lambda = spd_inverse(Sigma) if is_spd(Sigma) else None

        # --- Sum strategy ---
        # Note: Don't pass Sigma/Lambda - sum needs N×N matrix, not 2N×2N
        val, info = invest_value_mixed_fbm(
            H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="sum"
        )
        val_sum_fbm[i] = val
        if info["error"]:
            print(f"  Sum: FAILED - {info['error']}")
        else:
            print(f"  Sum: {val:.6f} ({info['time']:.2f}s)")

        # --- Markovian strategy ---
        if run_markovian:
            val, info = invest_value_mixed_fbm(
                H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="markovian",
                method=method, Sigma=Sigma, Lambda=Lambda, basis=basis_markov,
                tol=5e-8, x_init=x_markov_prev
            )
            val_markovian[i] = val
            x_markov_prev = info.get("x")  # Update warm start for next H
            if info["error"]:
                print(f"  Markovian: FAILED - {info['error']}")
            else:
                print(f"  Markovian: {val:.6f} ({info['time']:.2f}s, {info['iters']} iters, {info['method']})")

        # --- Full strategy ---
        if run_full:
            val, info = invest_value_mixed_fbm(
                H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="full",
                method=method, solver=solver, Sigma=Sigma, Lambda=Lambda,
                basis=basis_full, basis_perp=basis_full_perp, x_init=x_full_prev
            )
            val_full_info[i] = val
            x_full_prev = info.get("x")  # Update warm start for next H
            if info["error"]:
                print(f"  Full-info: FAILED - {info['error']}")
            else:
                print(f"  Full-info: {val:.6f} ({info['time']:.2f}s, {info['iters']} iters, {info['method']})")

    total_time = time.time() - total_start
    print(f"\n=== Total time: {total_time:.2f} seconds ({total_time/n_H:.2f} sec/H value) ===")

    return val_markovian, val_full_info, val_sum_fbm


def _compute_single_H(args):
    """Worker function for parallel H computation."""
    H, N, alpha, delta_t, method, strategy, basis_markov_data, basis_full_data = args

    run_markovian = strategy in ("both", "markovian")
    run_full = strategy in ("both", "full")

    result = {"H": H, "val_markovian": np.nan, "val_full_info": np.nan, "val_sum_fbm": np.nan,
              "iters_markov": 0, "iters_full": 0, "time": 0, "error": None,
              "time_sum": 0, "time_markov": 0, "time_full": 0}

    t_start = time.time()

    try:
        # Rebuild bases from pre-generated data
        n_basis = 2 * N
        basis_markov = SymBasis(n=n_basis, coo_mats=basis_markov_data, name=f"mixed_fbm_markovian_N={N}") if run_markovian else None
        basis_full = SymBasis(n=n_basis, coo_mats=basis_full_data, name=f"mixed_fbm_full_info_N={N}") if run_full else None

        # Build shared Sigma/Lambda
        Sigma = spd_mixed_fbm(N, H=H, alpha=alpha, delta_t=delta_t)
        Lambda = spd_inverse(Sigma) if is_spd(Sigma) else None

        # Sum strategy
        # Note: Don't pass Sigma/Lambda - sum needs N×N matrix, not 2N×2N
        val, info = invest_value_mixed_fbm(
            H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="sum"
        )
        result["val_sum_fbm"] = val
        result["time_sum"] = info["time"]
        if info["error"]:
            print(f"  [H={H:.4f}] Sum FAILED: {info['error']}")

        # Markovian strategy
        if run_markovian:
            val, info = invest_value_mixed_fbm(
                H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="markovian",
                method="newton", Sigma=Sigma, Lambda=Lambda, basis=basis_markov, tol=1e-6
            )
            result["val_markovian"] = val
            result["iters_markov"] = info["iters"]
            result["time_markov"] = info["time"]
            if info["error"]:
                result["error"] = info["error"]

        # Full-info strategy
        if run_full:
            val, info = invest_value_mixed_fbm(
                H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="full",
                method="newton-cg", Sigma=Sigma, Lambda=Lambda, basis=basis_full
            )
            result["val_full_info"] = val
            result["iters_full"] = info["iters"]
            result["time_full"] = info["time"]
            if info["error"]:
                result["error"] = info["error"]

    except Exception as e:
        result["error"] = str(e)

    result["time"] = time.time() - t_start
    return result


def compute_value_vs_H_mixed_fbm_parallel(H_vec, N=50, alpha=1.0, delta_t=None,
                                          method="newton", strategy="markovian", workers=None):
    """
    Parallel version of compute_value_vs_H_mixed_fbm.

    Parameters
    ----------
    workers : int or None
        Number of worker processes. Default: cpu_count - 2
    """
    if delta_t is None:
        delta_t = 1.0 / N

    n_H = len(H_vec)
    run_markovian = strategy in ("both", "markovian")
    run_full = strategy in ("both", "full")

    if workers is None:
        workers = max(1, os.cpu_count() - 2)

    n = 2 * N
    print(f"Mixed fBM (PARALLEL): N={N}, matrix size={n}x{n}, delta_t={delta_t:.6f}")
    print(f"Strategy: {strategy}, Workers: {workers}")
    print(f"H values: {n_H} (from {H_vec[0]:.4f} to {H_vec[-1]:.4f})")

    # --- Pre-generate basis data to avoid re-computing in every worker ---
    print("Pre-generating basis data for parallel workers...")
    t_basis_start = time.time()
    
    basis_markov_coo_data = None
    if run_markovian:
        markov_coo_mats = []
        for l in range(1, N + 1):
            j_col = 2 * l - 1
            rows1, cols1, vals1 = [], [], []
            for i in range(1, n + 1):
                if i < j_col and (i % 2 == 1):
                    rows1.extend([i - 1, j_col - 1])
                    cols1.extend([j_col - 1, i - 1])
                    vals1.extend([1.0, 1.0])
            if rows1:
                markov_coo_mats.append((np.array(rows1, dtype=int), np.array(cols1, dtype=int), np.array(vals1, dtype=float)))
            
            rows2, cols2, vals2 = [], [], []
            for i in range(1, n + 1):
                if i < j_col and (i % 2 == 0):
                    rows2.extend([i - 1, j_col - 1])
                    cols2.extend([j_col - 1, i - 1])
                    vals2.extend([1.0, 1.0])
            if rows2:
                markov_coo_mats.append((np.array(rows2, dtype=int), np.array(cols2, dtype=int), np.array(vals2, dtype=float)))
        basis_markov_coo_data = markov_coo_mats

    basis_full_coo_data = None
    if run_full:
        full_coo_mats = []
        for l in range(1, N + 1):
            j_col = 2 * l - 1 - 1
            for k in range(1, 2 * l - 1):
                i_row = k - 1
                rows = np.array([i_row, j_col], dtype=int)
                cols = np.array([j_col, i_row], dtype=int)
                vals = np.array([1.0, 1.0], dtype=float)
                full_coo_mats.append((rows, cols, vals))
        basis_full_coo_data = full_coo_mats
        
    print(f"... basis data generated in {time.time() - t_basis_start:.2f}s")
    print()  # blank line before results

    # Prepare arguments for workers (basis rebuilt in each worker)
    args_list = [(H, N, alpha, delta_t, method, strategy, basis_markov_coo_data, basis_full_coo_data) for H in H_vec]

    total_start = time.time()

    # Store results by H value for correct ordering at end
    results_dict = {}
    completed = 0

    # Run in parallel with imap_unordered for streaming results
    with mp.Pool(processes=workers) as pool:
        for res in pool.imap_unordered(_compute_single_H, args_list):
            completed += 1
            results_dict[res["H"]] = res

            # Print progress immediately as each H completes
            if res["error"]:
                print(f"  [{completed:3d}/{n_H}] H={res['H']:.4f}: ERROR - {res['error']}")
            else:
                info_str = []
                info_str.append(f"sum={res['val_sum_fbm']:.6f} [{res['time_sum']:.1f}s]")
                if run_markovian:
                    info_str.append(f"markov={res['val_markovian']:.6f} ({res['iters_markov']} it) [{res['time_markov']:.1f}s]")
                if run_full:
                    info_str.append(f"full={res['val_full_info']:.6f} ({res['iters_full']} it) [{res['time_full']:.1f}s]")
                print(f"  [{completed:3d}/{n_H}] H={res['H']:.4f}: {', '.join(info_str)} [total={res['time']:.1f}s]", flush=True)

    # Collect results in original H order
    val_markovian = np.zeros(n_H) if run_markovian else None
    val_full_info = np.zeros(n_H) if run_full else None
    val_sum_fbm = np.zeros(n_H)

    for i, H in enumerate(H_vec):
        res = results_dict[H]
        if run_markovian:
            val_markovian[i] = res["val_markovian"]
        if run_full:
            val_full_info[i] = res["val_full_info"]
        val_sum_fbm[i] = res["val_sum_fbm"]

    total_time = time.time() - total_start
    print(f"\n=== Total time: {total_time:.2f} seconds ({total_time/n_H:.2f} sec/H value) ===")
    print(f"=== Parallel speedup: {workers}x theoretical, actual depends on load balance ===")

    return val_markovian, val_full_info, val_sum_fbm


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Finance example: fBM or mixed fBM")
    parser.add_argument("--model", type=str, choices=["fbm", "mixed_fbm"], default="mixed_fbm",
                        help="Model type: 'fbm' or 'mixed_fbm' (default: mixed_fbm)")
    parser.add_argument("--n", type=int, default=100,
                        help="Matrix dimension (default: 100). For mixed_fbm, N=n//2 time steps.")
    parser.add_argument("--solver", type=str, choices=["primal", "dual"], default="primal",
                        help="Solver for full-info: 'primal' or 'dual' (Newton on S⊥)")
    parser.add_argument("--method", type=str, choices=["newton", "newton-cg", "precond-newton-cg", "quasi-newton", "lbfgs"], default="newton-cg",
                        help="Optimization method: 'newton' (auto-switches to newton-cg for large m), "
                             "'newton-cg' (matrix-free), 'precond-newton-cg' (with diagonal preconditioning), "
                             "'quasi-newton' (BFGS), or 'lbfgs' (L-BFGS)")
    parser.add_argument("--strategy", type=str, choices=["both", "markovian", "full"], default="both",
                        help="Which strategies to run: 'both', 'markovian' (fast, O(N)), or 'full' (slow, O(N²))")
    parser.add_argument("--hres", type=float, default=0.1,
                        help="H resolution step size (default: 0.1). E.g., 0.1 gives H=0.1,0.2,...,0.9; "
                             "0.02 gives H=0.02,0.04,...,0.98")
    parser.add_argument("--hmin", type=float, default=0.0,
                        help="Minimum H value (default: 0.0). H range starts at max(hmin, hres).")
    parser.add_argument("--hmax", type=float, default=1.0,
                        help="Maximum H value exclusive (default: 1.0). H range ends before hmax.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight of fBM component in mixed index (default: 1.0). "
                             "Higher alpha increases fBM influence vs BM.")
    parser.add_argument("--parallel", action="store_true",
                        help="Run H values in parallel using multiprocessing")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 2)")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Force recomputation even if cached results exist")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only plot from cached results, don't run computation")
    parser.add_argument("--plot-all", action="store_true",
                        help="Generate plots for all (model, n, alpha) combinations in results CSV")
    parser.add_argument("--plot-hmin", type=float, default=None,
                        help="Minimum H for plotting (default: use hmin). Filters cached data for plotting.")
    parser.add_argument("--plot-hmax", type=float, default=None,
                        help="Maximum H for plotting (default: use hmax). Filters cached data for plotting.")
    parser.add_argument("--show-title", action="store_true",
                        help="Show title on plots (default: no title)")
    parser.add_argument("--incremental", action="store_true",
                        help="Save results incrementally after each H value. Resume from existing results.")
    parser.add_argument("--max-cond", type=float, default=1e6,
                        help="Maximum condition number for Sigma matrix. Skip H values exceeding this (default: 1e6).")
    parser.add_argument("--cg-max-iter", type=int, default=500,
                        help="Maximum CG iterations for newton-cg solver (default: 500).")
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="Convergence tolerance for optimization (default: 1e-8). "
                             "Try 1e-4 for faster but less precise results.")
    parser.add_argument("--sort-h-by-center", action="store_true",
                        help="Sort H values for optimal warm start: 0.5->1 (ascending), then 0.5->0 (descending). "
                             "Enables warm start to build from well-conditioned center outward.")
    parser.add_argument("--reverse-h", action="store_true",
                        help="Process H values in descending order (hmax to hmin). "
                             "Useful for warm start when going from 0.5 down to 0.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output from optimization solvers (show iteration progress).")
    parser.add_argument("--heatmap", action="store_true",
                        help="Generate decomposition heatmap (slow, disabled by default).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute values but don't save to CSV or plot. For diagnostics.")
    parser.add_argument("--warm-start", action="store_true", default=True,
                        help="Use linear extrapolation warm start from previous H values (default: enabled).")
    parser.add_argument("--no-warm-start", action="store_false", dest="warm_start",
                        help="Disable warm start (start from zero for each H).")
    args = parser.parse_args()

    model_type = args.model
    n = args.n
    solver = args.solver
    method = args.method
    strategy = args.strategy
    hres = args.hres
    hmin = args.hmin
    hmax = args.hmax
    alpha = args.alpha
    parallel = args.parallel
    workers = args.workers
    force_rerun = args.force_rerun
    plot_only = args.plot_only
    plot_all = args.plot_all
    plot_hmin = args.plot_hmin
    plot_hmax = args.plot_hmax
    show_title = args.show_title
    incremental = args.incremental
    dry_run = args.dry_run

    # Both dry_run and force_rerun should use incremental code path (has warm start, verbose)
    # They just differ in whether results are saved
    if dry_run or force_rerun:
        incremental = True  # Use incremental path for warm start support
    warm_start = args.warm_start
    max_cond = args.max_cond
    cg_max_iter = args.cg_max_iter
    tol = args.tol
    sort_h_by_center = args.sort_h_by_center
    reverse_h = args.reverse_h
    verbose_solver = args.verbose

    if workers is None:
        workers = max(1, os.cpu_count() - 2)

    # --- Experiment settings ---
    # H range: from max(hmin, hres) to hmax (exclusive), with step hres
    h_start = max(hmin, hres) if hmin == 0.0 else hmin
    H_vec = np.arange(h_start, hmax, hres)

    # Optionally sort H values for optimal warm start: 0.5->1, then 0.5->0
    if sort_h_by_center:
        # Split into upper (H >= 0.5) and lower (H < 0.5) parts
        # Upper: sorted ascending (0.5, 0.6, ..., 1.0) - warm start builds up
        # Lower: sorted descending (0.49, 0.48, ..., 0.01) - warm start builds down
        upper = np.sort(H_vec[H_vec >= 0.5])  # ascending
        lower = np.sort(H_vec[H_vec < 0.5])[::-1]  # descending
        H_vec = np.concatenate([upper, lower])
        print(f"H values sorted for warm start (0.5->1, then 0.5->0):")
        if len(H_vec) <= 12:
            print(f"  Order: {', '.join([f'{h:.2f}' for h in H_vec])}")
        else:
            print(f"  Order: {', '.join([f'{h:.2f}' for h in H_vec[:5]])}, ..., "
                  f"{', '.join([f'{h:.2f}' for h in H_vec[-5:]])}")

    # Optionally reverse H values (for warm start going from 0.5 down to 0)
    if reverse_h:
        H_vec = H_vec[::-1]
        print(f"H values reversed (descending): {H_vec[0]:.2f} -> {H_vec[-1]:.2f}")

    if model_type == "fbm":
        N = n
        delta_t = 1.0
    else:  # mixed_fbm
        N = n // 2  # Number of time steps (matrix is 2N x 2N)
        delta_t = 1.0 / N  # Time step for consistent scaling between sum and mixed

    # Prepare params dict for saving
    params = {
        'model': model_type,
        'n': n,
        'N': N,
        'alpha': alpha,
        'delta_t': delta_t,
        'strategy': strategy,
    }

    results_file = get_results_file()

    if plot_all:
        # --- Plot all mode: generate plots for all (model, n, alpha) combinations ---
        combos = get_all_param_combinations()
        if not combos:
            print("ERROR: No results found in CSV file")
            exit(1)

        # In plot-all mode, use --hmin/--hmax as plot range if --plot-hmin/--plot-hmax not specified
        if plot_hmin is None and hmin != 0.0:
            plot_hmin = hmin
        if plot_hmax is None and hmax != 1.0:  # default hmax in arg parser is 1.0
            plot_hmax = hmax

        print(f"\n{'='*60}")
        print(f"Plot-all mode: found {len(combos)} unique (model, n, alpha) combinations")
        if plot_hmin is not None or plot_hmax is not None:
            print(f"H range filter: [{plot_hmin or 0.0:.2f}, {plot_hmax or 1.0:.2f}]")
        print(f"{'='*60}\n")

        here = Path(__file__).resolve().parent
        n_plotted = 0

        for model_i, n_i, alpha_i in combos:
            n_i = int(n_i)
            # Use 'all' to merge results from 'both', 'full', and 'markovian' strategy runs
            H_vec_i, val_markov_i, val_general_i, val_sum_i = load_results_for_params(model_i, n_i, alpha_i, 'all')
            if H_vec_i is None or len(H_vec_i) == 0:
                print(f"  Skipping {model_i}, n={n_i}, alpha={alpha_i}: no data")
                continue

            # Apply H range filter
            p_hmin = plot_hmin if plot_hmin is not None else H_vec_i[0]
            p_hmax = plot_hmax if plot_hmax is not None else H_vec_i[-1] + 1e-9
            mask = (H_vec_i >= p_hmin) & (H_vec_i <= p_hmax)

            if not np.any(mask):
                print(f"  Skipping {model_i}, n={n_i}, alpha={alpha_i}: no data in H range [{p_hmin}, {p_hmax}]")
                continue

            H_plot = H_vec_i[mask]
            val_markov_plot = val_markov_i[mask] if val_markov_i is not None else None
            val_general_plot = val_general_i[mask] if val_general_i is not None else None
            val_sum_plot = val_sum_i[mask] if val_sum_i is not None else None

            # Check if we have any non-NaN values to plot
            has_markov = val_markov_plot is not None and not np.all(np.isnan(val_markov_plot))
            has_general = val_general_plot is not None and not np.all(np.isnan(val_general_plot))
            has_sum = val_sum_plot is not None and not np.all(np.isnan(val_sum_plot))

            if not (has_markov or has_general or has_sum):
                print(f"  Skipping {model_i}, n={n_i}, alpha={alpha_i}: all values are NaN")
                continue

            # Create plot
            fig, ax = plt.subplots(figsize=(8, 5))
            if has_markov:
                ax.plot(H_plot, val_markov_plot, 'b-o', label="Markovian", markersize=4)
            if has_general:
                ax.plot(H_plot, val_general_plot, 'r-s', label="Full-information", markersize=4)
            if has_sum:
                ax.plot(H_plot, val_sum_plot, 'g-^', label="Sum (no decomp)", markersize=4)
            ax.set_xlabel(r'$\mathcal{H}$', fontsize=14)
            ax.set_ylabel(r'$v_N^*$', fontsize=14)

            if model_i == "mixed_fbm":
                N_i = n_i // 2
                if show_title:
                    ax.set_title(f"Mixed fBM: Strategy value vs H (N={N_i}, α={alpha_i})", fontsize=13)
                # Add H=3/4 vertical line (no legend entry) with text at top
                if H_plot[0] <= 0.75 <= H_plot[-1]:
                    ax.axvline(x=0.75, color='gray', linestyle='--', alpha=0.5)
                    # Add "0.75" text at top of plot area
                    ax.text(0.75, 1.02, '0.75', transform=ax.get_xaxis_transform(),
                           ha='center', va='bottom', fontsize=10, color='gray')
            else:
                if show_title:
                    ax.set_title(f"fBM: Strategy value vs H (n={n_i})", fontsize=13)

            ax.legend(fontsize=9, framealpha=0.8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            # Save figure
            fig_dir = here / "figs" / model_i
            fig_dir.mkdir(parents=True, exist_ok=True)
            # Always include H range in filename for consistency
            hmin_str = f"{plot_hmin:.2f}" if plot_hmin is not None else "0.00"
            hmax_str = f"{plot_hmax:.2f}" if plot_hmax is not None else "1.00"
            h_range_str = f"H_{hmin_str}_{hmax_str}"
            alpha_str = f"_a{alpha_i:.1f}" if model_i == "mixed_fbm" and alpha_i != 1.0 else ""
            out_png = fig_dir / f"value_{model_i}_n_{n_i}_{h_range_str}{alpha_str}_all.png"
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"  Saved: {out_png}")
            n_plotted += 1

        print(f"\n{'='*60}")
        print(f"Generated {n_plotted} plots")
        print(f"{'='*60}")
        exit(0)

    if plot_only:
        # --- Plot only mode: load from master CSV ---
        H_vec, val_markov, val_general, val_sum = load_results_for_params(model_type, n, alpha, strategy)
        if H_vec is None:
            print(f"ERROR: --plot-only specified but no results found for model={model_type}, n={n}, alpha={alpha}, strategy={strategy}")
            exit(1)
        print(f"\n{'='*60}")
        print(f"Plot-only mode: loaded {len(H_vec)} results")
        print(f"{'='*60}\n")

    elif incremental or not force_rerun:
        # --- Incremental mode: compute missing H values one by one ---
        print(f"\n{'='*60}")
        print(f"INCREMENTAL mode: {model_type}, n={n}, strategy={strategy}")
        print(f"H range: [{hmin}, {hmax}) with step {hres}, alpha={alpha}")
        print(f"Max condition number: {max_cond:.0e}, CG max iter: {cg_max_iter}, tol: {tol:.0e}")
        print(f"Results file: {results_file}")
        print(f"{'='*60}\n")

        # Load already computed H values (skip for dry_run/force_rerun - recompute everything)
        if dry_run or force_rerun:
            completed_H = set()
            mode = "Dry run" if dry_run else "Force rerun"
            print(f"[{mode}] Forcing recomputation of all H values")
        else:
            completed_H = get_completed_H_values(model_type, n, alpha, strategy)
            if completed_H:
                print(f"Found {len(completed_H)} already computed H values")

        run_markovian = strategy in ("both", "markovian")
        run_full = strategy in ("both", "full")

        # Pre-build bases (model-specific)
        if model_type == "fbm":
            basis_markov = TridiagC_Basis(n) if run_markovian else None
            basis_full = None  # Full strategy for fbm is closed-form, no basis needed
        else:  # mixed_fbm
            basis_markov = make_mixed_fbm_markovian_basis(N) if run_markovian else None
            basis_full = make_mixed_fbm_full_info_basis(N) if run_full else None

        total_start = time.time()
        n_computed = 0
        n_skipped_done = 0
        n_skipped_cond = 0

        # Warm start with linear extrapolation:
        # x_init(h) = 2*x(h-δ) - x(h-2δ)  (predicts based on trend)
        # Falls back to x(h-δ) if only one previous solution available
        # Feasibility projection is done in constrained_decomposition (shrinks toward zero)
        x_markov_prev = None      # x(h-δ)
        x_markov_prev_prev = None # x(h-2δ)
        x_full_prev = None
        x_full_prev_prev = None

        def extrapolate_warm_start(x_prev, x_prev_prev):
            """Linear extrapolation: 2*x_prev - x_prev_prev, or just x_prev if no prev_prev."""
            if not warm_start:
                return None
            if x_prev is None:
                return None
            if x_prev_prev is None:
                return x_prev
            return 2 * x_prev - x_prev_prev

        if warm_start:
            print(f"Warm start: enabled (linear extrapolation from previous H values)")

        for i, H in enumerate(H_vec):
            H_rounded = round(H, 6)
            if H_rounded in completed_H:
                n_skipped_done += 1
                continue

            print(f"\n--- H = {H:.4f} ({i+1}/{len(H_vec)}), method={method} ---")

            # Build Sigma and check condition number (model-specific)
            if model_type == "fbm":
                Sigma = spd_fractional_BM(n, H=H, T=1.0)
            else:
                Sigma = spd_mixed_fbm(N, H=H, alpha=alpha, delta_t=delta_t)

            cond = np.linalg.cond(Sigma)
            ill_conditioned = cond > max_cond

            # For fbm: full strategy uses A_diff (well-conditioned), only markovian needs Lambda
            # For mixed_fbm: all strategies need Lambda, so skip if ill-conditioned
            if ill_conditioned and model_type != "fbm":
                print(f"  SKIPPED: cond(Sigma)={cond:.2e} > {max_cond:.0e}")
                n_skipped_cond += 1
                continue

            # Compute Lambda only if needed and well-conditioned
            Lambda = None
            if not ill_conditioned and is_spd(Sigma):
                Lambda = spd_inverse(Sigma)

            if Lambda is None and model_type != "fbm":
                print(f"  SKIPPED: Sigma not SPD or ill-conditioned")
                n_skipped_cond += 1
                continue

            v_sum = None
            v_markov = None
            v_full = None

            if model_type == "fbm":
                # === Pure fBM: 2 strategies (markovian, full) ===
                # Markovian strategy - uses direct barrier method, works for all H!
                if run_markovian:
                    x_init_markov = extrapolate_warm_start(x_markov_prev, x_markov_prev_prev)
                    v_markov, info = invest_value_fbm(
                        H=H, n=n, strategy="markovian", method=method,
                        Sigma=Sigma, basis=basis_markov,
                        tol=tol, verbose=verbose_solver, cg_max_iter=cg_max_iter,
                        x_init=x_init_markov
                    )
                    if info["error"]:
                        print(f"  Markovian: FAILED - {info['error']}")
                        v_markov = np.nan
                    else:
                        print(f"  Markovian: {v_markov:.6f} ({info['time']:.2f}s, {info['iters']} iters)")
                        # Update warm start history (shift)
                        x_markov_prev_prev = x_markov_prev
                        x_markov_prev = info.get("x")

                # Full strategy (closed-form using A_diff - always well-conditioned!)
                if run_full:
                    v_full, info = invest_value_fbm(
                        H=H, n=n, strategy="full", method=method,
                        Sigma=None, Lambda=None  # Not needed for full strategy
                    )
                    if info["error"]:
                        print(f"  Full: FAILED - {info['error']}")
                        v_full = np.nan
                    else:
                        print(f"  Full: {v_full:.6f} ({info['time']:.2f}s, {info['method']})")

            else:
                # === Mixed fBM: 3 strategies (sum, markovian, full) ===
                # Sum strategy (always computed for mixed_fbm)
                # Note: Don't pass Sigma/Lambda - sum needs N×N matrix, not 2N×2N
                v_sum, info = invest_value_mixed_fbm(
                    H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="sum"
                )
                print(f"  Sum: {v_sum:.6f} ({info['time']:.2f}s)")

                # Markovian strategy
                if run_markovian:
                    x_init_markov = extrapolate_warm_start(x_markov_prev, x_markov_prev_prev)
                    v_markov, info = invest_value_mixed_fbm(
                        H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="markovian",
                        method=method, Sigma=Sigma, Lambda=Lambda, basis=basis_markov,
                        tol=tol, verbose=verbose_solver, cg_max_iter=cg_max_iter,
                        x_init=x_init_markov
                    )
                    if info["error"]:
                        print(f"  Markovian: FAILED - {info['error']}")
                        v_markov = np.nan
                    else:
                        print(f"  Markovian: {v_markov:.6f} ({info['time']:.2f}s, {info['iters']} iters)")
                        # Update warm start history (shift)
                        x_markov_prev_prev = x_markov_prev
                        x_markov_prev = info.get("x")

                # Full-info strategy
                if run_full:
                    x_init_full = extrapolate_warm_start(x_full_prev, x_full_prev_prev)
                    v_full, info = invest_value_mixed_fbm(
                        H=H, N=N, alpha=alpha, delta_t=delta_t, strategy="full",
                        method=method, Sigma=Sigma, Lambda=Lambda, basis=basis_full,
                        tol=tol, verbose=verbose_solver, cg_max_iter=cg_max_iter,
                        x_init=x_init_full
                    )
                    if info["error"]:
                        print(f"  Full-info: FAILED - {info['error']}")
                        v_full = np.nan
                    else:
                        print(f"  Full-info: {v_full:.6f} ({info['time']:.2f}s, {info['iters']} iters)")
                        # Update warm start history (shift)
                        x_full_prev_prev = x_full_prev
                        x_full_prev = info.get("x")

            # Save incrementally to master CSV (unless dry run)
            if not dry_run:
                append_result(H, v_sum, v_markov, v_full, params)
            n_computed += 1

        total_time = time.time() - total_start
        print(f"\n=== Completed: {n_computed} computed, {n_skipped_done} already done, {n_skipped_cond} skipped (ill-conditioned) ===")
        print(f"=== Total time: {total_time:.2f}s ===")

        # Load all results for plotting
        H_vec, val_markov, val_general, val_sum = load_results_for_params(model_type, n, alpha, strategy)
        if H_vec is None:
            print("No results to plot.")
            exit(0)

    else:
        # --- Full batch computation (force rerun) ---
        print(f"\n{'='*60}")
        print(f"FORCE RERUN: {model_type}, n={n}, solver={solver}, method={method}, strategy={strategy}")
        print(f"H range: [{hmin}, {hmax}) with step {hres}, alpha={alpha}")
        if parallel:
            print(f"PARALLEL mode: {workers} workers")
        print(f"{'='*60}\n")

        if model_type == "fbm":
            val_markov, val_general = compute_value_vs_H_fbm(H_vec, n=n)
            val_sum = None
        else:  # mixed_fbm
            if parallel:
                val_markov, val_general, val_sum = compute_value_vs_H_mixed_fbm_parallel(
                    H_vec, N=N, alpha=alpha, delta_t=delta_t, method=method, strategy=strategy, workers=workers
                )
            else:
                val_markov, val_general, val_sum = compute_value_vs_H_mixed_fbm(
                    H_vec, N=N, alpha=alpha, delta_t=delta_t, solver=solver, method=method, strategy=strategy
                )

        # Save all results to master CSV (unless dry run)
        if not dry_run:
            for i, H in enumerate(H_vec):
                v_sum = val_sum[i] if val_sum is not None else None
                v_markov = val_markov[i] if val_markov is not None else None
                v_full = val_general[i] if val_general is not None else None
                append_result(H, v_sum, v_markov, v_full, params)

    # --- Skip plotting in dry-run mode ---
    if dry_run:
        print("\n[Dry run] Skipping save to CSV and plotting.")
    else:
        # --- Filter data for plotting if plot range specified ---
        p_hmin = plot_hmin if plot_hmin is not None else H_vec[0]
        p_hmax = plot_hmax if plot_hmax is not None else H_vec[-1] + 1e-9  # inclusive
        mask = (H_vec >= p_hmin) & (H_vec <= p_hmax)

        if not np.any(mask):
            print(f"WARNING: No data in plot range [{p_hmin}, {p_hmax}]. Available: [{H_vec[0]:.2f}, {H_vec[-1]:.2f}]")
        else:
            H_plot = H_vec[mask]
            val_markov_plot = val_markov[mask] if val_markov is not None else None
            val_general_plot = val_general[mask] if val_general is not None else None
            val_sum_plot = val_sum[mask] if val_sum is not None else None

            if plot_hmin is not None or plot_hmax is not None:
                print(f"Plotting H range: [{H_plot[0]:.2f}, {H_plot[-1]:.2f}] ({len(H_plot)} points)")

        # --- Plot ---
        # Use filled markers for computed values, empty markers for missing values
        fig, ax = plt.subplots(figsize=(8, 5))

        def plot_with_missing(ax, H, vals, color, marker, label):
            """Plot computed values with filled markers, missing with empty markers."""
            if vals is None:
                return
            computed_mask = ~np.isnan(vals)
            missing_mask = np.isnan(vals)

            # Computed values: filled markers with line
            if np.any(computed_mask):
                ax.plot(H[computed_mask], vals[computed_mask],
                        color=color, linestyle='-', marker=marker,
                        markersize=5, label=label, markerfacecolor=color)

            # Missing values: empty markers at bottom of plot (y=0 or min)
            if np.any(missing_mask):
                # Use small y value to show missing points at bottom
                y_missing = np.zeros(np.sum(missing_mask))
                ax.plot(H[missing_mask], y_missing,
                        color=color, linestyle='', marker=marker,
                        markersize=5, markerfacecolor='none', markeredgecolor=color,
                        alpha=0.5, label=f'{label} (missing)' if np.any(computed_mask) else label)

        plot_with_missing(ax, H_plot, val_markov_plot, 'blue', 'o', 'Markovian')
        plot_with_missing(ax, H_plot, val_general_plot, 'red', 's', 'Full-information')
        plot_with_missing(ax, H_plot, val_sum_plot, 'green', '^', 'Sum (no decomp)')
        ax.set_xlabel(r'$\mathcal{H}$', fontsize=14)
        ax.set_ylabel(r'$v_N^*$', fontsize=14)
        if model_type == "mixed_fbm":
            if show_title:
                ax.set_title(f"Mixed fBM: Strategy value vs H (N={N}, α={alpha})", fontsize=13)
            # Add H=3/4 vertical line (no legend entry) with text at top
            if H_plot[0] <= 0.75 <= H_plot[-1]:
                ax.axvline(x=0.75, color='gray', linestyle='--', alpha=0.5)
                ax.text(0.75, 1.02, '0.75', transform=ax.get_xaxis_transform(),
                       ha='center', va='bottom', fontsize=10, color='gray')
        else:
            if show_title:
                ax.set_title(f"fBM: Strategy value vs H (n={n})", fontsize=13)
        ax.legend(fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # --- Save in figs/ next to this script ---
        here = Path(__file__).resolve().parent
        fig_dir = here / "figs" / model_type
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Always include H range in filename for consistency
        hmin_str = f"{plot_hmin:.2f}" if plot_hmin is not None else "0.00"
        hmax_str = f"{plot_hmax:.2f}" if plot_hmax is not None else "1.00"
        h_range_str = f"H_{hmin_str}_{hmax_str}"
        alpha_str = f"_a{alpha:.1f}" if model_type == "mixed_fbm" and alpha != 1.0 else ""
        out_png = fig_dir / f"value_{model_type}_n_{n}_{h_range_str}{alpha_str}_{strategy}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"\nSaved value figure to: {out_png}")


        # ---- Save decomposition heatmaps for a chosen H (optional, slow) ----
        if args.heatmap:
            H0 = 0.8  # pick one (or loop over a few)

            if model_type == "fbm":
                A = spd_fractional_BM(n, H=H0, T=1.0, diff_flag=True)
                A_inv = spd_inverse(A)
                basis = TridiagC_Basis(n)
            else:  # mixed_fbm
                Sigma = spd_mixed_fbm(N, H=H0, alpha=alpha, delta_t=delta_t)
                A_inv = spd_inverse(Sigma)
                basis = make_mixed_fbm_markovian_basis(N)  # Use Markovian for heatmap

            B0, C0, x0 = constrained_decomposition(
                A=A_inv,
                basis=basis,
                method="newton",
                tol=1e-6,
                max_iter=500,
                verbose=False
            )

            out_heat = fig_dir / f"heatmap_{model_type}_H_{H0:.2f}_n_{n}.png"
            plot_decomposition_heatmaps(
                A=A_inv,
                B=B0,
                C=C0,
                basis=basis,
                out_file=out_heat,
            )
            print(f"Saved heatmap to: {out_heat}")
