"""
Matrix and basis generators used for demos/experiments.

This module contains:
  - SPD matrix generators (Hilbert, AR(1) Toeplitz, Brownian, Gaussian kernel, fractional BM, ...)
  - Convenient basis constructors for structured subspaces (banded, selected off-diagonals, block patterns, ...)
  - Small helpers for block/group demo construction

It should not contain solvers or plotting.
"""


import numpy as np
from constrained_decomposition_core import SymBasis, block_reynolds_project



def spd_hilbert(n):
    """Hilbert matrix, dense SPD."""
    i = np.arange(1, n+1)
    j = np.arange(1, n+1)
    return 1.0 / (i[:, None] + j[None, :] - 1)

def spd_toeplitz_ar1(n: int, rho: float = 0.8, sigma2: float = 1.0) -> np.ndarray:
    """
    SPD Toeplitz covariance/correlation matrix:
        A[i,j] = sigma2 * rho^{|i-j|}
    This is the covariance of a stationary AR(1) process (up to scaling).
    Guaranteed SPD for any n when abs(rho) < 1 and sigma2 > 0.
    """
    if not (isinstance(n, int) and n >= 1):
        raise ValueError("n must be a positive integer")
    if not (abs(rho) < 1.0):
        raise ValueError("Need abs(rho) < 1 for SPD Toeplitz AR(1)")
    if not (sigma2 > 0):
        raise ValueError("Need sigma2 > 0")

    idx = np.arange(n)
    # Toeplitz via |i-j|
    A = sigma2 * (rho ** np.abs(idx[:, None] - idx[None, :]))
    # Symmetry is exact, but keep it clean numerically:
    return  0.5 * (A + A.T)

def spd_brownian(n):
    """Brownian-motion covariance matrix: K_ij = min(i,j)."""
    i = np.arange(1, n+1)
    j = np.arange(1, n+1)
    return np.minimum(i[:, None], j[None, :]).astype(float)

def spd_gaussian_kernel(n, gamma=0.1):
    """Gaussian kernel matrix: e^{-gamma (i-j)^2}."""
    i = np.arange(n)
    j = np.arange(n)
    return np.exp(-gamma * (i[:, None] - j[None, :])**2)

def spd_fractional_BM(n, H=0.5, T=1.0, diff_flag=False):
    """
    Fractional Brownian motion covariance matrix on an equispaced grid.

    A[i,j] = 0.5 * (T/n)^(2H) * ( i^(2H) + j^(2H) - |i-j|^(2H) )
    with i,j = 1,...,n (1-based indices in the formula).

    For diff:
    A[i,j] = 0.5 * (T/n)^(2H) * ( (i-j-1)^(2H) + (i-j+1)^(2H) - 2*|i-j|^(2H) )

    Parameters
    ----------
    n : int
        Matrix size.
    H : float, default 0.5
        Hurst parameter in (0,1).
    T : float, default 1.0
        Final time horizon.
    diff_flag: Show covariance of differences
    """
    i = np.arange(1, n + 1, dtype=float)
    j = np.arange(1, n + 1, dtype=float)
    I = i[:, None]
    J = j[None, :]

    factor = 0.5 * (T / n) ** (2.0 * H)

    if not diff_flag:
        A = factor * (I ** (2.0 * H) + J ** (2.0 * H) - np.abs(I - J) ** (2.0 * H))
    else:
        A = factor * (np.abs(I - J - 1) ** (2.0 * H) + np.abs(I - J + 1) ** (2.0 * H) - 2.0 * np.abs(I - J) ** (2.0 * H))
    return A





def make_random_spd(n: int, seed: int = 0, diag_boost: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M @ M.T + (diag_boost * n) * np.eye(n)
    return 0.5 * (A + A.T)

def make_banded_spd(n: int, b: int, seed: int = 0, diag_boost: float = 5.0) -> np.ndarray:
    """
    Dense representation of a symmetric banded SPD matrix (bandwidth b).
    (In code we still store dense; structure is in the pattern.)
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=float)
    for k in range(b + 1):
        v = rng.standard_normal(n - k)
        if k == 0:
            A += np.diag(v)
        else:
            A += np.diag(v, k) + np.diag(v, -k)
    # boost diagonal to ensure SPD
    A += (diag_boost * (b + 1)) * np.eye(n)
    return 0.5 * (A + A.T)

def make_offdiag_pair_basis(n: int, pairs):
    mats = []
    for (i, j) in pairs:
        D = np.zeros((n, n), dtype=float)
        D[i, j] = 1.0
        D[j, i] = 1.0
        mats.append(D)
    return SymBasis(n=n, dense_mats=mats, name=f"offdiag_pairs_m={len(mats)}")

def make_banded_basis(n: int, b: int, include_diag: bool = True):
    """
    Basis for symmetric banded matrices with half-bandwidth b.
    include_diag=True => includes diagonal E_ii.
    """
    mats = []
    if include_diag:
        for i in range(n):
            D = np.zeros((n, n), dtype=float)
            D[i, i] = 1.0
            mats.append(D)
    for k in range(1, b + 1):
        for i in range(n - k):
            j = i + k
            D = np.zeros((n, n), dtype=float)
            D[i, j] = 1.0
            D[j, i] = 1.0
            mats.append(D)
    return SymBasis(n, dense_mats=mats, name=f"banded(b={b}, diag={include_diag})")


def make_banded_basis_coo(n: int, b: int, include_diag: bool = True):
    """
    Sparse COO basis for symmetric banded matrices with half-bandwidth b.
    Each basis matrix has 1 (diag) or 2 (offdiag) non-zeros.
    """
    coo_mats = []
    if include_diag:
        for i in range(n):
            rows = np.array([i], dtype=int)
            cols = np.array([i], dtype=int)
            vals = np.array([1.0], dtype=float)
            coo_mats.append((rows, cols, vals))
    for k in range(1, b + 1):
        for i in range(n - k):
            j = i + k
            rows = np.array([i, j], dtype=int)
            cols = np.array([j, i], dtype=int)
            vals = np.array([1.0, 1.0], dtype=float)
            coo_mats.append((rows, cols, vals))
    return SymBasis(n=n, coo_mats=coo_mats, name=f"banded_coo(b={b}, diag={include_diag})")

def make_blocks(n: int, r: int):
    """
    Partition {0,...,n-1} into r contiguous blocks (sizes differ by at most 1).
    """
    r = int(r)
    if r < 1 or r > n:
        raise ValueError("blocks r must satisfy 1 <= r <= n")
    sizes = [n // r] * r
    for t in range(n % r):
        sizes[t] += 1
    blocks = []
    start = 0
    for sz in sizes:
        blocks.append(list(range(start, start + sz)))
        start += sz
    return blocks

def make_block_fixed_spd(n: int, r: int, seed: int = 0):
    """
    Create a random SPD A and Reynolds-project it to be fixed under block-permutations.
    """
    A0 = make_random_spd(n, seed=seed, diag_boost=2.0)
    blocks = make_blocks(n, r)
    A = block_reynolds_project(A0, blocks)
    # ensure strictly SPD by diagonal shift if needed
    lam_min = float(np.min(np.linalg.eigvalsh(A)))
    if lam_min <= 1e-8:
        A = A + (abs(lam_min) + 1e-2) * np.eye(n)
    return A, blocks

def make_block_fixed_basis_offdiag(n: int, blocks):
    """
    Build a *small* basis for S^G consisting of block-constant OFF-DIAGONAL patterns:
      - one matrix per block: within-block off-diagonal entries = 1 (diag=0)
      - one matrix per block-pair: between-block entries = 1 (both rectangles)
    This ensures S ∩ SPSD = {0} (zero diagonal).
    """
    mats = []

    # within-block off-diagonal
    for I in blocks:
        D = np.zeros((n, n), dtype=float)
        for a in I:
            for b in I:
                if a != b:
                    D[a, b] = 1.0
        mats.append(D)

    # between blocks
    for bi in range(len(blocks)):
        for bj in range(bi + 1, len(blocks)):
            I = blocks[bi]
            J = blocks[bj]
            D = np.zeros((n, n), dtype=float)
            for a in I:
                for b in J:
                    D[a, b] = 1.0
                    D[b, a] = 1.0
            mats.append(D)

    return SymBasis(n=n, dense_mats=mats, name=f"block_fixed_offdiag_r={len(blocks)}_m={len(mats)}")


def make_blocks_variable(n: int, sizes=None, r: int = None, seed: int = 0):
    """
    Return a list of index lists (0-based). Either provide explicit sizes (sum to n),
    or sample random positive sizes for r blocks.
    """
    rng = np.random.default_rng(seed)

    if sizes is None:
        assert r is not None and r >= 1
        # random positive sizes summing to n
        cuts = np.sort(rng.choice(np.arange(1, n), size=r-1, replace=False))
        sizes = np.diff(np.concatenate(([0], cuts, [n]))).tolist()
    else:
        assert sum(sizes) == n and all(s > 0 for s in sizes)

    blocks = []
    start = 0
    for s in sizes:
        blocks.append(list(range(start, start + s)))
        start += s
    return blocks


def make_block_constant_spd(blocks,
                            seed: int = 0,
                            diag_range=(4.0, 7.0),
                            within_range=(0.4, 1.0),
                            cross_range=(1.2, 2.4),
                            diag_margin=0.5,
                            spd_method="eigenvalue"):
    """
    Construct a symmetric block-constant matrix A with:
      - different diagonal levels per block
      - different within-block offdiag per block
      - different cross-block constants per pair (typically larger for visibility)

    Then enforce SPD using the specified method.

    Parameters
    ----------
    spd_method : str
        "eigenvalue" - shift eigenvalues to make min eigenvalue = diag_margin (preserves structure)
        "diagonal_dominance" - strict diagonal dominance (obscures off-diagonal structure)
    """
    rng = np.random.default_rng(seed)
    n = sum(len(I) for I in blocks)
    k = len(blocks)

    d = rng.uniform(*diag_range, size=k)          # diagonal per block
    u = rng.uniform(*within_range, size=k)        # within-block offdiag per block

    # cross-block constants (symmetric)
    c = np.zeros((k, k))
    for a in range(k):
        for b in range(a+1, k):
            c[a, b] = c[b, a] = rng.uniform(*cross_range)

    A = np.zeros((n, n), dtype=float)

    # fill blocks
    for a, Ia in enumerate(blocks):
        for ii, i in enumerate(Ia):
            A[i, i] = d[a]
            for jj, j in enumerate(Ia):
                if i != j:
                    A[i, j] = u[a]

    # fill cross-block
    for a in range(k):
        for b in range(a+1, k):
            Ia, Ib = blocks[a], blocks[b]
            for i in Ia:
                for j in Ib:
                    A[i, j] = A[j, i] = c[a, b]

    # Make SPD
    if spd_method == "eigenvalue":
        # Shift eigenvalues: A <- A + (diag_margin - lambda_min) * I
        # This preserves the block structure visually
        eigvals = np.linalg.eigvalsh(A)
        lambda_min = eigvals[0]
        shift = diag_margin - lambda_min
        if shift > 0:
            A = A + shift * np.eye(n)
    else:  # diagonal_dominance
        # Strict diagonal dominance (Gershgorin / sufficient for SPD)
        for i in range(n):
            off = np.sum(np.abs(A[i, :])) - abs(A[i, i])
            A[i, i] = off + diag_margin + abs(A[i, i])

    return A, {"d": d, "u": u, "c": c}



def spd_mixed_fbm(N: int, H: float = 0.75, alpha: float = 1.0, delta_t: float = 1.0) -> np.ndarray:
    """
    Mixed fractional Brownian motion covariance matrix.

    Models a discrete-time market with two processes:
      - X_i^1 = W_{i*dt} - W_{(i-1)*dt} + alpha * (B^H_{i*dt} - B^H_{(i-1)*dt})  (mixed index)
      - X_i^2 = W_{i*dt} - W_{(i-1)*dt}  (pure Brownian increment)

    The covariance matrix Sigma is 2N x 2N with entries:
      Sigma[i,j] = (alpha^2 / 2^{1+2H}) * dt^{2H} * (|i-j+2|^{2H} + |i-j-2|^{2H} - 2|i-j|^{2H})
                   if i,j are both odd (fBm increment covariance between mixed indices)
      Sigma[i,j] = dt   if j is even and (i == j or i == j-1)
      Sigma[i,j] = dt   if j is odd and i == j+1
      Sigma[i,j] = 0    otherwise

    Parameters
    ----------
    N : int
        Number of time steps. Matrix size is 2N x 2N.
    H : float, default 0.75
        Hurst parameter in (0,1). For H > 3/4, the mixed model is arbitrage-free.
    alpha : float, default 1.0
        Weight of the fractional component in the mixed index.
    delta_t : float, default 1.0
        Time step size.

    Returns
    -------
    Sigma : np.ndarray
        The 2N x 2N covariance matrix (SPD).

    References
    ----------
    Cheridito (2001), "Mixed fractional Brownian motion", Bernoulli 7(6):913-934.
    """
    n = 2 * N
    Sigma = np.zeros((n, n), dtype=float)

    # Precompute fBm increment factor
    factor = (alpha ** 2) / (2 ** (1 + 2 * H)) * (delta_t ** (2 * H))

    # Indices: odd positions (0-based even) are mixed, even positions (0-based odd) are pure BM
    odd = np.arange(0, n, 2)
    even = odd + 1

    # fBM increment covariance for odd-odd block (Toeplitz-like)
    t = np.arange(N)
    diff = t[:, None] - t[None, :]
    fbm_cov = (
        np.abs(diff + 1) ** (2 * H)
        + np.abs(diff - 1) ** (2 * H)
        - 2 * np.abs(diff) ** (2 * H)
    )
    Sigma[np.ix_(odd, odd)] = factor * fbm_cov

    # Add BM variance on diagonal for even indices (pure BM increments)
    Sigma[even, even] = delta_t

    # Add mixed index variance on diagonal for odd indices
    Sigma[odd, odd] += delta_t

    # Cross terms between mixed (odd) and pure BM (even) at the same time step
    Sigma[odd, even] = delta_t
    Sigma[even, odd] = delta_t

    return Sigma


def spd_mixed_fbm_blocked(N: int, H: float = 0.75, alpha: float = 1.0, delta_t: float = 1.0) -> np.ndarray:
    """
    Mixed fractional Brownian motion covariance matrix in BLOCKED ordering.

    This version uses blocked variable ordering Z = (X₁,...,X_N, Y₁,...,Y_N) where:
      - X_i = W_{i*dt} - W_{(i-1)*dt} + alpha * (B^H_{i*dt} - B^H_{(i-1)*dt})  (mixed index)
      - Y_i = W_{i*dt} - W_{(i-1)*dt}  (pure Brownian increment)

    The covariance matrix has 2×2 block structure:

        Σ = [Δt I_N + α² Γ_H    Δt I_N ]
            [    Δt I_N          Δt I_N ]

    where Γ_H is the N×N Toeplitz fGn (fractional Gaussian noise) covariance matrix.

    The upper-left block (X-X covariance) has Toeplitz structure, which enables
    specialized fast solvers exploiting the Toeplitz structure.

    Parameters
    ----------
    N : int
        Number of time steps. Matrix size is 2N x 2N.
    H : float, default 0.75
        Hurst parameter in (0,1). For H > 3/4, the mixed model is arbitrage-free.
    alpha : float, default 1.0
        Weight of the fractional component in the mixed index.
    delta_t : float, default 1.0
        Time step size.

    Returns
    -------
    Sigma : np.ndarray
        The 2N x 2N covariance matrix (SPD) in blocked ordering.

    Notes
    -----
    This is equivalent to spd_mixed_fbm but with blocked variable ordering.
    The interleaved ordering is (X₁,Y₁,X₂,Y₂,...), while blocked is (X₁,...,X_N,Y₁,...,Y_N).

    The blocked ordering enables exploitation of the Toeplitz structure in the
    upper-left block for faster matrix operations via Schur complement.

    See Also
    --------
    spd_mixed_fbm : Interleaved ordering version
    fgn_cov_toeplitz : Returns just the Γ_H Toeplitz matrix
    """
    n = 2 * N

    # === Build the Toeplitz fGn covariance (N×N) ===
    # Use the SAME formula as spd_mixed_fbm for exact equivalence:
    #   factor = (alpha ** 2) / (2 ** (1 + 2 * H)) * (delta_t ** (2 * H))
    #   fbm_cov = |k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H}  (note: no 0.5 factor)
    #
    # Combined: factor * fbm_cov = alpha^2 * delta_t^{2H} / 2^{1+2H} * fbm_cov
    factor = (alpha ** 2) / (2 ** (1 + 2 * H)) * (delta_t ** (2 * H))

    k = np.arange(N)
    gamma = (
        np.abs(k + 1) ** (2 * H)
        + np.abs(k - 1) ** (2 * H)
        - 2 * np.abs(k) ** (2 * H)
    )

    # Build full Toeplitz matrix
    from scipy.linalg import toeplitz
    fbm_cov = toeplitz(gamma)

    # === Construct the 2N×2N block matrix ===
    Sigma = np.zeros((n, n), dtype=float)

    # Upper-left block: Δt I_N + α² Γ_H (fBM + BM variance)
    Sigma[:N, :N] = factor * fbm_cov + delta_t * np.eye(N)

    # Upper-right block: Δt I_N (X-Y cross-correlation from shared BM)
    Sigma[:N, N:] = delta_t * np.eye(N)

    # Lower-left block: Δt I_N (Y-X cross-correlation)
    Sigma[N:, :N] = delta_t * np.eye(N)

    # Lower-right block: Δt I_N (Y-Y: pure BM variance)
    Sigma[N:, N:] = delta_t * np.eye(N)

    return Sigma


def fgn_cov_toeplitz(N: int, H: float = 0.75, alpha: float = 1.0, delta_t: float = 1.0) -> np.ndarray:
    """
    Compute the scaled Toeplitz fGn (fractional Gaussian noise) covariance matrix.

    Uses the SAME scaling as spd_mixed_fbm for consistency:
        factor = (alpha ** 2) / (2 ** (1 + 2 * H)) * (delta_t ** (2 * H))
        gamma(k) = |k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H}  (note: no 0.5 factor)

    The result is factor * Toeplitz(gamma).

    Parameters
    ----------
    N : int
        Matrix dimension.
    H : float, default 0.75
        Hurst parameter in (0,1).
    alpha : float, default 1.0
        Weight of fBM component.
    delta_t : float, default 1.0
        Time step size.

    Returns
    -------
    Gamma_H : np.ndarray
        The N×N scaled Toeplitz covariance matrix.
    """
    factor = (alpha ** 2) / (2 ** (1 + 2 * H)) * (delta_t ** (2 * H))
    k = np.arange(N)
    gamma = (
        np.abs(k + 1) ** (2 * H)
        + np.abs(k - 1) ** (2 * H)
        - 2 * np.abs(k) ** (2 * H)
    )
    from scipy.linalg import toeplitz
    return factor * toeplitz(gamma)


def spd_sum_fbm(n: int, H: float = 0.75, alpha: float = 1.0, delta_t: float = None) -> np.ndarray:
    """
    Sum of BM and fBM covariance matrix (increments).

    Models observing the sum process: X_i = W_i - W_{i-1} + alpha * (B^H_i - B^H_{i-1})
    where W is standard BM and B^H is fBM with Hurst parameter H, assumed independent.

    The covariance matrix Gamma is n x n with entries:
      Gamma_ij = dt * delta_ij + (alpha^2 / 2^{1+2H}) * dt^{2H} * (|i-j+1|^{2H} + |i-j-1|^{2H} - 2|i-j|^{2H})

    where delta_ij is Kronecker delta and dt is the time step.

    Parameters
    ----------
    n : int
        Number of time steps / matrix size.
    H : float, default 0.75
        Hurst parameter in (0,1). For H > 3/4, arbitrage-free.
    alpha : float, default 1.0
        Weight of the fractional component relative to BM.
    delta_t : float, optional
        Time step size. If None, defaults to 1/n for consistency with spd_mixed_fbm.

    Returns
    -------
    Gamma : np.ndarray
        The n x n covariance matrix (SPD).

    Notes
    -----
    This is simpler than spd_mixed_fbm which has a 2N x 2N interleaved structure.
    Here we only observe the sum, not the BM separately.
    Uses same scaling as spd_mixed_fbm for consistent comparison.
    """
    if delta_t is None:
        delta_t = 1.0 / n

    # fBM increment covariance factor - SAME as spd_mixed_fbm
    factor = (alpha ** 2) / (2 ** (1 + 2 * H)) * (delta_t ** (2 * H))

    t = np.arange(n)
    diff = t[:, None] - t[None, :]
    fbm_cov = (
        np.abs(diff + 1) ** (2.0 * H)
        + np.abs(diff - 1) ** (2.0 * H)
        - 2.0 * np.abs(diff) ** (2.0 * H)
    )
    Gamma = factor * fbm_cov

    # Add BM variance on diagonal: delta_t * delta_ij (same as spd_mixed_fbm)
    Gamma[np.diag_indices(n)] += delta_t

    return Gamma


def invest_value_sum_fbm(Gamma) -> float:
    """
    Compute log investment value for sum-observation strategy using decomposition.

    The trader only observes the sum of BM and fBM (one observation per time step).
    This corresponds to a Markovian constraint on the N×N sum covariance matrix:
    the optimal B is diagonal (no cross-time information in the strategy).

    We solve: Λ = B^{-1} + C with C having zero diagonal (B is diagonal).
    For diagonal B, the optimal solution is B_ii = 1/Λ_ii.

    Value formula (same as mixed strategies):
        log(Value) = 0.5 × (log|Γ| - log|B|)

    Parameters
    ----------
    Gamma : np.ndarray
        The n x n covariance matrix from spd_sum_fbm.

    Returns
    -------
    log_value : float
        The log of the investment value.
    """
    n = Gamma.shape[0]
    Lambda = np.linalg.inv(Gamma)

    # Optimal diagonal B: B_ii = 1/Λ_ii (Markovian strategy on sum observations)
    B_diag = np.diag(1.0 / np.diag(Lambda))

    _, log_det_Gamma = np.linalg.slogdet(Gamma)
    _, log_det_B = np.linalg.slogdet(B_diag)

    # Same formula as decomposition strategies
    log_value = 0.5 * (log_det_Gamma - log_det_B)

    return log_value


def make_block_support_basis_offdiag(n: int, blocks, active_pairs=None, active_within=True):
    """
    Robust basis for a big G-invariant subspace S from an active block-support pattern.
    Accepts blocks as list-of-lists OR dict -> block indices, possibly numpy arrays.
    Ensures indices are 0-based ints in [0, n-1].
    """

    def _as_int_list(block):
        # block can be list/np.array/etc
        arr = np.asarray(block).reshape(-1)

        # cast to int (handles float arrays like [0.,1.,2.])
        idx = [int(x) for x in arr.tolist()]

        return idx

    # --- normalize blocks container: dict -> list, then each block -> list[int]
    if isinstance(blocks, dict):
        # order blocks deterministically by their smallest index
        block_list = list(blocks.values())
        block_list = sorted(block_list, key=lambda b: float(np.min(np.asarray(b))))
        blocks = block_list

    blocks = [_as_int_list(b) for b in blocks]

    # --- detect and fix 1-based indexing (common gotcha)
    all_idx = [i for blk in blocks for i in blk]
    if len(all_idx) == 0:
        raise ValueError("blocks is empty")
    mn, mx = min(all_idx), max(all_idx)
    if mn == 1 and mx == n:
        blocks = [[i - 1 for i in blk] for blk in blocks]

    # --- validate range
    for blk in blocks:
        for i in blk:
            if not (0 <= i < n):
                raise ValueError(f"Block index {i} out of range [0,{n-1}]. "
                                 f"(Are blocks 1-based or mismatched with n?)")

    k = len(blocks)
    if active_pairs is None:
        active_pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]

    coo_mats = []

    # within-block arbitrary offdiag (COO: each matrix has only 2 non-zeros)
    if active_within:
        for a in range(k):
            I = blocks[a]
            for ii in range(len(I)):
                for jj in range(ii + 1, len(I)):
                    i, j = I[ii], I[jj]
                    rows = np.array([i, j], dtype=int)
                    cols = np.array([j, i], dtype=int)
                    vals = np.array([1.0, 1.0], dtype=float)
                    coo_mats.append((rows, cols, vals))

    # cross-block arbitrary (COO: each matrix has only 2 non-zeros)
    for (a, b) in active_pairs:
        Ia, Ib = blocks[a], blocks[b]
        for i in Ia:
            for j in Ib:
                rows = np.array([i, j], dtype=int)
                cols = np.array([j, i], dtype=int)
                vals = np.array([1.0, 1.0], dtype=float)
                coo_mats.append((rows, cols, vals))

    return SymBasis(n=n, coo_mats=coo_mats, name=f"block_support_offdiag_k={k}_m={len(coo_mats)}")
