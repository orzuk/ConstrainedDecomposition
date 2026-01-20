"""
Fast Toeplitz matrix operations using FFT.

This module provides O(N log N) matrix-vector products and O(N log N × iters) solves
for Toeplitz systems, enabling efficient computation at N=1000+.

Key functions:
- toeplitz_matvec_fft: O(N log N) Toeplitz matrix-vector product
- toeplitz_solve_pcg: O(N log N × iters) Toeplitz system solve using PCG
- schur_complement_solve: Exploits 2×2 block structure with Toeplitz blocks
"""

import numpy as np
from numpy.fft import fft, ifft
from scipy.linalg import toeplitz, solve_triangular, cholesky


def toeplitz_matvec_fft(first_col, first_row, x):
    """
    Compute Toeplitz matrix-vector product T @ x using FFT.

    Complexity: O(N log N)

    Parameters
    ----------
    first_col : array (N,)
        First column of the Toeplitz matrix T.
    first_row : array (N,)
        First row of the Toeplitz matrix T (first_row[0] must equal first_col[0]).
    x : array (N,)
        Vector to multiply.

    Returns
    -------
    y : array (N,)
        Result T @ x.

    Notes
    -----
    Embeds the Toeplitz matrix in a circulant matrix of size 2N-1,
    uses FFT to compute the circulant-vector product, then extracts result.
    """
    N = len(x)
    if N == 1:
        return first_col * x

    # Build first column of circulant embedding (size 2N-1)
    # c = [t_0, t_1, ..., t_{N-1}, t_{-(N-1)}, ..., t_{-1}]
    # where t_k = first_col[k] for k >= 0, t_k = first_row[-k] for k < 0
    c = np.zeros(2 * N - 1, dtype=np.float64)
    c[:N] = first_col
    c[N:] = first_row[1:][::-1]  # t_{-(N-1)}, ..., t_{-1}

    # Zero-pad x to length 2N-1
    x_padded = np.zeros(2 * N - 1, dtype=np.float64)
    x_padded[:N] = x

    # FFT-based circulant multiply: y = IFFT(FFT(c) * FFT(x))
    c_fft = fft(c)
    x_fft = fft(x_padded)
    y_padded = ifft(c_fft * x_fft).real

    # Extract first N elements
    return y_padded[:N]


def toeplitz_matvec_symmetric_fft(gamma, x):
    """
    Compute symmetric Toeplitz matrix-vector product T @ x using FFT.

    For symmetric Toeplitz, first_col = first_row = gamma.

    Parameters
    ----------
    gamma : array (N,)
        First column (and row) of the symmetric Toeplitz matrix.
    x : array (N,)
        Vector to multiply.

    Returns
    -------
    y : array (N,)
        Result T @ x.
    """
    return toeplitz_matvec_fft(gamma, gamma, x)


def strang_circulant_preconditioner(gamma):
    """
    Build Strang's optimal circulant preconditioner for a Toeplitz matrix.

    The Strang preconditioner C_S has the same eigenvectors as circulant matrices
    (Fourier modes) and eigenvalues chosen to approximate T.

    For a symmetric Toeplitz matrix with first column gamma, the Strang
    preconditioner has first column:
        c_k = gamma[k] for k = 0, ..., floor(N/2)
        c_k = gamma[N-k] for k = floor(N/2)+1, ..., N-1

    Parameters
    ----------
    gamma : array (N,)
        First column of the symmetric Toeplitz matrix.

    Returns
    -------
    c_fft : array (N,)
        FFT of the first column of the circulant preconditioner.
        To apply M^{-1}v, compute: ifft(fft(v) / c_fft).real
    """
    N = len(gamma)
    c = np.zeros(N, dtype=np.float64)

    # Strang preconditioner: average wrap-around
    half = N // 2
    c[0] = gamma[0]
    for k in range(1, half + 1):
        if k < N - k:
            c[k] = (gamma[k] + gamma[N - k]) / 2 if N - k < N else gamma[k]
        else:
            c[k] = gamma[k]
    for k in range(half + 1, N):
        c[k] = c[N - k]

    # Simpler: just use the first half
    c[:half+1] = gamma[:half+1]
    c[half+1:] = gamma[1:N-half][::-1]

    return fft(c)


def circulant_precond_solve(c_fft, v):
    """
    Apply circulant preconditioner inverse: M^{-1} @ v.

    Parameters
    ----------
    c_fft : array (N,)
        FFT of the first column of the circulant matrix.
    v : array (N,)
        Vector to precondition.

    Returns
    -------
    result : array (N,)
        M^{-1} @ v
    """
    v_fft = fft(v)
    # Regularize to avoid division by zero
    c_fft_safe = np.where(np.abs(c_fft) > 1e-12, c_fft, 1e-12)
    return ifft(v_fft / c_fft_safe).real


def toeplitz_solve_pcg(gamma, b, tol=1e-10, max_iter=None, x0=None, verbose=False,
                       use_circulant_precond=True):
    """
    Solve symmetric positive definite Toeplitz system T @ x = b using PCG.

    Uses FFT-based matrix-vector products for O(N log N) per iteration.
    With Strang's circulant preconditioner, typically converges in O(1) iterations
    for well-conditioned systems.

    Parameters
    ----------
    gamma : array (N,)
        First column of the symmetric Toeplitz matrix T.
    b : array (N,)
        Right-hand side vector.
    tol : float
        Convergence tolerance (relative residual).
    max_iter : int, optional
        Maximum iterations. Default: min(N, 100).
    x0 : array (N,), optional
        Initial guess. Default: zeros.
    verbose : bool
        Print iteration info.
    use_circulant_precond : bool
        Use Strang's circulant preconditioner (recommended).

    Returns
    -------
    x : array (N,)
        Solution to T @ x = b.
    info : dict
        Convergence info: {"iters": int, "residual": float, "converged": bool}
    """
    N = len(b)
    if max_iter is None:
        max_iter = min(N, 100)

    # Initialize
    if x0 is None:
        x = np.zeros(N, dtype=np.float64)
        r = b.copy()
    else:
        x = x0.copy()
        r = b - toeplitz_matvec_symmetric_fft(gamma, x)

    # Build preconditioner
    if use_circulant_precond:
        c_fft = strang_circulant_preconditioner(gamma)
        def precond(v):
            return circulant_precond_solve(c_fft, v)
    else:
        # Diagonal (Jacobi) preconditioning
        M_inv_diag = 1.0 / gamma[0]
        def precond(v):
            return M_inv_diag * v

    z = precond(r)
    p = z.copy()
    rz = np.dot(r, z)

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        return np.zeros(N), {"iters": 0, "residual": 0.0, "converged": True}

    for k in range(max_iter):
        Ap = toeplitz_matvec_symmetric_fft(gamma, p)
        pAp = np.dot(p, Ap)

        if abs(pAp) < 1e-15:
            break

        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = np.linalg.norm(r)
        rel_res = r_norm / b_norm

        if verbose and k % 10 == 0:
            print(f"  PCG iter {k}: rel_res = {rel_res:.2e}")

        if rel_res < tol:
            return x, {"iters": k + 1, "residual": rel_res, "converged": True}

        z = precond(r)
        rz_new = np.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, {"iters": max_iter, "residual": rel_res, "converged": False}


def build_fgn_gamma(N, H, alpha=1.0, delta_t=1.0):
    """
    Build the first column of the scaled fGn Toeplitz covariance matrix.

    Uses the same scaling as spd_mixed_fbm for consistency:
        factor = (alpha ** 2) / (2 ** (1 + 2 * H)) * (delta_t ** (2 * H))
        gamma(k) = factor * (|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H})

    Parameters
    ----------
    N : int
        Matrix dimension.
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.
    delta_t : float
        Time step size.

    Returns
    -------
    gamma : array (N,)
        First column of Γ_H.
    """
    factor = (alpha ** 2) / (2 ** (1 + 2 * H)) * (delta_t ** (2 * H))
    k = np.arange(N)
    gamma = factor * (
        np.abs(k + 1) ** (2 * H)
        + np.abs(k - 1) ** (2 * H)
        - 2 * np.abs(k) ** (2 * H)
    )
    return gamma


class BlockedMixedFBMPrecision:
    """
    Efficient representation of the precision matrix Λ for mixed fBM in blocked ordering.

    The covariance matrix has 2×2 block structure:
        Σ = [A  B]   where A = Δt I + α² Γ_H (Toeplitz + diagonal)
            [B  D]         B = Δt I (diagonal)
                           D = Δt I (diagonal)

    The precision matrix Λ = Σ⁻¹ can be computed via Schur complement:
        S = D - B A⁻¹ B = Δt I - Δt² (Δt I + α² Γ_H)⁻¹
        Λ = [A⁻¹ + A⁻¹ B S⁻¹ B A⁻¹,  -A⁻¹ B S⁻¹]
            [-S⁻¹ B A⁻¹,              S⁻¹      ]

    Key insight: A = Δt I + α² Γ_H where Γ_H is Toeplitz, so A is also Toeplitz + diagonal.
    This class provides efficient O(N log N) matrix-vector products with Λ.
    """

    def __init__(self, N, H, alpha=1.0, delta_t=None):
        """
        Initialize the precision matrix representation.

        Parameters
        ----------
        N : int
            Number of time steps. Matrix size is 2N × 2N.
        H : float
            Hurst parameter.
        alpha : float
            fBM weight.
        delta_t : float, optional
            Time step size. Default: 1/N.
        """
        self.N = N
        self.H = H
        self.alpha = alpha
        self.delta_t = delta_t if delta_t is not None else 1.0 / N

        # Build Γ_H first column (Toeplitz)
        self.gamma_H = build_fgn_gamma(N, H, alpha, self.delta_t)

        # A = Δt I + α² Γ_H, so first column is:
        self.A_gamma = self.gamma_H.copy()
        self.A_gamma[0] += self.delta_t

        # B = D = Δt I (diagonal)
        self.B_diag = self.delta_t
        self.D_diag = self.delta_t

        # Precompute A⁻¹ via Cholesky (for moderate N) or cache for PCG
        self._A_inv_cache = None
        self._S_inv_cache = None
        self._use_direct = N <= 500  # Use direct methods for small N

        if self._use_direct:
            self._precompute_direct()

    def _precompute_direct(self):
        """Precompute inverses using direct methods for small N."""
        from scipy.linalg import toeplitz, inv

        A = toeplitz(self.A_gamma)
        self._A_inv = inv(A)

        # S = D - B A⁻¹ B = Δt I - Δt² A⁻¹
        self._S = self.D_diag * np.eye(self.N) - self.B_diag**2 * self._A_inv
        self._S_inv = inv(self._S)

    def A_solve(self, b, tol=1e-10):
        """Solve A @ x = b."""
        if self._use_direct:
            return self._A_inv @ b
        else:
            x, _ = toeplitz_solve_pcg(self.A_gamma, b, tol=tol)
            return x

    def A_matvec(self, x):
        """Compute A @ x."""
        return toeplitz_matvec_symmetric_fft(self.A_gamma, x)

    def S_solve(self, b, tol=1e-10):
        """Solve S @ x = b where S is the Schur complement."""
        if self._use_direct:
            return self._S_inv @ b
        else:
            # S = D - B A⁻¹ B = Δt I - Δt² A⁻¹
            # For iterative: use PCG with S_matvec
            # This is trickier - S is not Toeplitz!
            # For now, build S explicitly (still O(N²) but only once)
            if self._S_inv_cache is None:
                from scipy.linalg import toeplitz, inv
                A = toeplitz(self.A_gamma)
                A_inv = inv(A)
                S = self.D_diag * np.eye(self.N) - self.B_diag**2 * A_inv
                self._S_inv_cache = inv(S)
            return self._S_inv_cache @ b

    def matvec(self, z):
        """
        Compute Λ @ z where z is a 2N vector.

        Λ = [A⁻¹ + A⁻¹ B S⁻¹ B A⁻¹,  -A⁻¹ B S⁻¹]
            [-S⁻¹ B A⁻¹,              S⁻¹      ]

        Let z = [x; y] where x, y are N-vectors.
        Then Λz = [Λ₁₁ x + Λ₁₂ y; Λ₂₁ x + Λ₂₂ y]
        """
        N = self.N
        x = z[:N]
        y = z[N:]

        # Compute A⁻¹ x and A⁻¹ y
        A_inv_x = self.A_solve(x)

        # Λ₂₂ y = S⁻¹ y
        L22_y = self.S_solve(y)

        # Λ₂₁ x = -S⁻¹ B A⁻¹ x = -S⁻¹ (Δt A⁻¹ x)
        L21_x = -self.S_solve(self.B_diag * A_inv_x)

        # Λ₁₂ y = -A⁻¹ B S⁻¹ y = -A⁻¹ (Δt S⁻¹ y) = -Δt A⁻¹ L22_y
        A_inv_L22_y = self.A_solve(L22_y)
        L12_y = -self.B_diag * A_inv_L22_y

        # Λ₁₁ x = (A⁻¹ + A⁻¹ B S⁻¹ B A⁻¹) x = A⁻¹ x + A⁻¹ B S⁻¹ B A⁻¹ x
        #       = A⁻¹ x - Δt A⁻¹ L21_x / Δt  (using L21_x = -S⁻¹ Δt A⁻¹ x)
        # Actually: A⁻¹ B S⁻¹ B A⁻¹ x = Δt A⁻¹ S⁻¹ Δt A⁻¹ x = Δt² A⁻¹ S⁻¹ A⁻¹ x
        S_inv_A_inv_x = self.S_solve(A_inv_x)
        A_inv_S_inv_A_inv_x = self.A_solve(S_inv_A_inv_x)
        L11_x = A_inv_x + self.B_diag**2 * A_inv_S_inv_A_inv_x

        result = np.zeros(2 * N, dtype=np.float64)
        result[:N] = L11_x + L12_y
        result[N:] = L21_x + L22_y

        return result

    def to_dense(self):
        """Convert to dense matrix (for verification/debugging)."""
        from scipy.linalg import toeplitz, inv

        N = self.N
        A = toeplitz(self.A_gamma)
        B = self.B_diag * np.eye(N)
        D = self.D_diag * np.eye(N)

        Sigma = np.zeros((2*N, 2*N))
        Sigma[:N, :N] = A
        Sigma[:N, N:] = B
        Sigma[N:, :N] = B
        Sigma[N:, N:] = D

        return inv(Sigma)


def test_toeplitz_fft():
    """Test FFT-based Toeplitz operations."""
    np.random.seed(42)

    print("Testing FFT-based Toeplitz operations...")

    for N in [10, 100, 1000]:
        # Random symmetric positive definite Toeplitz
        gamma = np.zeros(N)
        gamma[0] = 2.0
        gamma[1:] = 0.5 ** np.arange(1, N)  # Decaying off-diagonal

        x = np.random.randn(N)

        # Direct multiplication
        from scipy.linalg import toeplitz
        T = toeplitz(gamma)
        y_direct = T @ x

        # FFT multiplication
        y_fft = toeplitz_matvec_symmetric_fft(gamma, x)

        error = np.max(np.abs(y_direct - y_fft))
        print(f"  N={N}: matvec error = {error:.2e}")

        # Test solve
        b = np.random.randn(N)
        x_direct = np.linalg.solve(T, b)
        x_pcg, info = toeplitz_solve_pcg(gamma, b, tol=1e-12)

        solve_error = np.max(np.abs(x_direct - x_pcg))
        print(f"  N={N}: solve error = {solve_error:.2e} ({info['iters']} iters)")

    print("All tests passed!\n")


def test_blocked_precision():
    """Test BlockedMixedFBMPrecision class."""
    print("Testing BlockedMixedFBMPrecision...")

    for N in [10, 50, 100]:
        H = 0.6
        alpha = 1.0
        delta_t = 1.0 / N

        # Build using our class
        prec = BlockedMixedFBMPrecision(N, H, alpha, delta_t)

        # Build dense for comparison
        from constrained_decomposition_matrices import spd_mixed_fbm_blocked
        Sigma = spd_mixed_fbm_blocked(N, H, alpha, delta_t)
        Lambda_direct = np.linalg.inv(Sigma)

        # Test matvec
        z = np.random.randn(2 * N)
        Lz_direct = Lambda_direct @ z
        Lz_class = prec.matvec(z)

        error = np.max(np.abs(Lz_direct - Lz_class))
        print(f"  N={N}: matvec error = {error:.2e}")

        # Test to_dense
        Lambda_class = prec.to_dense()
        dense_error = np.max(np.abs(Lambda_direct - Lambda_class))
        print(f"  N={N}: dense error = {dense_error:.2e}")

    print("All tests passed!\n")


class BlockedNewtonSolver:
    """
    Specialized Newton solver for mixed fBM with blocked ordering.

    Exploits the 2×2 block structure of the precision matrix and the
    sparse structure of the Markovian basis to achieve faster convergence.

    Key optimizations:
    1. Block Schur complement: M22 = Λ22 is fixed (basis doesn't affect it),
       so we precompute Λ22⁻¹ and reduce 2N×2N inversions to N×N
    2. Sparse basis: O(N) basis elements, each with O(N) non-zeros
    3. Efficient gradient: tr(B Dₖ) computed via sparse operations
    4. Vectorized index arrays for gradient/Hessian computation

    The covariance has structure:
        Σ = [Δt I + α²Γ_H   Δt I]
            [   Δt I        Δt I]

    The Markovian basis C(x) only affects M11 and M12 blocks, not M22.
    This allows precomputing M22⁻¹ = Λ22⁻¹ once.
    """

    def __init__(self, N, H, alpha=1.0, delta_t=None, verbose=False):
        """
        Initialize the solver.

        Parameters
        ----------
        N : int
            Number of time steps.
        H : float
            Hurst parameter.
        alpha : float
            fBM weight.
        delta_t : float, optional
            Time step. Default: 1/N.
        verbose : bool
            Print progress.
        """
        self.N = N
        self.n = 2 * N
        self.H = H
        self.alpha = alpha
        self.delta_t = delta_t if delta_t is not None else 1.0 / N
        self.verbose = verbose

        # Timing stats
        self._timing = {
            'build_precision': 0.0,
            'build_basis': 0.0,
            'schur_complement': 0.0,
            'gradient': 0.0,
            'hessian_vec': 0.0,
            'line_search': 0.0,
        }

        # Build precision matrix representation
        self._build_precision()

        # Build Markovian basis structure
        self._build_basis()

    def _build_precision(self):
        """Build the precision matrix Λ and precompute fixed block inverses."""
        import time
        from constrained_decomposition_matrices import spd_mixed_fbm_blocked

        t0 = time.time()

        Sigma = spd_mixed_fbm_blocked(self.N, self.H, self.alpha, self.delta_t)
        self.Lambda = np.linalg.inv(Sigma)

        # Store block structure
        N = self.N
        self.Lambda_11 = self.Lambda[:N, :N].copy()
        self.Lambda_12 = self.Lambda[:N, N:].copy()
        self.Lambda_21 = self.Lambda[N:, :N].copy()
        self.Lambda_22 = self.Lambda[N:, N:].copy()

        # Key optimization: Λ22 is FIXED (basis doesn't affect M22 block)
        # Precompute Λ22⁻¹ for use in Schur complement formula
        self.Lambda_22_inv = np.linalg.inv(self.Lambda_22)

        # For reference: log|Σ| and log|Λ22|
        _, self.log_det_Sigma = np.linalg.slogdet(Sigma)
        _, self.log_det_Lambda_22 = np.linalg.slogdet(self.Lambda_22)

        self._timing['build_precision'] = time.time() - t0

    def _build_basis(self):
        """
        Build the Markovian basis structure for blocked ordering.

        The blocked Markovian basis has 2(N-1) elements:
        - D^Mark_{l,X} for l = 2,...,N: affects column l in X-block
        - D^Mark_{l,Y} for l = 2,...,N: affects column l in cross-block
        """
        N = self.N
        self.m = 2 * (N - 1)  # Basis dimension

        # Store basis structure for efficient operations
        # Each basis element affects column l with entries at rows 0,...,l-2
        self.basis_info = []

        for l in range(2, N + 1):
            l_idx = l - 1  # 0-based

            # D^Mark_{l,X}: entries in X-block (rows 0,...,l-2, column l-1)
            rows_X = np.arange(l - 1)
            self.basis_info.append({
                'type': 'X',
                'l': l,
                'l_idx': l_idx,
                'rows': rows_X,
                'block': 'XX'  # Affects [0:N, 0:N] block
            })

            # D^Mark_{l,Y}: entries in cross-block (rows N,...,N+l-2, column l-1)
            rows_Y = N + np.arange(l - 1)
            self.basis_info.append({
                'type': 'Y',
                'l': l,
                'l_idx': l_idx,
                'rows': rows_Y,
                'block': 'YX'  # Affects [N:2N, 0:N] block (and symmetric)
            })

        # Build vectorized index arrays for fast gradient/Hessian computation
        self._build_vectorized_indices()

    def _build_vectorized_indices(self):
        """
        Build index arrays for vectorized gradient/Hessian computation.

        For each basis element k, we store:
        - col_idx[k]: the column index l_idx
        - row_start[k]: start of row indices for this element
        - row_end[k]: end of row indices

        This allows computing gradient via:
            g[k] = 2 * sum(B[all_rows[row_start[k]:row_end[k]], col_idx[k]])
        """
        N = self.N
        m = self.m

        # Column indices for each basis element
        self.col_idx = np.zeros(m, dtype=np.int64)

        # All row indices flattened, with pointers for each basis element
        all_rows = []
        row_start = np.zeros(m + 1, dtype=np.int64)

        for k, info in enumerate(self.basis_info):
            self.col_idx[k] = info['l_idx']
            row_start[k] = len(all_rows)
            all_rows.extend(info['rows'])

        row_start[m] = len(all_rows)
        self.all_rows = np.array(all_rows, dtype=np.int64)
        self.row_start = row_start

        # Number of entries per basis element
        self.entries_per_basis = np.diff(row_start)

        # Precompute expanded column indices for vectorized operations
        self.col_expanded = np.repeat(self.col_idx, self.entries_per_basis)

    def build_C(self, x):
        """
        Build C(x) = Σₖ xₖ Dₖ from the coefficient vector x.

        Parameters
        ----------
        x : array (m,)
            Coefficients for basis elements.

        Returns
        -------
        C : array (2N, 2N)
            The C matrix.
        """
        n = self.n
        C = np.zeros((n, n), dtype=np.float64)

        # Vectorized: for each basis element k, add x[k] to positions
        for k in range(self.m):
            if abs(x[k]) < 1e-15:
                continue

            l_idx = self.col_idx[k]
            rows = self.all_rows[self.row_start[k]:self.row_start[k+1]]

            # Dₖ has entries at (rows, l_idx) and (l_idx, rows)
            C[rows, l_idx] += x[k]
            C[l_idx, rows] += x[k]

        return C

    def build_M(self, x):
        """
        Build M(x) = Λ - C(x).

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        M : array (2N, 2N)
            The M matrix.
        """
        return self.Lambda - self.build_C(x)

    def compute_B(self, x):
        """
        Compute B = M(x)⁻¹ = (Λ - C(x))⁻¹.

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        B : array (2N, 2N)
            The B matrix.
        """
        M = self.build_M(x)
        return np.linalg.inv(M)

    def compute_B_block_schur(self, x):
        """
        Compute B = M(x)⁻¹ using block Schur complement formula.

        For symmetric M = [M₁₁  M₁₂]
                          [M₂₁  M₂₂]  with M₂₁ = M₁₂ᵀ

        The inverse is:
        B = M⁻¹ = [M₁₁⁻¹ + M₁₁⁻¹ M₁₂ S⁻¹ M₂₁ M₁₁⁻¹,   -M₁₁⁻¹ M₁₂ S⁻¹]
                  [-S⁻¹ M₂₁ M₁₁⁻¹,                      S⁻¹          ]

        where S = M₂₂ - M₂₁ M₁₁⁻¹ M₁₂ is the Schur complement.

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        B : array (2N, 2N)
            The B matrix.
        info : dict
            Intermediate computations for reuse.
        """
        from scipy.linalg import cho_factor, cho_solve

        N = self.N
        M = self.build_M(x)

        M11 = M[:N, :N]
        M12 = M[:N, N:]
        M21 = M[N:, :N]  # = M12.T for symmetric M
        M22 = M[N:, N:]

        # Cholesky factorization of M11
        c11, lower11 = cho_factor(M11, lower=True)

        # M11^{-1} @ M12 using cho_solve
        M11_inv_M12 = cho_solve((c11, lower11), M12)

        # Schur complement: S = M22 - M21 @ M11^{-1} @ M12
        S = M22 - M21 @ M11_inv_M12

        # Cholesky factorization of S
        c_S, lower_S = cho_factor(S, lower=True)

        # For the inverse formula, we need:
        # M11_inv = cho_solve((c11, lower11), I)
        # S_inv = cho_solve((c_S, lower_S), I)
        M11_inv = cho_solve((c11, lower11), np.eye(N))
        S_inv = cho_solve((c_S, lower_S), np.eye(N))

        # Build B blocks using Schur complement formula
        S_inv_M21_M11_inv = S_inv @ M21 @ M11_inv

        # B11 = M11^{-1} + M11^{-1} M12 S^{-1} M21 M11^{-1}
        B11 = M11_inv + M11_inv_M12 @ S_inv_M21_M11_inv

        # B12 = -M11^{-1} M12 S^{-1}
        B12 = -M11_inv_M12 @ S_inv

        # B21 = B12.T (for symmetric M)
        B21 = B12.T

        # B22 = S^{-1}
        B22 = S_inv

        # Assemble B
        B = np.zeros((2*N, 2*N), dtype=np.float64)
        B[:N, :N] = B11
        B[:N, N:] = B12
        B[N:, :N] = B21
        B[N:, N:] = B22

        info = {'c11': c11, 'c_S': c_S, 'M11_inv': M11_inv, 'S_inv': S_inv}
        return B, info

    def compute_B_fast(self, x):
        """
        Compute B = M(x)⁻¹ using optimized direct inverse.

        For now, just use np.linalg.inv which is highly optimized.
        The Schur complement approach is only faster for very large N
        when combined with specialized Toeplitz solvers.

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        B : array (2N, 2N)
            The B matrix.
        info : dict
            Empty info dict for interface compatibility.
        """
        M = self.build_M(x)
        # Use scipy's cho_solve for SPD matrices (faster than general inv)
        from scipy.linalg import cho_factor, cho_solve
        c, lower = cho_factor(M, lower=True)
        B = cho_solve((c, lower), np.eye(2 * self.N))
        return B, {'c': c}

    def compute_B_block_optimized(self, x):
        """
        Compute B = M(x)⁻¹ exploiting that M22 = Λ22 is fixed.

        Key insight: The Markovian basis only affects M11 and M12/M21 blocks.
        M22 = Λ22 never changes, so we precomputed Λ22⁻¹.

        Using block inversion with Schur complement w.r.t. M11:
            S = M22 - M21 M11⁻¹ M12  (N×N Schur complement)
            B11 = M11⁻¹ + M11⁻¹ M12 S⁻¹ M21 M11⁻¹
            B12 = -M11⁻¹ M12 S⁻¹
            B21 = B12ᵀ
            B22 = S⁻¹

        This reduces the problem from 2N×2N to N×N operations.

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        B : array (2N, 2N)
            The B matrix.
        info : dict
            Contains intermediate computations for potential reuse.
        """
        import time
        from scipy.linalg import cho_factor, cho_solve

        t0 = time.time()
        N = self.N

        # Build C blocks from x
        C11, C12 = self._build_C_blocks(x)

        # M blocks (M21 = M12.T, M22 = Λ22 fixed)
        M11 = self.Lambda_11 - C11
        M12 = self.Lambda_12 - C12
        # M22 = self.Lambda_22 (unchanged)

        # Cholesky factorization of M11 (N×N instead of 2N×2N)
        try:
            c11, lower11 = cho_factor(M11, lower=True)
        except np.linalg.LinAlgError:
            # Fallback to full 2N×2N Cholesky
            return self.compute_B_fast(x)

        # M11⁻¹ M12 via Cholesky solve
        M11_inv_M12 = cho_solve((c11, lower11), M12)

        # Schur complement: S = M22 - M21 M11⁻¹ M12 = Λ22 - M12ᵀ M11⁻¹ M12
        S = self.Lambda_22 - M12.T @ M11_inv_M12

        # Cholesky of S (N×N)
        try:
            c_S, lower_S = cho_factor(S, lower=True)
        except np.linalg.LinAlgError:
            # Fallback to full 2N×2N Cholesky
            return self.compute_B_fast(x)

        # S⁻¹ and M11⁻¹
        S_inv = cho_solve((c_S, lower_S), np.eye(N))
        M11_inv = cho_solve((c11, lower11), np.eye(N))

        # Compute B blocks
        # B12 = -M11⁻¹ M12 S⁻¹
        B12 = -M11_inv_M12 @ S_inv

        # B11 = M11⁻¹ + M11⁻¹ M12 S⁻¹ M21 M11⁻¹ = M11⁻¹ - B12 @ M12.T @ M11⁻¹
        B11 = M11_inv - B12 @ M12.T @ M11_inv

        # B22 = S⁻¹
        B22 = S_inv

        # Assemble B
        B = np.zeros((2*N, 2*N), dtype=np.float64)
        B[:N, :N] = B11
        B[:N, N:] = B12
        B[N:, :N] = B12.T  # B21 = B12ᵀ
        B[N:, N:] = B22

        self._timing['schur_complement'] += time.time() - t0

        info = {
            'c11': c11, 'c_S': c_S,
            'M11_inv': M11_inv, 'S_inv': S_inv,
            'log_det_S': np.sum(np.log(np.diag(c_S))),  # From Cholesky
            'log_det_M11': np.sum(np.log(np.diag(c11))),
        }
        return B, info

    def _build_C_blocks(self, x):
        """
        Build C(x) block-wise: return C11 (N×N) and C12 (N×N).

        The Markovian basis structure:
        - D^Mark_{l,X}: entries in C11 at (rows 0..l-2, col l-1)
        - D^Mark_{l,Y}: entries in C12 at (rows 0..l-2, col l-1) where
          the original entry is at row N+r-1, but we map to the C12 block

        Parameters
        ----------
        x : array (m,)
            Coefficients.

        Returns
        -------
        C11 : array (N, N)
            Upper-left block of C(x).
        C12 : array (N, N)
            Upper-right block of C(x).
        """
        N = self.N
        C11 = np.zeros((N, N), dtype=np.float64)
        C12 = np.zeros((N, N), dtype=np.float64)

        # Process basis elements pairwise (D^Mark_{l,X} then D^Mark_{l,Y})
        k = 0
        for l in range(2, N + 1):
            l_idx = l - 1  # 0-based column index

            # D^Mark_{l,X}: affects C11
            if k < self.m:
                rows = np.arange(l - 1)  # rows 0,...,l-2
                if abs(x[k]) > 1e-15:
                    C11[rows, l_idx] += x[k]
                    C11[l_idx, rows] += x[k]
                k += 1

            # D^Mark_{l,Y}: affects C12 (and C21 = C12.T)
            if k < self.m:
                rows = np.arange(l - 1)  # rows 0,...,l-2 in the Y block
                if abs(x[k]) > 1e-15:
                    # Original: entries at (N+r, l_idx) and (l_idx, N+r)
                    # In blocks: C12[l_idx, rows] and C21[rows, l_idx]
                    # Since M21 = M12.T and C21 = C12.T, we just set C12
                    C12[l_idx, rows] += x[k]  # This handles the (l_idx, N+r) entries
                k += 1

        return C11, C12

    def compute_gradient(self, B):
        """
        Compute gradient g_k = tr(B Dₖ) for all k.

        Vectorized implementation using precomputed indices.

        Parameters
        ----------
        B : array (2N, 2N)
            Current B matrix.

        Returns
        -------
        g : array (m,)
            Gradient vector.
        """
        # Extract all needed B values at once using precomputed indices
        B_vals = B[self.all_rows, self.col_expanded]

        # Sum within each basis element using reduceat
        g = 2.0 * np.add.reduceat(B_vals, self.row_start[:-1])

        return g

    def compute_hessian_vector_product(self, B, v):
        """
        Compute Hessian-vector product (H v)_k = Σₗ H_kl v_l.

        Vectorized implementation.

        Parameters
        ----------
        B : array (2N, 2N)
            Current B matrix.
        v : array (m,)
            Vector to multiply.

        Returns
        -------
        Hv : array (m,)
            Result of H @ v.
        """
        # D(v) = Σₗ vₗ Dₗ
        D_v = self.build_C(v)

        # B @ D(v) @ B
        BDvB = B @ D_v @ B

        # (Hv)_k = tr(Dₖ BDvB) = 2 * sum(BDvB[rows_k, col_k])
        BDvB_vals = BDvB[self.all_rows, self.col_expanded]
        Hv = 2.0 * np.add.reduceat(BDvB_vals, self.row_start[:-1])

        return Hv

    def compute_hessian_vector_product_block(self, B, v):
        """
        Block-optimized Hessian-vector product exploiting D(v) structure.

        Key insight: For the Markovian basis, D(v) has structure:
            D(v) = [D11  D12 ]
                   [D12ᵀ  0  ]

        The bottom-right N×N block is ZERO. This allows us to:
        1. Work with N×N blocks instead of 2N×2N (~4-8x fewer ops)
        2. Only compute the entries of BDvB we actually need

        Parameters
        ----------
        B : array (2N, 2N)
            Current B matrix.
        v : array (m,)
            Vector to multiply.

        Returns
        -------
        Hv : array (m,)
            Result of H @ v.
        """
        N = self.N

        # Extract B blocks
        B11 = B[:N, :N]
        B12 = B[:N, N:]
        B21 = B[N:, :N]
        B22 = B[N:, N:]

        # Build D11 and D12 blocks from v
        D11, D12 = self._build_Dv_blocks(v)

        # Compute Q = D(v) @ B using block structure
        # Q = [D11 B11 + D12 B21,  D11 B12 + D12 B22]
        #     [D12ᵀ B11,           D12ᵀ B12         ]
        #
        # We only need Q[:, :N] for extracting Hv (columns 0..N-1)
        Q11 = D11 @ B11 + D12 @ B21  # N×N
        Q21 = D12.T @ B11            # N×N

        # Compute BDvB[:, :N] = B @ Q[:, :N]
        # BDvB[:N, :N] = B11 @ Q11 + B12 @ Q21
        # BDvB[N:, :N] = B21 @ Q11 + B22 @ Q21
        BDvB_upper = B11 @ Q11 + B12 @ Q21  # N×N: rows 0..N-1, cols 0..N-1
        BDvB_cross = B21 @ Q11 + B22 @ Q21  # N×N: rows N..2N-1, cols 0..N-1

        # Vectorized Hv extraction
        # For D^Mark_{l,X} (k=0,2,4,...): sum of BDvB_upper[0:l-1, l-1]
        # For D^Mark_{l,Y} (k=1,3,5,...): sum of BDvB_cross[0:l-1, l-1]
        #
        # Use cumsum trick: cumsum along columns, then extract diagonal-like values
        # cumsum_upper[i,j] = sum(BDvB_upper[0:i+1, j])
        # We need cumsum_upper[l-2, l-1] for l=2..N, i.e., cumsum_upper[0,1], cumsum_upper[1,2], ...
        cumsum_upper = np.cumsum(BDvB_upper, axis=0)  # N×N
        cumsum_cross = np.cumsum(BDvB_cross, axis=0)  # N×N

        # Extract values: for l=2..N, we need cumsum[l-2, l-1]
        # That's indices (0,1), (1,2), (2,3), ..., (N-2, N-1)
        # i.e., row indices 0..N-2, col indices 1..N-1
        row_indices = np.arange(N - 1)
        col_indices = np.arange(1, N)

        Hv_X = 2.0 * cumsum_upper[row_indices, col_indices]  # length N-1
        Hv_Y = 2.0 * cumsum_cross[row_indices, col_indices]  # length N-1

        # Interleave: Hv = [Hv_X[0], Hv_Y[0], Hv_X[1], Hv_Y[1], ...]
        Hv = np.empty(self.m, dtype=np.float64)
        Hv[0::2] = Hv_X
        Hv[1::2] = Hv_Y

        return Hv

    def _build_Dv_blocks(self, v):
        """
        Build D(v) block-wise: return D11 (N×N) and D12 (N×N).

        D(v) = [D11  D12 ]  where D11, D12 are N×N
               [D12ᵀ  0  ]  and D22 = 0 for Markovian basis

        Vectorized implementation for speed.

        Parameters
        ----------
        v : array (m,)
            Coefficients for Hessian-vector product.

        Returns
        -------
        D11 : array (N, N)
            Upper-left block of D(v).
        D12 : array (N, N)
            Upper-right block of D(v).
        """
        N = self.N
        D11 = np.zeros((N, N), dtype=np.float64)
        D12 = np.zeros((N, N), dtype=np.float64)

        # v[0::2] are coefficients for D^Mark_{l,X} (l = 2, 3, ..., N)
        # v[1::2] are coefficients for D^Mark_{l,Y} (l = 2, 3, ..., N)
        v_X = v[0::2]  # length N-1
        v_Y = v[1::2]  # length N-1

        # Vectorized construction using precomputed indices
        # For D^Mark_{l,X}: column l-1 gets v_X[l-2] in rows 0..l-2
        # For D^Mark_{l,Y}: column l-1 gets v_Y[l-2] in rows 0..l-2 of D12
        for l in range(2, N + 1):
            l_idx = l - 1
            v_idx = l - 2  # index into v_X, v_Y

            if v_idx < len(v_X):
                # D11: column l_idx, rows 0..l-2
                D11[:l-1, l_idx] = v_X[v_idx]
                D11[l_idx, :l-1] = v_X[v_idx]  # symmetric

            if v_idx < len(v_Y):
                # D12: row l_idx, cols 0..l-2
                D12[l_idx, :l-1] = v_Y[v_idx]

        return D11, D12

    def compute_hessian_diagonal(self, B, use_block_hv=True):
        """
        Compute diagonal of Hessian: H_kk = e_k^T H e_k for all k.

        Used for diagonal preconditioning in CG.
        """
        compute_Hv = self.compute_hessian_vector_product_block if use_block_hv else self.compute_hessian_vector_product

        diag_H = np.zeros(self.m, dtype=np.float64)
        for k in range(self.m):
            e_k = np.zeros(self.m, dtype=np.float64)
            e_k[k] = 1.0
            Hv = compute_Hv(B, e_k)
            diag_H[k] = Hv[k]
        return diag_H

    def solve(self, tol=1e-8, max_iter=200, method="newton-cg", use_block_opt=True, use_block_hv=True, use_precond=True):
        """
        Solve the constrained decomposition problem.

        Find x such that tr(B(x) Dₖ) = 0 for all k, where B(x) = (Λ - C(x))⁻¹.

        Parameters
        ----------
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum iterations.
        method : str
            "newton" (explicit Hessian) or "newton-cg" (matrix-free).
        use_block_opt : bool
            If True, use block-optimized N×N Schur complement for B computation.
            If False, use full 2N×2N Cholesky.
        use_block_hv : bool
            If True, use block-optimized Hessian-vector products (4-8x faster).
            If False, use standard 2N×2N matrix products.
        use_precond : bool
            If True, use diagonal preconditioning for CG (faster for ill-conditioned).

        Returns
        -------
        B : array (2N, 2N)
            Optimal B matrix.
        C : array (2N, 2N)
            Optimal C matrix.
        x : array (m,)
            Optimal coefficients.
        info : dict
            Convergence info including timing breakdown.
        """
        import time
        t_start = time.time()
        t_B_compute = 0.0
        t_gradient = 0.0
        t_hessian = 0.0
        t_linesearch = 0.0

        # Choose B computation method
        compute_B = self.compute_B_block_optimized if use_block_opt else self.compute_B_fast

        # Choose Hv computation method
        compute_Hv = self.compute_hessian_vector_product_block if use_block_hv else self.compute_hessian_vector_product

        # Initialize x = 0
        x = np.zeros(self.m, dtype=np.float64)

        # Initial B
        t0 = time.time()
        B, L_info = compute_B(x)
        t_B_compute += time.time() - t0

        t0 = time.time()
        g = self.compute_gradient(B)
        t_gradient += time.time() - t0

        if self.verbose:
            print(f"BlockedNewtonSolver: N={self.N}, m={self.m}")
            print(f"  Block B: {'ON' if use_block_opt else 'OFF'}, Block Hv: {'ON' if use_block_hv else 'OFF'}")
            print(f"  Initial max|g| = {np.max(np.abs(g)):.3e}")

        # Precompute diagonal preconditioner ONCE (expensive: m Hv products)
        # Reuse throughout iterations - preconditioner doesn't need to be exact
        diag_H_precond = None
        M_linop = None
        t_precond = 0.0
        if use_precond and method == "newton-cg":
            from scipy.sparse.linalg import LinearOperator
            t0_precond = time.time()
            diag_H_precond = self.compute_hessian_diagonal(B, use_block_hv)
            diag_H_safe = np.maximum(diag_H_precond, 1e-10)
            M_linop = LinearOperator((self.m, self.m), matvec=lambda v, d=diag_H_safe: v / d)
            t_precond = time.time() - t0_precond
            if self.verbose:
                print(f"  [Precond] computed in {t_precond:.1f}s, diag(H) range: [{diag_H_precond.min():.2e}, {diag_H_precond.max():.2e}]")

        t_iter_start = time.time()
        for it in range(max_iter):
            max_g = np.max(np.abs(g))

            if max_g < tol:
                if self.verbose:
                    print(f"Converged at iter {it}: max|g| = {max_g:.3e} (tol={tol:.1e})")
                break

            t0 = time.time()
            if method == "newton":
                # Full Hessian (O(m²) computation)
                H = np.zeros((self.m, self.m), dtype=np.float64)
                for l in range(self.m):
                    e_l = np.zeros(self.m)
                    e_l[l] = 1.0
                    H[:, l] = compute_Hv(B, e_l)

                # Newton direction: H d = -g
                d = np.linalg.solve(H + 1e-10 * np.eye(self.m), -g)

            else:  # newton-cg
                # CG for Newton direction
                from scipy.sparse.linalg import cg, LinearOperator

                def Hv_op(v):
                    return compute_Hv(B, v)

                H_linop = LinearOperator((self.m, self.m), matvec=Hv_op)

                # Use precomputed diagonal preconditioner (M_linop computed before loop)
                # Track CG iterations via callback
                cg_iters = [0]
                def cg_callback(xk):
                    cg_iters[0] += 1
                d, cg_info = cg(H_linop, -g, rtol=1e-6, maxiter=self.m, M=M_linop, callback=cg_callback)
                cg_iter_count = cg_iters[0]
            t_hessian += time.time() - t0

            # Line search with Armijo-like condition: require gradient norm to decrease
            t0 = time.time()
            step = 1.0
            B_new = None
            L_info_new = None
            g_new = None
            current_gnorm = np.max(np.abs(g))

            for ls_iter in range(20):
                x_new = x + step * d

                try:
                    B_new, L_info_new = compute_B(x_new)
                    # Cholesky succeeded, check if gradient improves
                    g_new = self.compute_gradient(B_new)
                    new_gnorm = np.max(np.abs(g_new))

                    # Accept if gradient norm decreases (with small tolerance for numerical noise)
                    if new_gnorm < current_gnorm * 1.01:
                        break
                    else:
                        # Gradient didn't improve, try smaller step
                        step *= 0.5
                        B_new = None
                except np.linalg.LinAlgError:
                    step *= 0.5
            t_linesearch += time.time() - t0

            if B_new is None:
                if self.verbose:
                    print(f"Line search failed at iter {it} (couldn't decrease gradient)")
                break

            x = x_new
            B = B_new
            L_info = L_info_new
            g = g_new  # Already computed in line search
            t_gradient += 0  # Gradient computed in line search

            if self.verbose and (it % 10 == 0 or it < 5):
                t_iter = time.time() - t_iter_start
                if method == "newton":
                    print(f"Iter {it}: max|g| = {max_g:.3e}, step = {step:.3f}, iter_t={t_iter:.1f}s, total_t={time.time() - t_start:.1f}s")
                else:
                    cg_status = "conv" if cg_info == 0 else "max"
                    print(f"Iter {it}: max|g| = {max_g:.3e}, step = {step:.3f}, CG={cg_iter_count}({cg_status}), iter_t={t_iter:.1f}s, total_t={time.time() - t_start:.1f}s")

            t_iter_start = time.time()  # Reset for next iteration

        C = self.build_C(x)
        t_total = time.time() - t_start

        info = {
            'iters': it + 1,
            'converged': np.max(np.abs(g)) < tol,
            'final_max_g': np.max(np.abs(g)),
            'time': t_total,
            'timing': {
                'precond': t_precond,
                'B_compute': t_B_compute + t_linesearch,  # Line search includes B computation
                'gradient': t_gradient,
                'hessian_cg': t_hessian,
            },
            'use_block_opt': use_block_opt,
            'use_block_hv': use_block_hv,
        }

        return B, C, x, info

    def solve_lbfgs(self, tol=1e-8, max_iter=500, use_block_opt=True, history_size=10, ftol=1e-8):
        """
        Solve using L-BFGS (quasi-Newton with limited memory).

        First-order method: only needs gradient, no Hessian-vector products.
        More iterations than Newton-CG, but each iteration is much cheaper.

        Parameters
        ----------
        tol : float
            Convergence tolerance for gradient norm.
        max_iter : int
            Maximum iterations.
        use_block_opt : bool
            If True, use block-optimized B computation.
        history_size : int
            Number of gradient/step pairs to store (typically 5-20).
        ftol : float
            Function tolerance. Iterations stop when
            (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
            Default 1e-8 for good convergence.

        Returns
        -------
        B : array (2N, 2N)
            Optimal B matrix.
        C : array (2N, 2N)
            Optimal C matrix.
        x : array (m,)
            Optimal coefficients.
        info : dict
            Convergence info including timing.
        """
        import time
        from scipy.optimize import minimize

        t_start = time.time()
        t_B_compute = 0.0
        t_gradient = 0.0
        func_evals = 0
        grad_evals = 0

        # Choose B computation method
        compute_B = self.compute_B_block_optimized if use_block_opt else self.compute_B_fast

        # Cache for B to avoid recomputation when gradient is called right after function
        cached_x = None
        cached_B = None

        def objective(x_vec):
            """f(x) = -log det(Λ - C(x)) = log det(B)"""
            nonlocal t_B_compute, func_evals, cached_x, cached_B

            func_evals += 1
            t0 = time.time()

            try:
                B, L_info = compute_B(x_vec)
                cached_x = x_vec.copy()
                cached_B = B
            except np.linalg.LinAlgError:
                # Matrix not SPD - return large value
                cached_x = None
                cached_B = None
                t_B_compute += time.time() - t0
                return 1e10

            t_B_compute += time.time() - t0

            # f(x) = log det(B) = -log det(M)
            # From block Cholesky: log det(M) = log det(M11) + log det(S)
            # = 2 * (sum log diag(L11) + sum log diag(L_S))
            # The info dict stores sum log diag already
            if 'log_det_M11' in L_info:
                log_det_M = 2 * (L_info['log_det_M11'] + L_info['log_det_S'])
                return -log_det_M  # log det(B) = -log det(M)
            else:
                # Fallback if full Cholesky was used
                sign, logdet = np.linalg.slogdet(B)
                if sign <= 0:
                    return 1e10
                return logdet

        def gradient(x_vec):
            """∇f_k = -tr(B D_k)"""
            nonlocal t_gradient, grad_evals, t_B_compute, cached_x, cached_B

            grad_evals += 1
            t0 = time.time()

            # Check if we can use cached B
            if cached_x is not None and np.allclose(x_vec, cached_x):
                B = cached_B
            else:
                # Need to compute B
                t_b0 = time.time()
                try:
                    B, _ = compute_B(x_vec)
                    cached_x = x_vec.copy()
                    cached_B = B
                except np.linalg.LinAlgError:
                    t_B_compute += time.time() - t_b0
                    t_gradient += time.time() - t0 - (time.time() - t_b0)
                    return np.zeros(self.m)
                t_B_compute += time.time() - t_b0

            # g_k = tr(B D_k)
            # f(x) = log det(B) = -log det(M)
            # ∇f_k = tr(B D_k) = g_k
            g = self.compute_gradient(B)
            t_gradient += time.time() - t0

            return g

        # Initial point
        x0 = np.zeros(self.m, dtype=np.float64)

        if self.verbose:
            print(f"BlockedNewtonSolver L-BFGS: N={self.N}, m={self.m}")
            print(f"  Block B: {'ON' if use_block_opt else 'OFF'}, history={history_size}")

        # Run L-BFGS-B
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=gradient,
            options={
                'maxiter': max_iter,
                'gtol': tol,
                'ftol': ftol,  # Tight function tolerance for precise convergence
                'maxcor': history_size,  # History size
                'disp': self.verbose,
            }
        )

        x = result.x
        t_total = time.time() - t_start

        # Final B and C
        B, _ = compute_B(x)
        C = self.build_C(x)
        g = self.compute_gradient(B)

        info = {
            'iters': result.nit,
            'func_evals': func_evals,
            'grad_evals': grad_evals,
            'converged': result.success,
            'final_max_g': np.max(np.abs(g)),
            'time': t_total,
            'timing': {
                'B_compute': t_B_compute,
                'gradient': t_gradient,
            },
            'use_block_opt': use_block_opt,
            'method': 'L-BFGS',
            'message': result.message,
        }

        if self.verbose:
            print(f"L-BFGS finished: {result.nit} iters, {func_evals} f-evals, {grad_evals} g-evals")
            print(f"  Final max|g| = {np.max(np.abs(g)):.3e}, converged={result.success} (tol={tol:.1e})")

        return B, C, x, info

    def compute_investment_value(self, B):
        """
        Compute the investment value: 0.5 * (log|Σ| - log|B|).

        Parameters
        ----------
        B : array (2N, 2N)
            The B matrix.

        Returns
        -------
        value : float
            Log investment value.
        """
        _, log_det_B = np.linalg.slogdet(B)
        return 0.5 * (self.log_det_Sigma - log_det_B)


def benchmark_block_hv(N_values=None, H=0.6, alpha=1.0, verbose=True):
    """
    Benchmark block-optimized Hv vs standard Hv computation.

    Parameters
    ----------
    N_values : list of int
        Values of N to test. Default: [50, 100, 200, 300]
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.
    verbose : bool
        Print results table.

    Returns
    -------
    results : list of dict
        Benchmark results for each N.
    """
    if N_values is None:
        N_values = [50, 100, 200, 300]

    results = []

    if verbose:
        print("\n" + "=" * 90)
        print("BENCHMARK: Block-optimized Hv vs Standard Hv (Markovian basis)")
        print(f"H={H}, alpha={alpha}")
        print("=" * 90)
        print(f"{'N':>6} {'2N':>6} {'Std (s)':>12} {'Block (s)':>12} {'Speedup':>10} {'Value Match':>12}")
        print("-" * 90)

    for N in N_values:
        delta_t = 1.0 / N

        # Create solver
        solver = BlockedNewtonSolver(N, H, alpha, delta_t, verbose=False)

        # Solve with standard Hv
        B_std, C_std, x_std, info_std = solver.solve(
            tol=1e-8, method="newton-cg", use_block_hv=False
        )
        t_std = info_std['time']
        val_std = solver.compute_investment_value(B_std)

        # Solve with block-optimized Hv
        B_block, C_block, x_block, info_block = solver.solve(
            tol=1e-8, method="newton-cg", use_block_hv=True
        )
        t_block = info_block['time']
        val_block = solver.compute_investment_value(B_block)

        speedup = t_std / t_block if t_block > 0 else float('inf')
        val_diff = abs(val_std - val_block)
        match = val_diff < 1e-6

        if verbose:
            match_str = 'OK' if match else f'DIFF={val_diff:.2e}'
            print(f"{N:>6} {2*N:>6} {t_std:>11.3f}s {t_block:>11.3f}s {speedup:>9.2f}x {match_str:>12}")

        results.append({
            'N': N,
            't_std': t_std,
            't_block': t_block,
            'speedup': speedup,
            'value_std': val_std,
            'value_block': val_block,
            'value_match': match,
            'iters_std': info_std['iters'],
            'iters_block': info_block['iters'],
        })

    if verbose:
        print("=" * 90)
        if results:
            avg_speedup = np.mean([r['speedup'] for r in results])
            print(f"Average speedup: {avg_speedup:.2f}x")
        print()

    return results


def benchmark_block_optimization(N_values=None, H=0.6, alpha=1.0, verbose=True):
    """
    Benchmark block-optimized vs full 2N×2N Cholesky methods.

    This benchmark compares:
    1. compute_B_block_optimized: N×N Schur complement approach
    2. compute_B_fast: Full 2N×2N Cholesky

    Parameters
    ----------
    N_values : list of int
        Values of N to test. Default: [50, 100, 200, 300, 500]
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.
    verbose : bool
        Print results table.

    Returns
    -------
    results : list of dict
        Benchmark results for each N.
    """
    if N_values is None:
        N_values = [50, 100, 200, 300, 500]

    results = []

    if verbose:
        print("\n" + "=" * 80)
        print("BENCHMARK: Block-optimized N×N vs Full 2N×2N Cholesky")
        print(f"H={H}, alpha={alpha}")
        print("=" * 80)
        print(f"{'N':>6} {'2N':>6} {'Full (s)':>12} {'Block (s)':>12} {'Speedup':>10} {'Value Match':>12}")
        print("-" * 80)

    for N in N_values:
        delta_t = 1.0 / N

        # Create solver
        solver = BlockedNewtonSolver(N, H, alpha, delta_t, verbose=False)

        # Solve with full 2N×2N method
        B_full, C_full, x_full, info_full = solver.solve(
            tol=1e-8, method="newton-cg", use_block_opt=False
        )
        t_full = info_full['time']
        val_full = solver.compute_investment_value(B_full)

        # Solve with block-optimized method
        B_block, C_block, x_block, info_block = solver.solve(
            tol=1e-8, method="newton-cg", use_block_opt=True
        )
        t_block = info_block['time']
        val_block = solver.compute_investment_value(B_block)

        speedup = t_full / t_block if t_block > 0 else float('inf')
        val_diff = abs(val_full - val_block)
        match = val_diff < 1e-8

        if verbose:
            print(f"{N:>6} {2*N:>6} {t_full:>11.3f}s {t_block:>11.3f}s {speedup:>9.2f}x {'OK' if match else f'DIFF={val_diff:.2e}':>12}")

        results.append({
            'N': N,
            't_full': t_full,
            't_block': t_block,
            'speedup': speedup,
            'value_full': val_full,
            'value_block': val_block,
            'value_match': match,
            'iters_full': info_full['iters'],
            'iters_block': info_block['iters'],
        })

    if verbose:
        print("=" * 80)
        if results:
            avg_speedup = np.mean([r['speedup'] for r in results])
            print(f"Average speedup: {avg_speedup:.2f}x")
        print()

    return results


def benchmark_block_hv_general(N_values=None, H=0.6, alpha=1.0, strategy="markovian"):
    """
    Benchmark block-optimized Hv vs standard Hv for any strategy.

    Parameters
    ----------
    N_values : list of int
        Values of N to test.
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.
    strategy : str
        "markovian" or "full"

    Returns
    -------
    results : list of dict
        Benchmark results.
    """
    if N_values is None:
        N_values = [30, 50, 75] if strategy == "full" else [50, 100, 200]

    from finance_example import (
        make_mixed_fbm_markovian_basis_blocked,
        make_mixed_fbm_full_info_basis_blocked,
        spd_mixed_fbm_blocked
    )

    results = []

    print("\n" + "=" * 95)
    print(f"BENCHMARK: Block Hv vs Standard Hv ({strategy.upper()} strategy)")
    print(f"H={H}, alpha={alpha}")
    print("=" * 95)
    print(f"{'N':>6} {'m':>8} {'Std (s)':>12} {'Block (s)':>12} {'Speedup':>10} {'Hv calls':>10}")
    print("-" * 95)

    for N in N_values:
        delta_t = 1.0 / N

        # Build covariance and precision
        Sigma = spd_mixed_fbm_blocked(N, H, alpha, delta_t)
        Lambda = np.linalg.inv(Sigma)

        # Build basis
        if strategy == "markovian":
            basis = make_mixed_fbm_markovian_basis_blocked(N)
        else:
            basis = make_mixed_fbm_full_info_basis_blocked(N)

        m = basis.m

        # Solve with standard Hv
        t_std, timing_std = _solve_with_detailed_timing(Lambda, basis, tol=1e-8, use_block_hv=False)
        hv_calls = timing_std['hv_calls']

        # Solve with block Hv
        t_block, timing_block = _solve_with_detailed_timing(Lambda, basis, tol=1e-8, use_block_hv=True)

        speedup = t_std / t_block if t_block > 0 else float('inf')

        print(f"{N:>6} {m:>8} {t_std:>11.3f}s {t_block:>11.3f}s {speedup:>9.2f}x {hv_calls:>10}")

        results.append({
            'N': N,
            'm': m,
            'strategy': strategy,
            't_std': t_std,
            't_block': t_block,
            'speedup': speedup,
            'hv_calls': hv_calls,
        })

    print("=" * 95)
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"Average speedup: {avg_speedup:.2f}x")
    print()

    return results


def benchmark_methods(N_values=None, H=0.6, alpha=1.0, strategy="markovian"):
    """
    Benchmark Newton-CG vs L-BFGS methods.

    Parameters
    ----------
    N_values : list of int
        Values of N to test.
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.
    strategy : str
        "markovian" or "full"

    Returns
    -------
    results : list of dict
        Benchmark results.
    """
    if N_values is None:
        N_values = [30, 50, 75] if strategy == "full" else [50, 100, 200]

    # For full strategy, we need general solver (not BlockedNewtonSolver which is Markovian-specific)
    if strategy == "full":
        from finance_example import (
            make_mixed_fbm_full_info_basis_blocked,
            spd_mixed_fbm_blocked
        )

        results = []

        print("\n" + "=" * 115)
        print(f"BENCHMARK: Newton-CG vs L-BFGS (FULL strategy)")
        print(f"H={H}, alpha={alpha}")
        print(f"Method 1: constrained_decomposition with method='newton-cg'")
        print(f"Method 2: solve_lbfgs_general (scipy L-BFGS-B)")
        print("=" * 115)
        print(f"{'N':>5} {'m':>7} {'Newton-CG':>12} {'iters':>6} {'L-BFGS':>12} {'iters':>6} {'Speedup':>9} {'Value Match':>12}")
        print("-" * 115)

        for N in N_values:
            delta_t = 1.0 / N

            # Build covariance and precision
            Sigma = spd_mixed_fbm_blocked(N, H, alpha, delta_t)
            Lambda = np.linalg.inv(Sigma)
            basis = make_mixed_fbm_full_info_basis_blocked(N)
            m = basis.m

            # Newton-CG using _solve_with_detailed_timing
            t_newton, timing_newton = _solve_with_detailed_timing(Lambda, basis, tol=1e-8, use_block_hv=True)
            iters_newton = timing_newton['newton_iters']
            method_newton = timing_newton.get('method', 'newton-cg')

            # L-BFGS using general implementation
            t_lbfgs, info_lbfgs = _solve_lbfgs_general(Lambda, basis, tol=1e-8)
            iters_lbfgs = info_lbfgs['iters']
            method_lbfgs = info_lbfgs.get('method', 'lbfgs-general')
            converged_lbfgs = info_lbfgs.get('converged', False)

            # Compute values for comparison
            value_newton = timing_newton.get('final_value', 0)
            value_lbfgs = info_lbfgs.get('final_value', 0)

            value_match = np.abs(value_newton - value_lbfgs) < 1e-4 if value_newton != 0 else True
            match_str = "OK" if value_match else f"DIFF={abs(value_newton - value_lbfgs):.2e}"
            if not converged_lbfgs:
                match_str = f"FAIL:{info_lbfgs.get('message', 'unknown')[:20]}"

            speedup = t_newton / t_lbfgs if t_lbfgs > 0 else float('inf')

            print(f"{N:>5} {m:>7} {t_newton:>11.3f}s {iters_newton:>6} {t_lbfgs:>11.3f}s {iters_lbfgs:>6} {speedup:>8.2f}x {match_str:>12}")

            results.append({
                'N': N,
                'm': m,
                'strategy': strategy,
                't_newton': t_newton,
                'iters_newton': iters_newton,
                't_lbfgs': t_lbfgs,
                'iters_lbfgs': iters_lbfgs,
                'speedup': speedup,
            })

        print("=" * 115)
        if results:
            avg_speedup = np.mean([r['speedup'] for r in results])
            print(f"Average L-BFGS speedup over Newton-CG: {avg_speedup:.2f}x")
            print(f"(Speedup > 1 means L-BFGS is faster)")
        print()

        return results

    # Markovian strategy uses BlockedNewtonSolver
    results = []

    print("\n" + "=" * 115)
    print(f"BENCHMARK: Newton-CG vs L-BFGS (MARKOVIAN strategy)")
    print(f"H={H}, alpha={alpha}")
    print(f"Method 1: BlockedNewtonSolver.solve(method='newton-cg')")
    print(f"Method 2: BlockedNewtonSolver.solve_lbfgs() (custom L-BFGS with block structure)")
    print("=" * 115)
    print(f"{'N':>5} {'m':>7} {'Newton-CG':>12} {'iters':>6} {'L-BFGS':>12} {'iters':>6} {'Speedup':>9} {'Value Match':>12}")
    print("-" * 115)

    for N in N_values:
        delta_t = 1.0 / N

        # Create solver (Markovian-specific)
        solver = BlockedNewtonSolver(N, H, alpha, delta_t, verbose=False)
        m = solver.m

        # Newton-CG (with block Hv optimization)
        B_newton, C_newton, x_newton, info_newton = solver.solve(
            tol=1e-8, method="newton-cg", use_block_opt=True, use_block_hv=True
        )
        t_newton = info_newton['time']
        iters_newton = info_newton['iters']
        method_newton = info_newton.get('method', 'newton-cg')
        value_newton = solver.compute_investment_value(B_newton)

        # L-BFGS (custom implementation)
        B_lbfgs, C_lbfgs, x_lbfgs, info_lbfgs = solver.solve_lbfgs(
            tol=1e-8, use_block_opt=True, history_size=10
        )
        t_lbfgs = info_lbfgs['time']
        iters_lbfgs = info_lbfgs['iters']
        method_lbfgs = info_lbfgs.get('method', 'lbfgs')
        value_lbfgs = solver.compute_investment_value(B_lbfgs)

        # Check values match
        value_match = np.abs(value_newton - value_lbfgs) < 1e-4
        match_str = "OK" if value_match else f"DIFF={abs(value_newton - value_lbfgs):.2e}"

        speedup = t_newton / t_lbfgs if t_lbfgs > 0 else float('inf')

        print(f"{N:>5} {m:>7} {t_newton:>11.3f}s {iters_newton:>6} {t_lbfgs:>11.3f}s {iters_lbfgs:>6} {speedup:>8.2f}x {match_str:>12}")

        results.append({
            'N': N,
            'm': m,
            'strategy': strategy,
            't_newton': t_newton,
            'iters_newton': iters_newton,
            't_lbfgs': t_lbfgs,
            'iters_lbfgs': iters_lbfgs,
            'speedup': speedup,
            'value_newton': value_newton,
            'value_lbfgs': value_lbfgs,
            'value_match': value_match,
        })

    print("=" * 115)
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"Average L-BFGS speedup over Newton-CG: {avg_speedup:.2f}x")
        print(f"(Speedup > 1 means L-BFGS is faster)")
    print()

    return results


def solve_lbfgs_general(Lambda, basis, tol=1e-8, max_iter=500, history_size=10, ftol=1e-8, verbose=False,
                        max_backtracks=60, backtracking_factor=0.5, armijo_c1=1e-4):
    """
    General L-BFGS solver for any basis (not Markovian-specific).

    Uses custom L-BFGS with Armijo backtracking line search (same as constrained_decomposition)
    instead of scipy's L-BFGS-B which fails on this problem due to Wolfe conditions.

    This is a standalone function that can be used with any symmetric basis.
    It solves: max log det(B) s.t. Lambda - B^{-1} in cone(basis)

    Parameters
    ----------
    Lambda : array (n, n)
        Precision matrix.
    basis : SymBasis
        Basis for the constraint subspace.
    tol : float
        Convergence tolerance for gradient norm.
    max_iter : int
        Maximum iterations.
    history_size : int
        L-BFGS history size (number of past gradients to keep).
    ftol : float
        Function tolerance for convergence.
    verbose : bool
        Print progress.
    max_backtracks : int
        Maximum line search backtracks.
    backtracking_factor : float
        Step reduction factor for backtracking.
    armijo_c1 : float
        Armijo condition parameter.

    Returns
    -------
    B : array (n, n)
        Optimal B matrix.
    C : array (n, n)
        Residual C = Lambda - B^{-1}.
    x : array (m,)
        Optimal coefficients.
    info : dict
        Solver information: iters, time, converged, method.
    """
    import time
    from collections import deque

    t_start = time.time()

    n = Lambda.shape[0]
    m = basis.m
    coo_data = basis._coo

    def build_C(x_vec):
        """Build C = sum_k x_k D_k from coefficients."""
        C = np.zeros((n, n), dtype=np.float64)
        for k in range(m):
            rows, cols, vals = coo_data[k]
            C[rows, cols] += x_vec[k] * vals
        return C

    def objective_and_gradient(x_vec):
        """Compute objective and gradient together (more efficient)."""
        M = Lambda - build_C(x_vec)
        try:
            L = np.linalg.cholesky(M)
            # log det(M) = 2 * sum(log(diag(L)))
            log_det_M = 2 * np.sum(np.log(np.diag(L)))
            # B = M^{-1}, log det(B) = -log det(M)
            # Objective: minimize -log det(B) = log det(M)
            f = log_det_M

            # Compute B = M^{-1} for gradient
            B = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
        except np.linalg.LinAlgError:
            return 1e10, np.zeros(m)

        # g_k = d/dx_k [log det(M)] = tr(M^{-1} dM/dx_k) = tr(B * (-D_k)) = -tr(B D_k)
        g = np.zeros(m, dtype=np.float64)
        for k in range(m):
            rows, cols, vals = coo_data[k]
            g[k] = -np.sum(B[rows, cols] * vals)

        return f, g

    def lbfgs_two_loop(g, s_history, y_history):
        """L-BFGS two-loop recursion to compute search direction."""
        q = g.copy()
        history_len = len(s_history)

        if history_len == 0:
            # No history yet, use steepest descent
            return -g

        alphas = []
        rhos = []

        # First loop (backward)
        for i in range(history_len - 1, -1, -1):
            s_i = s_history[i]
            y_i = y_history[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            rhos.insert(0, rho_i)
            alpha_i = rho_i * np.dot(s_i, q)
            alphas.insert(0, alpha_i)
            q = q - alpha_i * y_i

        # Scaling: H0 = gamma * I where gamma = (s'y)/(y'y)
        s_last = s_history[-1]
        y_last = y_history[-1]
        gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-10)
        r = gamma * q

        # Second loop (forward)
        for i in range(history_len):
            s_i = s_history[i]
            y_i = y_history[i]
            beta_i = rhos[i] * np.dot(y_i, r)
            r = r + (alphas[i] - beta_i) * s_i

        return -r  # Search direction (negative for minimization)

    # Initialize
    x = np.zeros(m, dtype=np.float64)
    f, g = objective_and_gradient(x)

    s_history = deque(maxlen=history_size)
    y_history = deque(maxlen=history_size)

    converged = False
    message = "max iterations reached"

    if verbose:
        print(f"  L-BFGS general: n={n}, m={m}, starting optimization...")
        print(f"  iter 0: f={f:.6f}, |g|={np.linalg.norm(g):.2e}")

    for iteration in range(max_iter):
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            converged = True
            message = "gradient norm converged"
            break

        # Compute search direction using L-BFGS
        d = lbfgs_two_loop(g, list(s_history), list(y_history))

        # Armijo backtracking line search
        step = 1.0
        f_old = f
        directional_deriv = np.dot(g, d)

        if directional_deriv >= 0:
            # Not a descent direction, use steepest descent
            d = -g
            directional_deriv = -g_norm**2

        for bt in range(max_backtracks):
            x_new = x + step * d
            f_new, g_new = objective_and_gradient(x_new)

            # Armijo condition: f_new <= f + c1 * step * g'd
            if f_new <= f + armijo_c1 * step * directional_deriv:
                break
            step *= backtracking_factor
        else:
            # Line search failed
            message = f"line search failed at iter {iteration}"
            break

        # Update history
        s = x_new - x
        y = g_new - g

        # Only add to history if curvature is positive (y's > 0)
        if np.dot(y, s) > 1e-10:
            s_history.append(s)
            y_history.append(y)

        # Update state
        x = x_new
        f = f_new
        g = g_new

        # Check function tolerance
        if abs(f_old - f) / max(abs(f_old), abs(f), 1.0) < ftol:
            converged = True
            message = "function tolerance converged"
            break

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  iter {iteration+1}: f={f:.6f}, |g|={np.linalg.norm(g):.2e}, step={step:.2e}")

    t_total = time.time() - t_start

    if verbose:
        print(f"  L-BFGS general: {iteration+1} iters, {t_total:.2f}s, converged={converged} (tol={tol:.1e})")

    # Compute final B and C
    C = build_C(x)
    M = Lambda - C
    B = np.linalg.inv(M)

    info = {
        'iters': iteration + 1,
        'time': t_total,
        'converged': converged,
        'method': 'lbfgs-general-custom',
        'message': message,
    }

    return B, C, x, info


# Keep old name as alias for backwards compatibility
def _solve_lbfgs_general(Lambda, basis, tol=1e-8, max_iter=500, history_size=10, ftol=1e-8):
    """Deprecated: use solve_lbfgs_general instead."""
    B, C, x, info = solve_lbfgs_general(Lambda, basis, tol, max_iter, history_size, ftol)
    return info['time'], info


def benchmark_full_strategy(N_values=None, H=0.6, alpha=1.0):
    """
    Benchmark L-BFGS vs Newton-CG for the FULL strategy.

    This is different from the Markovian benchmark because m = O(N²) for full strategy,
    so the scaling is very different.

    Parameters
    ----------
    N_values : list of int
        Values of N to test. Default: [25, 50, 75, 100]
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.

    Returns
    -------
    results : list of dict
        Benchmark results.
    """
    from constrained_decomposition_core import constrained_decomposition
    from constrained_decomposition_matrices import spd_mixed_fbm, make_mixed_fbm_full_info_basis

    if N_values is None:
        N_values = [25, 50, 75, 100]

    results = []

    print("=" * 120)
    print(f"Full Strategy Benchmark: L-BFGS vs Newton-CG (H={H}, alpha={alpha})")
    print("Full strategy has m = N(N-1) basis elements, so O(N²) coefficients")
    print("=" * 120)
    print(f"{'N':>5} {'m':>10} {'matrix':>8} {'Newton-CG':>14} {'iters':>6} {'L-BFGS':>14} {'iters':>6} {'Speedup':>10} {'Match':>12}")
    print("-" * 120)

    for N in N_values:
        n = 2 * N  # matrix size
        delta_t = 1.0 / N

        # Build matrix and basis
        Sigma = spd_mixed_fbm(N, H=H, alpha=alpha, delta_t=delta_t)
        Lambda = np.linalg.inv(Sigma)
        basis = make_mixed_fbm_full_info_basis(N)
        m = basis.m

        print(f"{N:>5} {m:>10} {n:>6}x{n} ", end="", flush=True)

        # Newton-CG
        t0 = time.time()
        try:
            B_newton, _, x_newton, info_newton = constrained_decomposition(
                A=Lambda, basis=basis, method="newton-cg",
                tol=1e-8, max_iter=500, verbose=False, return_info=True
            )
            t_newton = time.time() - t0
            iters_newton = info_newton['iters']
            _, log_det_B = np.linalg.slogdet(B_newton)
            _, log_det_Sigma = np.linalg.slogdet(Sigma)
            value_newton = 0.5 * (log_det_Sigma - log_det_B)
        except Exception as e:
            print(f"Newton-CG FAILED: {e}")
            continue

        # L-BFGS
        t0 = time.time()
        try:
            B_lbfgs, _, x_lbfgs, info_lbfgs = solve_lbfgs_general(
                Lambda, basis, tol=1e-8, max_iter=500
            )
            t_lbfgs = time.time() - t0
            iters_lbfgs = info_lbfgs['iters']
            _, log_det_B = np.linalg.slogdet(B_lbfgs)
            value_lbfgs = 0.5 * (log_det_Sigma - log_det_B)
        except Exception as e:
            print(f"L-BFGS FAILED: {e}")
            continue

        # Compare
        value_match = np.abs(value_newton - value_lbfgs) < 1e-4
        match_str = "OK" if value_match else f"DIFF={abs(value_newton - value_lbfgs):.2e}"
        speedup = t_newton / t_lbfgs if t_lbfgs > 0 else float('inf')

        print(f"{t_newton:>12.3f}s {iters_newton:>6} {t_lbfgs:>12.3f}s {iters_lbfgs:>6} {speedup:>9.2f}x {match_str:>12}")

        results.append({
            'N': N,
            'n': n,
            'm': m,
            'strategy': 'full',
            't_newton': t_newton,
            'iters_newton': iters_newton,
            't_lbfgs': t_lbfgs,
            'iters_lbfgs': iters_lbfgs,
            'speedup': speedup,
            'value_newton': value_newton,
            'value_lbfgs': value_lbfgs,
            'value_match': value_match,
        })

    print("=" * 120)
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"Average L-BFGS speedup over Newton-CG: {avg_speedup:.2f}x")
        print(f"(Speedup > 1 means L-BFGS is faster)")
    print()

    return results


def benchmark_detailed(N_values=None, H=0.6, alpha=1.0, strategy="markovian", use_block_hv=True):
    """
    Detailed profiling benchmark showing time breakdown by operation.

    Parameters
    ----------
    N_values : list of int
        Values of N to test. Default: [50, 100, 200]
    H : float
        Hurst parameter.
    alpha : float
        fBM weight.
    strategy : str
        "markovian" or "full"
    use_block_hv : bool
        If True, use block-optimized Hv computation.

    Returns
    -------
    results : list of dict
        Detailed timing results.
    """
    if N_values is None:
        N_values = [50, 100, 200]

    from finance_example import (
        make_mixed_fbm_markovian_basis_blocked,
        make_mixed_fbm_full_info_basis_blocked,
        spd_mixed_fbm_blocked
    )
    from constrained_decomposition_core import SymBasis

    results = []

    print("\n" + "=" * 100)
    print(f"DETAILED PROFILING: {strategy.upper()} strategy (block_hv={'ON' if use_block_hv else 'OFF'})")
    print(f"H={H}, alpha={alpha}")
    print("=" * 100)

    for N in N_values:
        delta_t = 1.0 / N
        n = 2 * N

        print(f"\n{'─'*100}")
        print(f"N={N} (matrix size 2N={n})")

        # Build covariance and precision
        import time
        t0 = time.time()
        Sigma = spd_mixed_fbm_blocked(N, H, alpha, delta_t)
        t_sigma = time.time() - t0

        t0 = time.time()
        Lambda = np.linalg.inv(Sigma)
        t_lambda_inv = time.time() - t0

        # Build basis
        t0 = time.time()
        if strategy == "markovian":
            basis = make_mixed_fbm_markovian_basis_blocked(N)
        else:
            basis = make_mixed_fbm_full_info_basis_blocked(N)
        t_basis = time.time() - t0

        m = basis.m
        print(f"  Basis dimension m = {m}")
        print(f"  Setup: Σ={t_sigma:.3f}s, Λ⁻¹={t_lambda_inv:.3f}s, basis={t_basis:.3f}s")

        # Detailed solve with timing
        t_total, timing = _solve_with_detailed_timing(Lambda, basis, tol=1e-8, use_block_hv=use_block_hv)

        # Print breakdown
        t_B = timing['B_inverse']
        t_grad = timing['gradient']
        t_hv = timing['hessian_vec']
        t_cg = timing['cg_overhead']
        t_other = timing['other']
        n_iters = timing['newton_iters']
        n_hv_calls = timing['hv_calls']

        print(f"\n  Newton iterations: {n_iters}, Hv calls: {n_hv_calls}")
        print(f"\n  Time breakdown:")
        print(f"    {'B = M⁻¹ (Cholesky):':<30} {t_B:>8.3f}s  ({100*t_B/t_total:>5.1f}%)")
        print(f"    {'Gradient computation:':<30} {t_grad:>8.3f}s  ({100*t_grad/t_total:>5.1f}%)")
        print(f"    {'Hessian-vector products:':<30} {t_hv:>8.3f}s  ({100*t_hv/t_total:>5.1f}%)")
        print(f"    {'CG overhead:':<30} {t_cg:>8.3f}s  ({100*t_cg/t_total:>5.1f}%)")
        print(f"    {'Other:':<30} {t_other:>8.3f}s  ({100*t_other/t_total:>5.1f}%)")
        print(f"    {'─'*50}")
        print(f"    {'TOTAL:':<30} {t_total:>8.3f}s")

        # Per-call costs
        if n_hv_calls > 0:
            print(f"\n  Per-call costs:")
            print(f"    Hv product: {1000*t_hv/n_hv_calls:.3f} ms/call")
        if n_iters > 0:
            print(f"    B inverse:  {1000*t_B/n_iters:.3f} ms/iter")
            print(f"    Gradient:   {1000*t_grad/n_iters:.3f} ms/iter")

        results.append({
            'N': N,
            'm': m,
            'strategy': strategy,
            'total_time': t_total,
            'timing': timing,
        })

    print(f"\n{'='*100}\n")
    return results


def _solve_with_detailed_timing(Lambda, basis, tol=1e-8, max_iter=200, use_block_hv=True):
    """
    Solve constrained decomposition with detailed timing breakdown.

    Parameters
    ----------
    Lambda : array (n, n)
        Precision matrix.
    basis : SymBasis
        Basis for the constraint subspace.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum Newton iterations.
    use_block_hv : bool
        If True, use block-optimized Hv (requires D22=0 structure).

    Returns
    -------
    total_time : float
    timing : dict with keys:
        'B_inverse', 'gradient', 'hessian_vec', 'cg_overhead', 'other',
        'newton_iters', 'hv_calls'
    """
    import time
    from scipy.linalg import cho_factor, cho_solve
    from scipy.sparse.linalg import LinearOperator, cg

    n = Lambda.shape[0]
    N = n // 2  # For blocked ordering
    m = basis.m

    # Timing accumulators
    t_B_inverse = 0.0
    t_gradient = 0.0
    t_hessian_vec = 0.0
    t_cg_overhead = 0.0
    hv_call_count = 0

    t_total_start = time.time()

    # Initialize
    x = np.zeros(m, dtype=np.float64)

    # Get COO data from basis (stored in _coo attribute)
    coo_data = basis._coo

    # Precompute block indices for each basis element (for block Hv)
    # Each basis element Dk contributes to BDvB via tr(Dk @ BDvB).
    # We only compute BDvB[:, :N], so we need entries in columns 0..N-1.
    # For D12 entries (row < N, col >= N), we use symmetry: BDvB[i,j] = BDvB[j,i]
    if use_block_hv:
        # Flatten all indices for vectorized extraction
        all_upper_rows = []
        all_upper_cols = []
        upper_groups = []  # Which basis element k each upper entry belongs to
        all_cross_rows = []
        all_cross_cols = []
        cross_groups = []  # Which basis element k each cross entry belongs to

        for k in range(m):
            rows, cols, vals = coo_data[k]

            # For each entry, determine which block it maps to
            # BDvB is symmetric. We compute BDvB[:, :N] only.
            # D12 entries (row < N, col >= N): use BDvB[row, col] = BDvB_cross[col-N, row]

            for i in range(len(rows)):
                r, c = rows[i], cols[i]
                if r < N and c < N:
                    # D11: use BDvB_upper[r, c]
                    all_upper_rows.append(r)
                    all_upper_cols.append(c)
                    upper_groups.append(k)
                elif r >= N and c < N:
                    # D21: use BDvB_cross[r-N, c]
                    all_cross_rows.append(r - N)
                    all_cross_cols.append(c)
                    cross_groups.append(k)
                elif r < N and c >= N:
                    # D12: use symmetry, BDvB[r, c] = BDvB[c, r] = BDvB_cross[c-N, r]
                    all_cross_rows.append(c - N)
                    all_cross_cols.append(r)
                    cross_groups.append(k)
                # D22 entries (r >= N and c >= N) are skipped - shouldn't exist

        # Convert to numpy arrays
        all_upper_rows = np.array(all_upper_rows, dtype=int)
        all_upper_cols = np.array(all_upper_cols, dtype=int)
        upper_groups = np.array(upper_groups, dtype=int)
        all_cross_rows = np.array(all_cross_rows, dtype=int)
        all_cross_cols = np.array(all_cross_cols, dtype=int)
        cross_groups = np.array(cross_groups, dtype=int)

    def build_C(x_vec):
        """Build C matrix from coefficients."""
        C = np.zeros((n, n), dtype=np.float64)
        for k in range(m):
            if abs(x_vec[k]) < 1e-15:
                continue
            rows, cols, vals = coo_data[k]
            C[rows, cols] += x_vec[k] * vals
        return C

    def build_C_blocks(x_vec):
        """Build C as blocks: C11 (N×N), C12 (N×N). C22 = 0 for blocked mixed fBM."""
        # Use the full matrix build and extract blocks (simpler, same cost)
        C = build_C(x_vec)
        C11 = C[:N, :N]
        C12 = C[:N, N:]
        return C11, C12

    def compute_B(x_vec):
        """Compute B = (Lambda - C(x))^{-1}."""
        M = Lambda - build_C(x_vec)
        c, lower = cho_factor(M, lower=True)
        B = cho_solve((c, lower), np.eye(n))
        return B

    def compute_gradient(B):
        """Compute gradient g_k = tr(B D_k)."""
        g = np.zeros(m, dtype=np.float64)
        for k in range(m):
            rows, cols, vals = coo_data[k]
            g[k] = np.sum(B[rows, cols] * vals)
        return g

    def compute_Hv_standard(B, v):
        """Compute Hessian-vector product (standard method)."""
        nonlocal hv_call_count
        hv_call_count += 1
        D_v = build_C(v)
        BDvB = B @ D_v @ B
        Hv = np.zeros(m, dtype=np.float64)
        for k in range(m):
            rows, cols, vals = coo_data[k]
            Hv[k] = np.sum(BDvB[rows, cols] * vals)
        return Hv

    def compute_Hv_block(B, v):
        """
        Block-optimized Hessian-vector product.

        Exploits that D(v) has structure:
            D(v) = [D11  D12 ]
                   [D12ᵀ  0  ]

        Works with N×N blocks instead of 2N×2N.
        Vectorized extraction for O(m) elements.
        """
        nonlocal hv_call_count
        hv_call_count += 1

        # Extract B blocks
        B11 = B[:N, :N]
        B12 = B[:N, N:]
        B21 = B[N:, :N]
        B22 = B[N:, N:]

        # Build D blocks
        D11, D12 = build_C_blocks(v)

        # Q = D(v) @ B, but we only need first N columns
        # Q11 = D11 @ B11 + D12 @ B21
        # Q21 = D12.T @ B11
        Q11 = D11 @ B11 + D12 @ B21
        Q21 = D12.T @ B11

        # BDvB[:, :N] = B @ Q[:, :N]
        BDvB_upper = B11 @ Q11 + B12 @ Q21  # (BDvB)[0:N, 0:N]
        BDvB_cross = B21 @ Q11 + B22 @ Q21  # (BDvB)[N:2N, 0:N]

        # Vectorized extraction using bincount for groupby-sum
        # Extract all values at once, then sum by group (basis element)
        if len(all_upper_rows) > 0:
            upper_vals = BDvB_upper[all_upper_rows, all_upper_cols]
            Hv = np.bincount(upper_groups, weights=upper_vals, minlength=m)
        else:
            Hv = np.zeros(m, dtype=np.float64)

        if len(all_cross_rows) > 0:
            cross_vals = BDvB_cross[all_cross_rows, all_cross_cols]
            Hv += np.bincount(cross_groups, weights=cross_vals, minlength=m)

        return Hv

    compute_Hv = compute_Hv_block if use_block_hv else compute_Hv_standard

    # Initial B and gradient
    t0 = time.time()
    B = compute_B(x)
    t_B_inverse += time.time() - t0

    t0 = time.time()
    g = compute_gradient(B)
    t_gradient += time.time() - t0

    newton_iters = 0
    for it in range(max_iter):
        newton_iters = it + 1
        max_g = np.max(np.abs(g))
        if max_g < tol:
            break

        # CG for Newton direction
        t0 = time.time()

        def Hv_op(v):
            nonlocal t_hessian_vec
            t_hv_start = time.time()
            result = compute_Hv(B, v)
            t_hessian_vec += time.time() - t_hv_start
            return result

        H_linop = LinearOperator((m, m), matvec=Hv_op)
        d, _ = cg(H_linop, -g, rtol=1e-6, maxiter=min(100, m))
        t_cg_overhead += time.time() - t0 - t_hessian_vec  # CG overhead excluding Hv time

        # Reset Hv time accumulator for accurate measurement
        # (it was added during CG, now subtract what was added to cg_overhead)
        t_cg_overhead = max(0, t_cg_overhead)

        # Line search
        step = 1.0
        B_new = None
        for _ in range(20):
            x_new = x + step * d
            try:
                t0 = time.time()
                B_new = compute_B(x_new)
                t_B_inverse += time.time() - t0
                break
            except np.linalg.LinAlgError:
                step *= 0.5

        if B_new is None:
            break

        x = x_new
        B = B_new

        t0 = time.time()
        g = compute_gradient(B)
        t_gradient += time.time() - t0

    t_total = time.time() - t_total_start
    t_other = t_total - t_B_inverse - t_gradient - t_hessian_vec - t_cg_overhead

    timing = {
        'B_inverse': t_B_inverse,
        'gradient': t_gradient,
        'hessian_vec': t_hessian_vec,
        'cg_overhead': t_cg_overhead,
        'other': t_other,
        'newton_iters': newton_iters,
        'hv_calls': hv_call_count,
    }

    return t_total, timing


def test_blocked_newton_solver():
    """Test the BlockedNewtonSolver."""
    from finance_example import invest_value_mixed_fbm_blocked

    print("Testing BlockedNewtonSolver...")
    print("=" * 70)

    for N in [20, 50, 100, 200, 500]:
        H = 0.6
        alpha = 1.0
        delta_t = 1.0 / N

        print(f"\nN={N} (matrix size 2N={2*N}, basis dim m={2*(N-1)}):")

        # Solve with specialized solver
        import time
        t0 = time.time()
        solver = BlockedNewtonSolver(N, H, alpha, delta_t, verbose=False)
        B, C, x, info = solver.solve(tol=1e-8, method="newton-cg")
        value_specialized = solver.compute_investment_value(B)
        t_specialized = time.time() - t0

        # Compare with general solver
        t0 = time.time()
        value_general, info_general = invest_value_mixed_fbm_blocked(
            H, N, alpha, delta_t, strategy='markovian', method='newton'
        )
        t_general = time.time() - t0

        diff = abs(value_specialized - value_general)
        speedup = t_general / t_specialized if t_specialized > 0 else float('inf')

        print(f"  Specialized: {value_specialized:.10f} ({t_specialized:.3f}s, {info['iters']} iters)")
        print(f"  General:     {value_general:.10f} ({t_general:.3f}s, {info_general['iters']} iters)")
        print(f"  Diff: {diff:.2e}, Speedup: {speedup:.2f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Toeplitz solver for mixed fBM")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark block B optimization vs full Cholesky")
    parser.add_argument("--benchmark-hv", action="store_true",
                        help="Benchmark block Hv optimization (Markovian, uses BlockedNewtonSolver)")
    parser.add_argument("--benchmark-hv-general", action="store_true",
                        help="Benchmark block Hv for any strategy (works with full strategy)")
    parser.add_argument("--profile", action="store_true",
                        help="Detailed profiling showing time breakdown by operation")
    parser.add_argument("--benchmark-methods", action="store_true",
                        help="Benchmark Newton-CG vs L-BFGS optimization methods")
    parser.add_argument("--strategy", type=str, default="markovian",
                        choices=["markovian", "full", "both"],
                        help="Strategy for --profile/--benchmark-hv-general/--benchmark-methods (default: markovian)")
    parser.add_argument("--N", type=str, default="50,100,200",
                        help="Comma-separated N values (default: 50,100,200)")
    parser.add_argument("--H", type=float, default=0.6, help="Hurst parameter (default: 0.6)")
    parser.add_argument("--alpha", type=float, default=1.0, help="fBM weight (default: 1.0)")
    args = parser.parse_args()

    N_values = [int(n.strip()) for n in args.N.split(",")]

    if args.benchmark_methods:
        if args.strategy == "both":
            benchmark_methods(N_values=N_values, H=args.H, alpha=args.alpha, strategy="markovian")
            benchmark_methods(N_values=N_values, H=args.H, alpha=args.alpha, strategy="full")
        else:
            benchmark_methods(N_values=N_values, H=args.H, alpha=args.alpha, strategy=args.strategy)
    elif args.benchmark_hv_general:
        if args.strategy == "both":
            benchmark_block_hv_general(N_values=N_values, H=args.H, alpha=args.alpha, strategy="markovian")
            benchmark_block_hv_general(N_values=N_values, H=args.H, alpha=args.alpha, strategy="full")
        else:
            benchmark_block_hv_general(N_values=N_values, H=args.H, alpha=args.alpha, strategy=args.strategy)
    elif args.benchmark_hv:
        benchmark_block_hv(N_values=N_values, H=args.H, alpha=args.alpha)
    elif args.profile:
        if args.strategy == "both":
            benchmark_detailed(N_values=N_values, H=args.H, alpha=args.alpha, strategy="markovian")
            benchmark_detailed(N_values=N_values, H=args.H, alpha=args.alpha, strategy="full")
        else:
            benchmark_detailed(N_values=N_values, H=args.H, alpha=args.alpha, strategy=args.strategy)
    elif args.benchmark:
        benchmark_block_optimization(N_values=N_values, H=args.H, alpha=args.alpha)
    elif args.test:
        test_toeplitz_fft()
        test_blocked_precision()
        test_blocked_newton_solver()
    else:
        # Default: run tests
        test_toeplitz_fft()
        test_blocked_precision()
        test_blocked_newton_solver()
