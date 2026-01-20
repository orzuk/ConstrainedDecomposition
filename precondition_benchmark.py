
"""
precondition_benchmark.py

Benchmark "standard" vs "decomposition-based" preconditioners for SPD linear systems.

What's new vs previous version:
- spd_generate(n, matrix_type, **params): unified matrix generator that routes to your constrained_decomposition_utils.py
- stable condition-number estimation for P^{-1}A using symmetric form L^{-1} A L^{-T}
- benchmark over multiple RHS (default 10) with median iterations/time
- pipeline function benchmark_suite(...) to run multiple matrix types in one go

Decomposition-based preconditioner:
- P = B^{-1}, where (B,C) comes from your decomposition routine.

CURRENT LIMITATION:
Your uploaded constrained_decomposition_utils.py implements a specific tridiagonal subspace S (via build_C_from_x).
Therefore, "my" preconditioner is currently supported only for:
    my_class == "tridiagonal_markov"

Standard preconditioners supported:
    "diagonal"/"jacobi", "tridiagonal", "banded" (with bandwidth=k)

Run:
    python precondition_benchmark.py

Or import and call benchmark_suite / benchmark_preconditioners.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np

# ---- Import your existing code (same folder or PYTHONPATH) ----
import constrained_decomposition_core as bc
from constrained_decomposition_matrices import *


# -----------------------------
# Utilities
# -----------------------------
def sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def chol_factor(A: np.ndarray) -> np.ndarray:
    """Return lower-triangular Cholesky factor L s.t. A = L L^T."""
    return np.linalg.cholesky(A)


def chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (L L^T) x = b."""
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def apply_preconditioner_from_chol(L: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return function z = M^{-1} r given Cholesky factor of M."""
    def apply(r: np.ndarray) -> np.ndarray:
        return chol_solve(L, r)
    return apply


def spectral_condition_number_spd(M: np.ndarray) -> Tuple[float, float, float]:
    """
    Exact spectral condition number for SPD matrix (small/medium n).
    Returns (cond, lam_min, lam_max).
    """
    M = sym(M)
    w = np.linalg.eigvalsh(M)
    wmin = float(w[0])
    wmax = float(w[-1])
    if wmin <= 0:
        return (np.inf, wmin, wmax)
    return (wmax / wmin, wmin, wmax)


def cond_preconditioned(A: np.ndarray, P: np.ndarray) -> Tuple[float, float, float]:
    """
    Stable computation of eigenvalue extremes of P^{-1}A for SPD P:
        P = L L^T
        X = L^{-1} A L^{-T}  (symmetric SPD, similar to P^{-1}A)
    Returns (cond, lam_min, lam_max) of X.
    """
    A = sym(A)
    P = sym(P)
    L = chol_factor(P)
    # Compute X = L^{-1} A L^{-T} without forming inverses
    Y = np.linalg.solve(L, A)       # Y = L^{-1} A
    X = np.linalg.solve(L, Y.T).T   # X = (L^{-1} A) L^{-T}
    X = sym(X)
    return spectral_condition_number_spd(X)


# -----------------------------
# Matrix generator
# -----------------------------
def spd_generate(n: int, matrix_type: str, **params: Any) -> np.ndarray:
    """
    Unified SPD matrix generator that routes to functions in your B_C_Decomposition module.

    matrix_type (case-insensitive):
      - "fractional_bm" / "fbm": uses bc.spd_fractional_BM(n, H=..., T=...)
      - "hilbert": uses bc.spd_hilbert(n)
      - "ar1": uses bc.spd_ar1(n, rho=...)
      - "toeplitz_exp": uses bc.spd_toeplitz_ar1(n, rho=...) if available
      - "random_spd": uses bc.random_spd(n, cond=...) if available, else fallback
      - "wishart": uses bc.spd_wishart(n, df=..., seed=...) if available, else fallback

    If a requested type is not available in bc, raises ValueError with a helpful message.
    """
    t = matrix_type.lower()

    # fractional BM
    if t in {"fractional_bm", "fbm", "fractional_brownian_motion"}:
        if not hasattr(bc, "spd_fractional_BM"):
            raise ValueError("bc.spd_fractional_BM not found in constrained_decomposition_utils.py")
        H = float(params.get("H", 0.8))
        T = float(params.get("T", 1.0))
        return sym(bc.spd_fractional_BM(n, H=H, T=T))

    # Hilbert
    if t in {"hilbert", "spd_hilbert"}:
        if not hasattr(bc, "spd_hilbert"):
            raise ValueError("bc.spd_hilbert not found in constrained_decomposition_utils.py")
        return sym(bc.spd_hilbert(n))

    # AR(1)
    if t in {"ar1", "auto_regressive", "autoregressive"}:
        if not hasattr(bc, "spd_ar1"):
            raise ValueError("bc.spd_ar1 not found in constrained_decomposition_utils.py")
        rho = float(params.get("rho", 0.9))
        return sym(bc.spd_ar1(n, rho=rho))

    # Exponential Toeplitz (if you have it)
    if t in {"toeplitz_exp", "toeplitz", "exp_toeplitz", "toeplitz_ar1"}:
        if hasattr(bc, "spd_toeplitz_ar1"):
            rho = float(params.get("rho", 0.9))
            return sym(bc.spd_toeplitz_ar1(n, rho=rho))
        elif hasattr(bc, "spd_toeplitz"):
            rho = float(params.get("rho", 0.9))
            return sym(bc.spd_toeplitz(n, rho=rho))
        else:
            raise ValueError("Neither bc.spd_toeplitz_ar1 nor bc.spd_toeplitz found.")

    # Random SPD (if provided)
    if t in {"random_spd", "random"}:
        if hasattr(bc, "random_spd"):
            cond = float(params.get("cond", 1e4))
            seed = params.get("seed", None)
            return sym(bc.random_spd(n, cond=cond, seed=seed))
        # Fallback: random orthogonal * diagonal * orthogonal^T
        rng = np.random.default_rng(params.get("seed", 0))
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        # geometric spread
        cond = float(params.get("cond", 1e4))
        exponents = np.linspace(0.0, 1.0, n)
        d = cond ** exponents
        A = Q @ np.diag(d) @ Q.T
        return sym(A)

    # Wishart (if provided)
    if t in {"wishart"}:
        if hasattr(bc, "spd_wishart"):
            df = int(params.get("df", n + 5))
            seed = params.get("seed", None)
            return sym(bc.spd_wishart(n, df=df, seed=seed))
        # Fallback Wishart-ish
        rng = np.random.default_rng(params.get("seed", 0))
        df = int(params.get("df", n + 5))
        X = rng.standard_normal((df, n))
        A = X.T @ X + 1e-6 * np.eye(n)
        return sym(A)

    raise ValueError(
        f"Unknown matrix_type='{matrix_type}'. "
        f"Supported: fbm, hilbert, ar1, toeplitz_exp, random_spd, wishart."
    )


# -----------------------------
# Standard preconditioners
# -----------------------------
def build_standard_preconditioner(A: np.ndarray, structural_class: str, *,
                                  bandwidth: Optional[int] = None,
                                  jitter: float = 1e-10) -> np.ndarray:
    """
    Build a simple SPD preconditioner P for A based on a structural family.

    Returns P such that we will apply P^{-1} via Cholesky factorization of P.

    structural_class:
        - "jacobi" or "diagonal"
        - "tridiagonal"
        - "banded"  (requires bandwidth=k)
    """
    A = sym(np.asarray(A, dtype=float))
    n = A.shape[0]

    sc = structural_class.lower()
    if sc in {"jacobi", "diagonal"}:
        d = np.diag(A).copy()
        d = np.maximum(d, jitter)
        P = np.diag(d)

    elif sc == "tridiagonal":
        P = np.zeros_like(A)
        idx = np.arange(n)
        P[idx, idx] = np.diag(A)
        P[idx[:-1], idx[1:]] = np.diag(A, 1)
        P[idx[1:], idx[:-1]] = np.diag(A, -1)
        P = sym(P)
        P += jitter * np.eye(n)

    elif sc == "banded":
        if bandwidth is None or bandwidth < 0:
            raise ValueError("For structural_class='banded', provide bandwidth (nonnegative int).")
        k = int(bandwidth)
        P = np.zeros_like(A)
        for off in range(-k, k + 1):
            diag = np.diag(A, off)
            if off >= 0:
                P[np.arange(n - off), np.arange(off, n)] = diag
            else:
                P[np.arange(-off, n), np.arange(n + off)] = diag
        P = sym(P)
        P += jitter * np.eye(n)

    else:
        raise ValueError(f"Unknown structural_class='{structural_class}'.")

    if not bc.is_spd(P):
        t = jitter
        for _ in range(14):
            P_try = P + t * np.eye(n)
            if bc.is_spd(P_try):
                P = P_try
                break
            t *= 10.0
        else:
            raise np.linalg.LinAlgError("Failed to make standard preconditioner SPD.")
    return P


# -----------------------------
# Your decomposition-based preconditioner
# -----------------------------
@dataclass
class MyPreconditionerResult:
    P: np.ndarray               # P = B^{-1}
    B: np.ndarray
    C: np.ndarray
    x: np.ndarray
    runtime_sec: float
    residual_max_r: float       # max |Bkk - Bk,k+1| for your current structure


def build_my_preconditioner(A: np.ndarray, my_class: str, *,
                            tol: float = 1e-8,
                            max_iter: int = 500,
                            method: str = "newton",
                            verbose: bool = False) -> MyPreconditionerResult:
    """
    Compute your decomposition-based preconditioner.

    Supported:
        my_class == "tridiagonal_markov"
            Uses bc.constrained_decomposition on input A (must be SPD).
            Returns P = B^{-1}.
    """
    A = sym(np.asarray(A, dtype=float))

    eps = 1e-12 * np.trace(A) / A.shape[0]
    try:
        np.linalg.cholesky(A + eps * np.eye(A.shape[0]))
    except np.linalg.LinAlgError:
        raise ValueError("A must be SPD.")
    if not bc.is_spd(A):
        raise ValueError("A must be SPD for 'my' preconditioner.")

    sc = my_class.lower()
    if sc not in {"tridiagonal_markov"}:
        raise ValueError(
            "Your current decomposition code supports only my_class='tridiagonal_markov' "
            "(the specific tridiagonal S in build_C_from_x)."
        )

    t0 = time.perf_counter()
    basis = TridiagC_Basis(A.shape[0])
    B, C, x = constrained_decomposition(A=A, basis=basis, tol=tol, max_iter=max_iter, method=method, verbose=verbose)
    t1 = time.perf_counter()

    P = bc.spd_inverse(B)  # see next fix

    n = B.shape[0]
    r = np.array([B[k, k] - B[k, k + 1] for k in range(n - 1)]) if n > 1 else np.array([0.0])
    max_r = float(np.max(np.abs(r)))

    if not bc.is_spd(P):
        P += 1e-12 * np.eye(P.shape[0])

    return MyPreconditionerResult(
        P=P, B=B, C=C, x=x,
        runtime_sec=float(t1 - t0),
        residual_max_r=max_r
    )


# -----------------------------
# Preconditioned Conjugate Gradient
# -----------------------------
@dataclass
class PCGResult:
    iters: int
    converged: bool
    rel_resid: float
    runtime_sec: float


def pcg(A: np.ndarray,
        b: np.ndarray,
        M_inv: Callable[[np.ndarray], np.ndarray],
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-8,
        max_iter: Optional[int] = None,
        progress_every: Optional[int] = None) -> PCGResult:
    """
    Preconditioned Conjugate Gradient for SPD A.

    Convergence check: ||r||/||b|| <= tol.
    """
    A = sym(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    if max_iter is None:
        max_iter = 5 * n

    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=float).copy()

    t0 = time.perf_counter()
    r = b - A @ x
    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        return PCGResult(iters=0, converged=True, rel_resid=0.0, runtime_sec=0.0)

    z = M_inv(r)
    p = z.copy()
    rz_old = float(r @ z)

    rel = np.linalg.norm(r) / bnorm
    if rel <= tol:
        return PCGResult(iters=0, converged=True, rel_resid=float(rel), runtime_sec=float(time.perf_counter()-t0))

    for k in range(1, max_iter + 1):
        Ap = A @ p
        alpha = rz_old / float(p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rel = np.linalg.norm(r) / bnorm
        if progress_every is not None and k % progress_every == 0:
            print(f"  PCG iter {k}, rel_resid={rel:.3e}")
        if rel <= tol:
            t1 = time.perf_counter()
            return PCGResult(iters=k, converged=True, rel_resid=float(rel), runtime_sec=float(t1-t0))

        z = M_inv(r)
        rz_new = float(r @ z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    t1 = time.perf_counter()
    return PCGResult(iters=max_iter, converged=False, rel_resid=float(rel), runtime_sec=float(t1-t0))


# -----------------------------
# Benchmark driver
# -----------------------------
@dataclass
class BenchmarkReport:
    matrix_type: str
    matrix_params: Dict[str, Any]
    n: int

    cond_A: float

    standard_class: str
    cond_PinvA_standard: float
    lam_min_standard: float
    lam_max_standard: float
    pcg_iters_standard_median: float
    pcg_time_standard_median: float

    my_class: str
    cond_PinvA_my: float
    lam_min_my: float
    lam_max_my: float
    pcg_iters_my_median: float
    pcg_time_my_median: float
    build_time_my: float
    kkt_residual_my: float


def _median(vals: List[float]) -> float:
    arr = np.array(vals, dtype=float)
    return float(np.median(arr))


def benchmark_preconditioners(A: np.ndarray,
                              standard_class: str,
                              my_class: str = "tridiagonal_markov",
                              *,
                              bandwidth: Optional[int] = None,
                              cg_tol: float = 1e-8,
                              cg_max_iter: Optional[int] = None,
                              rhs_trials: int = 10,
                              rhs_seed: int = 0,
                              decomp_tol: float = 1e-8,
                              decomp_max_iter: int = 500,
                              decomp_method: str = "newton",
                              compute_conds: bool = True,) -> Tuple[BenchmarkReport, Dict[str, Any]]:
    """
    Compare:
      - Standard preconditioner in 'standard_class'
      - Your decomposition-based preconditioner in 'my_class'

    Metrics:
      - stable condition number of P^{-1}A via L^{-1} A L^{-T}
      - PCG iterations/time over multiple RHS (median)
      - build time and KKT residual for your decomposition

    Returns (report, debug_dict).
    """
    A = sym(np.asarray(A, dtype=float))
    n = A.shape[0]
    if not bc.is_spd(A, jitter="auto"):
        raise ValueError("A must be SPD ...")

    cond_A, *_ = spectral_condition_number_spd(A)

    rng = np.random.default_rng(rhs_seed)
    Bs = [rng.standard_normal(n) for _ in range(rhs_trials)]

    # --- standard ---
    P_std = build_standard_preconditioner(A, standard_class, bandwidth=bandwidth)
    L_std = chol_factor(P_std)
    M_inv_std = apply_preconditioner_from_chol(L_std)

    if compute_conds:
        cond_std, lam_min_std, lam_max_std = cond_preconditioned(A, P_std)
    else:
        cond_std, lam_min_std, lam_max_std = (np.nan, np.nan, np.nan)


    iters_std = []
    time_std = []
    for b in Bs:
        res = pcg(A, b, M_inv_std, tol=cg_tol, max_iter=cg_max_iter)
        iters_std.append(res.iters)
        time_std.append(res.runtime_sec)

    # --- mine ---
    try:
        my = build_my_preconditioner(
            A,
            my_class,
            tol=decomp_tol,
            max_iter=decomp_max_iter,
            method=decomp_method,
            verbose=False,
        )
    except Exception as e:
        print("  [Decomposition-based] failed:", repr(e))
        my = None
    P_my = my.P
    L_my = chol_factor(P_my)
    M_inv_my = apply_preconditioner_from_chol(L_my)

    if compute_conds:
        cond_my, lam_min_my, lam_max_my = cond_preconditioned(A, P_my)
    else:
        cond_my, lam_min_my, lam_max_my = (np.nan, np.nan, np.nan)

    iters_my = []
    time_my = []
    for b in Bs:
        res = pcg(A, b, M_inv_my, tol=cg_tol, max_iter=cg_max_iter)
        iters_my.append(res.iters)
        time_my.append(res.runtime_sec)

    rep = BenchmarkReport(
        matrix_type="custom",
        matrix_params={},
        n=n,
        cond_A=float(cond_A),
        standard_class=standard_class,
        cond_PinvA_standard=float(cond_std),
        lam_min_standard=float(lam_min_std),
        lam_max_standard=float(lam_max_std),
        pcg_iters_standard_median=_median(iters_std),
        pcg_time_standard_median=_median(time_std),
        my_class=my_class,
        cond_PinvA_my=float(cond_my),
        lam_min_my=float(lam_min_my),
        lam_max_my=float(lam_max_my),
        pcg_iters_my_median=_median(iters_my),
        pcg_time_my_median=_median(time_my),
        build_time_my=float(my.runtime_sec),
        kkt_residual_my=float(my.residual_max_r),
    )

    debug = {
        "iters_std": iters_std,
        "iters_my": iters_my,
        "time_std": time_std,
        "time_my": time_my,
    }
    return rep, debug


def pretty_print_report(rep: BenchmarkReport) -> None:
    print("\n=== Preconditioner Benchmark Report ===")
    print(f"type = {rep.matrix_type}  params = {rep.matrix_params}")
    print(f"n = {rep.n}")
    print(f"cond(A) = {rep.cond_A:.3e}")

    print("\n[Standard]")
    print(f"  class = {rep.standard_class}")
    if np.isfinite(rep.lam_min_standard) and np.isfinite(rep.lam_max_standard):
        print(f"  eig(P^-0.5 A P^-0.5) min/max = {rep.lam_min_standard:.3e} / {rep.lam_max_standard:.3e}")
    else:
        print("  eig(P^-0.5 A P^-0.5) min/max = (skipped)")
    if np.isfinite(rep.cond_PinvA_standard):
        print(f"  cond(P^-1A) = {rep.cond_PinvA_standard:.3e}")
    else:
        print("  cond(P^-1A) = (skipped)")


    print(f"  PCG iters (median over RHS) = {rep.pcg_iters_standard_median:.1f}")
    print(f"  PCG time  (median over RHS) = {rep.pcg_time_standard_median:.3e} sec")

    print("\n[Decomposition-based]")
    print(f"  class = {rep.my_class}")
    if np.isfinite(rep.lam_min_my) and np.isfinite(rep.lam_max_my):
        print(f"  eig(P^-0.5 A P^-0.5) min/max = {rep.lam_min_my:.3e} / {rep.lam_max_my:.3e}")
    else:
        print("  eig(P^-0.5 A P^-0.5) min/max = (skipped)")

    if np.isfinite(rep.cond_PinvA_my):
        print(f"  cond(P^-1A) = {rep.cond_PinvA_my:.3e}")
    else:
        print("  cond(P^-1A) = (skipped)")

    print(f"  PCG iters (median over RHS) = {rep.pcg_iters_my_median:.1f}")
    print(f"  PCG time  (median over RHS) = {rep.pcg_time_my_median:.3e} sec")
    print(f"  build time (decomposition) = {rep.build_time_my:.3e} sec")
    print(f"  KKT residual (max|Bkk-Bk,k+1|) = {rep.kkt_residual_my:.3e}")


def benchmark_suite(n: int,
                    cases: List[Tuple[str, Dict[str, Any]]],
                    *,
                    standard_class: str = "tridiagonal",
                    standard_bandwidth: Optional[int] = None,
                    my_class: str = "tridiagonal_markov",
                    cg_tol: float = 1e-8,
                    cg_max_iter: int = 2000,
                    rhs_trials: int = 10,
                    rhs_seed: int = 0,
                    decomp_tol: float = 1e-6,
                    decomp_max_iter: int = 500,
                    decomp_method: str = "newton") -> List[BenchmarkReport]:
    """
    Run a suite of benchmarks over multiple (matrix_type, params) cases.

    Example:
        cases = [
          ("fbm", {"H": 0.8, "T": 1.0}),
          ("ar1", {"rho": 0.9}),
          ("hilbert", {}),
        ]
    """
    reports: List[BenchmarkReport] = []
    for matrix_type, params in cases:
        print("\n----------------------------------------------")
        print(f"Case: {matrix_type}  params={params}")
        A = spd_generate(n, matrix_type, **params)

        rep, _ = benchmark_preconditioners(
            A,
            standard_class=standard_class,
            my_class=my_class,
            bandwidth=standard_bandwidth,
            cg_tol=cg_tol,
            cg_max_iter=cg_max_iter,
            rhs_trials=rhs_trials,
            rhs_seed=rhs_seed,
            decomp_tol=decomp_tol,
            decomp_max_iter=decomp_max_iter,
            decomp_method=decomp_method,
            compute_conds=False,
        )
        rep.matrix_type = matrix_type
        rep.matrix_params = params
        pretty_print_report(rep)
        reports.append(rep)

    return reports


if __name__ == "__main__":
    # Smaller n => much faster (both eigvalsh and PCG scale badly with n)
    n = 80

    # Deterministic structured cases (1 matrix each)
    cases = [
        ("ar1", {"rho": 0.9}),
        ("ar1", {"rho": 0.5}),
        ("toeplitz_ar1", {"rho": 0.9}),   # requires bc.spd_toeplitz_ar1 or bc.spd_toeplitz
        ("fbm", {"H": 0.8, "T": 1.0}),    # deterministic; keep if you want, or remove if you already know it's bad
        ("hilbert", {}),
    ]

    # Add multiple random draws (many A's)
    # random_spd and wishart accept seed -> will produce different matrices
    for seed in range(5):  # change to 10/20 if you want more
        cases.append(("random_spd", {"cond": 1e4, "seed": seed}))
        cases.append(("wishart", {"df": n + 10, "seed": seed}))

    # Faster CG settings (for quick compare)
    benchmark_suite(
        n=n,
        cases=cases,
        standard_class="tridiagonal",
        my_class="tridiagonal_markov",
        cg_tol=1e-8,
        cg_max_iter=800,
        rhs_trials=2,
        rhs_seed=0,
        decomp_tol=1e-6,
        decomp_max_iter=200,
        decomp_method="newton",
    )

