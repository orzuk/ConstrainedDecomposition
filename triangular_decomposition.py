"""
Bilinear triangular decomposition Lambda = L + U + U A L (Dolinsky-Zuk 2026).

Specialisation of the constrained inverse decomposition
    A^{-1} = Q^{-1} + C,    Q in S^perp,   C in S,
to the off-diagonal-block constraint subspace
    S = { ((0, U), (U^T, 0)) : U strictly upper triangular } subset M_{2n}(R).

For positive definite A, Lambda in M_n(R) with Lambda = Sigma^{-1}, the
problem reduces to finding (U, L) with U strictly upper triangular and
L lower triangular such that

    Lambda = L + U + U A L.

Unlike the inverse decomposition (which always exists, paper's Prop 1),
this bilinear triangular decomposition (paper's Prop 2) can fail to
exist or fail to be unique exactly when the backward elimination algorithm
below hits a zero pivot. The pair (A, Lambda) is triangularly REGULAR when
the decomposition exists, IRREGULAR (codimension-1 in the SPD cone) when
not. The robust variant `backward_triangular_elimination_robust` returns
the unique U from Prop 1 on the entire SPD cone in finite arithmetic.

References
----------
- Dolinsky and Zuk, "A Unique Inverse Decomposition of Positive Definite
  Matrices under Linear Constraints," arXiv:2601.18662 (2026).
- Dolinsky and Zuk, "Exponential Utility Maximization with Quadratic
  Costs in a Discrete-Time Gaussian Framework," 2026.

The general primal/dual Newton solvers in
`constrained_decomposition_core.py` solve the inverse decomposition for
any constraint subspace. This module implements the *direct* algorithm
that works only for the strictly-upper-triangular subspace but runs in
closed-form arithmetic (no iteration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TriangularDecompositionInfo:
    """Diagnostic record for `backward_triangular_elimination`.

    U is *always* returned: when status == "ok" it is the U from the
    elimination itself; when status == "no_bilinear_L" the elimination
    has supplied a certificate that no lower-triangular L exists, but
    Proposition 1's U still exists and is computed via the iterative
    fallback so the finance theorem (Theorem 1) can still be applied.
    """
    status: str
    # "ok"             : bilinear triangular decomposition computed; both U and L returned
    # "no_bilinear_L"  : zero scalar pivot pi_k; U returned, L is None
    failed_step: Optional[int] = None    # 1-based row index where pi_k=0 was hit
    pivot: Optional[float] = None
    detail: str = ""


# -----------------------------------------------------------------------------
# Main algorithm
# -----------------------------------------------------------------------------

def backward_triangular_elimination(
    A: np.ndarray,
    Lambda: np.ndarray,
    tol: float = 1e-12,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], TriangularDecompositionInfo]:
    """
    Compute the unique pair (U, L) with U strictly upper triangular and L
    lower triangular such that Lambda = L + U + U A L, if it exists.

    Implements Algorithm 2 (backward triangular elimination) of the paper
    "Exponential Utility Maximization with Quadratic Costs in a
    Discrete-Time Gaussian Framework" by Dolinsky and Zuk (2026).

    Parameters
    ----------
    A      : (n, n) symmetric positive definite matrix (transaction-cost matrix).
    Lambda : (n, n) symmetric positive definite matrix (precision matrix).
    tol    : pivot tolerance for declaring a zero pivot.

    Returns
    -------
    U, L, info
        On success info.status == "ok" and Lambda == L + U + U A L to floating
        point. On failure U and L are None and info describes the obstruction.
    """
    A = np.asarray(A, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or Lambda.shape != (n, n):
        raise ValueError("A and Lambda must be square of the same size.")

    U = np.zeros((n, n), dtype=float)
    L = np.zeros((n, n), dtype=float)

    # k runs over the rows of U / columns of L from bottom to top.
    # In 0-based indexing, k = n-1 is the trivial trailing block.
    # Cost: O(n^4). For O(n^3) use `backward_triangular_elimination_efficient`.
    for k in range(n - 1, -1, -1):
        J = np.arange(k + 1, n)

        if J.size == 0:
            L[k, k] = Lambda[k, k]
            continue

        A0 = A[np.ix_(J, J)]
        U0 = U[np.ix_(J, J)]
        L0 = L[np.ix_(J, J)]
        alpha = A[J, k]
        beta = Lambda[J, k]
        lam = Lambda[k, k]

        # The two trailing-block solves below are always non-singular for SPD
        # inputs (Remark "single_pivot" in the paper): det(I+A_0 L_0) and
        # det(K_0) multiply to det(I+A_0 Lambda_0) > 1.
        u = np.linalg.solve((np.eye(J.size) + A0 @ L0).T, beta)
        U[k, J] = u

        K0 = np.eye(J.size) + U0 @ A0
        z = np.linalg.solve(K0, beta)
        w = np.linalg.solve(K0, U0 @ alpha)

        pi_k = 1.0 + float(u @ alpha) - float(u @ A0 @ w)

        # The only essential test. By Remark "dichotomy" the consistency
        # branch (rhs == 0 inside pi_k == 0) is unreachable for SPD inputs,
        # so a zero pivot always means: no lower-triangular L exists.
        # U from Prop 1 still exists; recompute it via the robust variant
        # (which handles the entire SPD cone in finite arithmetic) so
        # callers can always use U in the finance theorem.
        if abs(pi_k) < tol:
            U_full, _, _ = backward_triangular_elimination_robust(A, Lambda)
            return U_full, None, TriangularDecompositionInfo(
                status="no_bilinear_L",
                failed_step=k + 1,
                pivot=pi_k,
                detail=f"pi_k={pi_k:g}; U returned is Prop. 1's inverse-decomposition U.",
            )

        x = (lam - float(u @ A0 @ z)) / pi_k
        y = z - w * x
        L[k, k] = x
        L[J, k] = y

    return U, L, TriangularDecompositionInfo(status="ok")


# -----------------------------------------------------------------------------
# O(n^3) variant
# -----------------------------------------------------------------------------

def backward_triangular_elimination_efficient(
    A: np.ndarray,
    Lambda: np.ndarray,
    tol: float = 1e-12,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], TriangularDecompositionInfo]:
    """
    O(n^3) version of `backward_triangular_elimination`.

    Identical inputs and outputs. The reduction comes from maintaining
    incrementally the inverses
        M_inv = (I + A_0 L_0)^{-1},     N_inv = (I + U_0 A_0)^{-1},
    as the trailing block grows by one row+column per step. The Schur
    complement formula updates each inverse in O(|J|^2), and the per-step
    solves are matrix-vector products of cost O(|J|^2). Summing over k
    gives O(n^3), matching standard LU.
    """
    A = np.asarray(A, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or Lambda.shape != (n, n):
        raise ValueError("A and Lambda must be square of the same size.")

    U = np.zeros((n, n), dtype=float)
    L = np.zeros((n, n), dtype=float)

    # Step k = n - 1 (trivial).
    L[n - 1, n - 1] = Lambda[n - 1, n - 1]
    if n == 1:
        return U, L, TriangularDecompositionInfo(status="ok")

    # M_inv and N_inv act on the trailing block J = {k+1, ..., n-1}.
    # At k = n-1 they are 0x0; we initialise them when entering step n-2.
    M_inv = np.empty((0, 0), dtype=float)
    N_inv = np.empty((0, 0), dtype=float)

    for k in range(n - 2, -1, -1):
        J = np.arange(k + 1, n)
        J_prev = np.arange(k + 2, n)         # the trailing block from step k+1
        new_idx = k + 1                       # the index being prepended to J

        a_new = float(A[new_idx, new_idx])
        L_diag = float(L[new_idx, new_idx])
        alpha_A = A[J_prev, new_idx] if J_prev.size else np.empty(0)
        y_new = L[J_prev, new_idx] if J_prev.size else np.empty(0)
        u_row = U[new_idx, J_prev] if J_prev.size else np.empty(0)

        if J_prev.size == 0:
            M_inv = np.array([[1.0 / (1.0 + a_new * L_diag)]])
            N_inv = np.array([[1.0]])
        else:
            L0_old = L[np.ix_(J_prev, J_prev)]
            A0_old = A[np.ix_(J_prev, J_prev)]
            U0_old = U[np.ix_(J_prev, J_prev)]

            # ---- Schur update for M_inv = (I + A_0 L_0)^{-1} ----
            a_p = 1.0 + a_new * L_diag + float(alpha_A @ y_new)
            b_p = L0_old.T @ alpha_A
            c_p = alpha_A * L_diag + A0_old @ y_new
            p = M_inv @ c_p
            q = b_p @ M_inv
            s_M = a_p - float(b_p @ p)
            if abs(s_M) < tol:
                U_full, _, _ = backward_triangular_elimination_robust(A, Lambda)
                return U_full, None, TriangularDecompositionInfo(
                    status="no_bilinear_L", failed_step=k + 1, pivot=s_M,
                    detail="Schur complement of (I + A_0 L_0) is singular; "
                           "U returned is Prop. 1's inverse-decomposition U.",
                )
            M_new = np.empty((J.size, J.size))
            M_new[0, 0] = 1.0 / s_M
            M_new[0, 1:] = -q / s_M
            M_new[1:, 0] = -p / s_M
            M_new[1:, 1:] = M_inv + np.outer(p, q) / s_M
            M_inv = M_new

            # ---- Schur update for N_inv = (I + U_0 A_0)^{-1} ----
            a_pp = 1.0 + float(u_row @ alpha_A)
            b_pp = A0_old @ u_row
            c_pp = U0_old @ alpha_A
            p2 = N_inv @ c_pp
            q2 = b_pp @ N_inv
            s_N = a_pp - float(b_pp @ p2)
            if abs(s_N) < tol:
                U_full, _, _ = backward_triangular_elimination_robust(A, Lambda)
                return U_full, None, TriangularDecompositionInfo(
                    status="no_bilinear_L", failed_step=k + 1, pivot=s_N,
                    detail="Schur complement of (I + U_0 A_0) is singular; "
                           "U returned is Prop. 1's inverse-decomposition U.",
                )
            N_new = np.empty((J.size, J.size))
            N_new[0, 0] = 1.0 / s_N
            N_new[0, 1:] = -q2 / s_N
            N_new[1:, 0] = -p2 / s_N
            N_new[1:, 1:] = N_inv + np.outer(p2, q2) / s_N
            N_inv = N_new

        # ---- Step k: solve and write row k of U, column k of L ----
        alpha = A[J, k]
        beta = Lambda[J, k]
        lam = float(Lambda[k, k])

        u = M_inv.T @ beta                    # solves (I + A_0 L_0)^T u = beta
        U[k, J] = u

        U0_new = U[np.ix_(J, J)]
        z = N_inv @ beta                      # solves K_0 z = beta
        w = N_inv @ (U0_new @ alpha)          # solves K_0 w = U_0 alpha

        A0_new = A[np.ix_(J, J)]
        pi_k = 1.0 + float(u @ alpha) - float(u @ A0_new @ w)
        if abs(pi_k) < tol:
            U_full, _, _ = backward_triangular_elimination_robust(A, Lambda)
            return U_full, None, TriangularDecompositionInfo(
                status="no_bilinear_L", failed_step=k + 1, pivot=pi_k,
                detail=f"pi_k={pi_k:g}; U returned is Prop. 1's inverse-decomposition U.",
            )

        x = (lam - float(u @ A0_new @ z)) / pi_k
        y = z - w * x
        L[k, k] = x
        L[J, k] = y

    return U, L, TriangularDecompositionInfo(status="ok")


# -----------------------------------------------------------------------------
# Robust variant: continue past zero pivots via Sherman-Morrison limit
# -----------------------------------------------------------------------------

def backward_triangular_elimination_robust(
    A: np.ndarray,
    Lambda: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Robust backward triangular elimination that always returns the unique U.

    Handles every SPD pair (A, Lambda) via rank-r Sherman-Morrison
    corrections on BOTH the primal row solve and the dual K_0 solve.

    Per step k, with J = {k+1, ..., n-1} and B the set of (j, v_j)
    pairs from previous zero pivots (j in J, v_j the recorded rank-one
    divergent direction):
      Build V, E (J_size x |B cap J|): column i of V has v_i placed at
      row p_i = j_i - k - 1; column i of E is e_{p_i}. Set A' = A_0 V.
      Primal SM (paper eq. A.1): Solve M^T u = beta s.t. A'^T u = 0,
                  with M = I + A_0 L_{J,J}.
      Dual SM (paper eq. A.2): Solve [K_0^T; V^T] s = [A_0 u; 0]
                  via augmented lstsq. Consistent because
                  V^T(A_0 u) = (A_0 V)^T u = A'^T u = 0.
      Compute pi_k = 1 + u^T alpha - s^T (U_0 alpha) and
              L_kk = (lam - s^T beta) / pi_k.
      L_{J,k}: direct in regular case; NaN in singular case
              (entries are perturbation-dependent off the diagonal).
      Bad pivot (S7-S9 in paper Algorithm 2): append (k, v_k=(1,-w)^T)
              to B with w from constrained augmented solve.

    Parameters
    ----------
    A      : (n, n) symmetric positive definite (transaction-cost matrix).
    Lambda : (n, n) symmetric positive definite (precision matrix).
    tol    : pivot tolerance for declaring a zero pivot.

    Returns
    -------
    U : (n, n) ndarray
        The unique Prop 1 strictly upper triangular U (always correct).
    L : (n, n) ndarray
        Bilinear triangular L. In the regular case L is the full Prop 2 factor.
        In the singular case entries that diverge or are perturbation-
        dependent are NaN-flagged; canonical entries (L_kk in particular)
        are filled in via the dual SM scalars.
    info : dict
        - 'bad_pivots' : list of (k, pi_k) for each step that vanished
        - 'num_bad'    : number of vanished pivots
        - 'status'     : 'ok' or 'has_bad_pivots'
    """
    A = np.asarray(A, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or Lambda.shape != (n, n):
        raise ValueError("A and Lambda must be square of the same size.")

    U = np.zeros((n, n), dtype=float)
    L = np.zeros((n, n), dtype=float)
    bad_cols = []  # each: {'index': k, 'pi_k': float, 'direction': v_k}

    if n == 0:
        return U, L, {'bad_pivots': [], 'num_bad': 0, 'status': 'ok'}

    L[n - 1, n - 1] = Lambda[n - 1, n - 1]

    for k in range(n - 2, -1, -1):
        J = np.arange(k + 1, n)
        J_size = J.size
        A0 = A[np.ix_(J, J)]
        U0 = U[np.ix_(J, J)]
        L0 = L[np.ix_(J, J)]
        L0_finite = np.where(np.isnan(L0), 0.0, L0)
        alpha = A[J, k]
        beta = Lambda[J, k]
        lam = float(Lambda[k, k])

        # Build V_embed (J_size, r) and E_mat (J_size, r):
        # V_embed columns are v_j embedded at row pos_j = j-(k+1) in J;
        # E_mat columns are the indicator e_{pos_j} (unit vector at row pos_j).
        # All bad indices are > current k, so all sit in J.
        r = len(bad_cols)
        if r > 0:
            V_embed = np.zeros((J_size, r))
            E_mat = np.zeros((J_size, r))
            for i, b in enumerate(bad_cols):
                j = b['index']
                pos = j - (k + 1)
                v_full = b['direction']
                end = min(pos + len(v_full), J_size)
                V_embed[pos:end, i] = v_full[: end - pos]
                E_mat[pos, i] = 1.0
        else:
            V_embed = np.zeros((J_size, 0))
            E_mat = np.zeros((J_size, 0))

        # --- PRIMAL SM: solve M^T u = beta with A'^T u = 0 ---
        # M(eps) = M_finite + sum_j alpha_j (A_0 v_j^embed) e_{pos_j}^T
        # ==> (M^T + alpha E (A')^T) u = beta, alpha -> infinity
        # ==> u = u_base - M^{-T} E (A'^T M^{-T} E)^{-1} A'^T u_base.
        M_finite = np.eye(J_size) + A0 @ L0_finite
        try:
            u_base = np.linalg.solve(M_finite.T, beta)
        except np.linalg.LinAlgError:
            u_base, *_ = np.linalg.lstsq(M_finite.T, beta, rcond=None)

        if r > 0:
            A_prime = A0 @ V_embed  # (J_size, r)
            try:
                MtinvE = np.linalg.solve(M_finite.T, E_mat)
            except np.linalg.LinAlgError:
                MtinvE, *_ = np.linalg.lstsq(M_finite.T, E_mat, rcond=None)
            denom = A_prime.T @ MtinvE  # (r, r)
            try:
                lhs = np.linalg.solve(denom, A_prime.T @ u_base)
            except np.linalg.LinAlgError:
                lhs, *_ = np.linalg.lstsq(denom, A_prime.T @ u_base, rcond=None)
            u = u_base - MtinvE @ lhs
        else:
            u = u_base

        U[k, J] = u

        # --- DUAL SM: solve K_0^T s = A_0 u with V^T s = 0 ---
        K0 = np.eye(J_size) + U0 @ A0
        A0u = A0 @ u
        if r > 0:
            aug_K0T = np.vstack([K0.T, V_embed.T])
            aug_rhs = np.concatenate([A0u, np.zeros(r)])
            s, *_ = np.linalg.lstsq(aug_K0T, aug_rhs, rcond=None)
        else:
            try:
                s = np.linalg.solve(K0.T, A0u)
            except np.linalg.LinAlgError:
                s, *_ = np.linalg.lstsq(K0.T, A0u, rcond=None)

        # --- pi_k, L_kk via dual SM scalars ---
        U0_alpha = U0 @ alpha
        pi_k = 1.0 + float(u @ alpha) - float(s @ U0_alpha)

        if abs(pi_k) < tol:
            # Bad pivot. Compute v_k = (1, -w^T)^T via constrained K_0 w = U_0 alpha.
            if r > 0:
                aug_K0 = np.vstack([K0, V_embed.T])
                aug_w_rhs = np.concatenate([U0_alpha, np.zeros(r)])
                w, *_ = np.linalg.lstsq(aug_K0, aug_w_rhs, rcond=None)
            else:
                try:
                    w = np.linalg.solve(K0, U0_alpha)
                except np.linalg.LinAlgError:
                    w, *_ = np.linalg.lstsq(K0, U0_alpha, rcond=None)
            v_k = np.concatenate([[1.0], -w])
            bad_cols.append({'index': k, 'pi_k': pi_k, 'direction': v_k})
            L[k, k] = np.nan
            L[J, k] = np.nan
        else:
            L_kk = (lam - float(s @ beta)) / pi_k
            L[k, k] = L_kk
            if r == 0:
                z = np.linalg.solve(K0, beta)
                w = np.linalg.solve(K0, U0_alpha)
            else:
                # Singular subproblem: K_0 is rank-deficient, but the
                # augmented system [K_0; V^T] z = [beta; 0] picks the
                # V-orthogonal z (likewise w). These values are needed
                # for subsequent steps' M = I + A_0 L_finite — flagging
                # them NaN here corrupts later computations.
                aug_K0 = np.vstack([K0, V_embed.T])
                rhs_z = np.concatenate([beta, np.zeros(r)])
                rhs_w = np.concatenate([U0_alpha, np.zeros(r)])
                z, *_ = np.linalg.lstsq(aug_K0, rhs_z, rcond=None)
                w, *_ = np.linalg.lstsq(aug_K0, rhs_w, rcond=None)
            L[J, k] = z - w * L_kk

    info = {
        'bad_pivots': [(b['index'], b['pi_k']) for b in bad_cols],
        'num_bad': len(bad_cols),
        'status': 'ok' if not bad_cols else 'has_bad_pivots',
    }
    return U, L, info


# -----------------------------------------------------------------------------
# Robust O(n^3) variant: incremental inverse maintenance + SM corrections
# -----------------------------------------------------------------------------

def backward_triangular_elimination_robust_efficient(
    A: np.ndarray,
    Lambda: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    O(n^3) version of `backward_triangular_elimination_robust` (regular phase).

    Maintains the trailing-block inverses
        M_inv  = (I + A_0 L_0^finite)^{-1},    N_inv = K_0^{-1} = (I + U_0 A_0)^{-1},
    via Schur-bordering updates as the trailing block J = {k+1, ..., n}
    grows, with NaN columns of L read as 0 throughout.

    M_inv is always maintained (M_finite stays non-singular even after a
    bad pivot, because the NaN-as-0 convention zeroes out the divergent
    column of L). The primal Sherman-Morrison correction uses M_inv
    directly, so each step costs O(|J|^2) + O(r |J|^2 + r^3) where
    r = |B cap J| is the number of bad pivots so far.

    N_inv is maintained while K_0 is non-singular (i.e., while B is empty).
    Once a pivot vanishes, K_0 becomes singular at the next step and
    further N_inv updates fail; from then on the dual augmented system
    [K_0^T; V^T] s = [A_0 u; 0] is solved from scratch via lstsq at each
    step (O(|J|^3) per step in the singular phase).

    Total complexity:
        - O(n^3) overall when no bad pivots occur (matches `_efficient`).
        - O(n^3) + O(n^4) worst-case when the first bad pivot occurs early;
          in practice when |B| = O(1) and the bad pivot is near step 1,
          the singular phase is short and total cost is close to O(n^3).

    Returns the same (U, L, info) triple as `backward_triangular_elimination_robust`.
    """
    A = np.asarray(A, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or Lambda.shape != (n, n):
        raise ValueError("A and Lambda must be square of the same size.")

    U = np.zeros((n, n), dtype=float)
    L = np.zeros((n, n), dtype=float)
    bad_cols = []

    if n == 0:
        return U, L, {'bad_pivots': [], 'num_bad': 0, 'status': 'ok'}

    L[n - 1, n - 1] = Lambda[n - 1, n - 1]
    if n == 1:
        return U, L, {'bad_pivots': [], 'num_bad': 0, 'status': 'ok'}

    M_inv = np.empty((0, 0), dtype=float)
    N_inv = np.empty((0, 0), dtype=float)
    N_inv_valid = True  # K_0^{-1} cached and current

    for k in range(n - 2, -1, -1):
        J = np.arange(k + 1, n)
        J_prev = np.arange(k + 2, n)
        new_idx = k + 1
        J_size = J.size

        a_new = float(A[new_idx, new_idx])
        L_diag_raw = L[new_idx, new_idx]
        L_diag = 0.0 if np.isnan(L_diag_raw) else float(L_diag_raw)
        alpha_A = A[J_prev, new_idx] if J_prev.size else np.empty(0)
        y_new_raw = L[J_prev, new_idx] if J_prev.size else np.empty(0)
        y_new = np.where(np.isnan(y_new_raw), 0.0, y_new_raw)
        u_row = U[new_idx, J_prev] if J_prev.size else np.empty(0)

        # ---- Schur bordering update for M_inv (always succeeds) ----
        if J_prev.size == 0:
            M_inv = np.array([[1.0 / (1.0 + a_new * L_diag)]])
            N_inv = np.array([[1.0]])
        else:
            L0_old_raw = L[np.ix_(J_prev, J_prev)]
            L0_old = np.where(np.isnan(L0_old_raw), 0.0, L0_old_raw)
            A0_old = A[np.ix_(J_prev, J_prev)]
            U0_old = U[np.ix_(J_prev, J_prev)]

            a_p = 1.0 + a_new * L_diag + float(alpha_A @ y_new)
            b_p = L0_old.T @ alpha_A
            c_p = alpha_A * L_diag + A0_old @ y_new
            p = M_inv @ c_p
            q = b_p @ M_inv
            s_M = a_p - float(b_p @ p)
            if abs(s_M) < tol:
                # rare; fall back to naive robust
                return backward_triangular_elimination_robust(A, Lambda, tol=tol)
            M_new = np.empty((J_size, J_size))
            M_new[0, 0] = 1.0 / s_M
            M_new[0, 1:] = -q / s_M
            M_new[1:, 0] = -p / s_M
            M_new[1:, 1:] = M_inv + np.outer(p, q) / s_M
            M_inv = M_new

            # ---- Schur bordering update for N_inv (only while K_0 non-singular) ----
            if N_inv_valid:
                a_pp = 1.0 + float(u_row @ alpha_A)
                b_pp = A0_old @ u_row
                c_pp = U0_old @ alpha_A
                p2 = N_inv @ c_pp
                q2 = b_pp @ N_inv
                s_N = a_pp - float(b_pp @ p2)
                if abs(s_N) < tol:
                    # K_0 has just become singular (s_N = pi_{new_idx} which
                    # was 0 at that step). From here on, use lstsq for dual.
                    N_inv_valid = False
                else:
                    N_new = np.empty((J_size, J_size))
                    N_new[0, 0] = 1.0 / s_N
                    N_new[0, 1:] = -q2 / s_N
                    N_new[1:, 0] = -p2 / s_N
                    N_new[1:, 1:] = N_inv + np.outer(p2, q2) / s_N
                    N_inv = N_new

        alpha = A[J, k]
        beta = Lambda[J, k]
        lam = float(Lambda[k, k])
        A0 = A[np.ix_(J, J)]
        U0 = U[np.ix_(J, J)]

        # ---- Build V, E from bad_cols (all have index > k, so all in J) ----
        r = len(bad_cols)
        if r > 0:
            V_embed = np.zeros((J_size, r))
            E_mat = np.zeros((J_size, r))
            for i, b in enumerate(bad_cols):
                j = b['index']
                pos = j - (k + 1)
                v_full = b['direction']
                end = min(pos + len(v_full), J_size)
                V_embed[pos:end, i] = v_full[: end - pos]
                E_mat[pos, i] = 1.0
            A_prime = A0 @ V_embed

        # ---- Primal SM correction on u (uses cached M_inv) ----
        # M_inv stores (I + A_0 L_finite)^{-1}; M^T inverse acts via M_inv.T
        u_base = M_inv.T @ beta
        if r > 0:
            MtinvE = M_inv.T @ E_mat
            denom = A_prime.T @ MtinvE
            try:
                lhs = np.linalg.solve(denom, A_prime.T @ u_base)
            except np.linalg.LinAlgError:
                lhs, *_ = np.linalg.lstsq(denom, A_prime.T @ u_base, rcond=None)
            u = u_base - MtinvE @ lhs
        else:
            u = u_base
        U[k, J] = u

        # ---- Dual: compute s for pi_k and L_kk ----
        A0u = A0 @ u
        if r > 0 or not N_inv_valid:
            # Augmented system from scratch (O(|J|^3) per step)
            K0 = np.eye(J_size) + U0 @ A0
            if r > 0:
                aug = np.vstack([K0.T, V_embed.T])
                rhs = np.concatenate([A0u, np.zeros(r)])
            else:
                aug = K0.T
                rhs = A0u
            s, *_ = np.linalg.lstsq(aug, rhs, rcond=None)
        else:
            s = N_inv.T @ A0u

        U0_alpha = U0 @ alpha
        pi_k = 1.0 + float(u @ alpha) - float(s @ U0_alpha)

        if abs(pi_k) < tol:
            # Bad pivot: compute w (constrained) and append (k, v_k) to bad_cols
            if r > 0:
                K0 = np.eye(J_size) + U0 @ A0
                aug = np.vstack([K0, V_embed.T])
                rhs_w = np.concatenate([U0_alpha, np.zeros(r)])
                w, *_ = np.linalg.lstsq(aug, rhs_w, rcond=None)
            elif N_inv_valid:
                w = N_inv @ U0_alpha
            else:
                K0 = np.eye(J_size) + U0 @ A0
                w, *_ = np.linalg.lstsq(K0, U0_alpha, rcond=None)
            v_k = np.concatenate([[1.0], -w])
            bad_cols.append({'index': k, 'pi_k': pi_k, 'direction': v_k})
            L[k, k] = np.nan
            L[J, k] = np.nan
            # Mark N_inv as invalid for future steps (K_0 will be singular)
            N_inv_valid = False
        else:
            L_kk = (lam - float(s @ beta)) / pi_k
            L[k, k] = L_kk
            if r == 0 and N_inv_valid:
                z = N_inv @ beta
                w = N_inv @ U0_alpha
            else:
                # Singular subproblem or N_inv invalidated: augmented
                # constrained solve picks the V-orthogonal z, w; these
                # are needed for subsequent steps' M_finite = I + A_0 L.
                K0_eff = np.eye(J_size) + U0 @ A0
                if r > 0:
                    aug = np.vstack([K0_eff, V_embed.T])
                    rhs_z = np.concatenate([beta, np.zeros(r)])
                    rhs_w = np.concatenate([U0_alpha, np.zeros(r)])
                else:
                    aug = K0_eff
                    rhs_z = beta
                    rhs_w = U0_alpha
                z, *_ = np.linalg.lstsq(aug, rhs_z, rcond=None)
                w, *_ = np.linalg.lstsq(aug, rhs_w, rcond=None)
            L[J, k] = z - w * L_kk

    info = {
        'bad_pivots': [(b['index'], b['pi_k']) for b in bad_cols],
        'num_bad': len(bad_cols),
        'status': 'ok' if not bad_cols else 'has_bad_pivots',
    }
    return U, L, info


# -----------------------------------------------------------------------------
# Inverse-triangular variant
# -----------------------------------------------------------------------------

def triangular_inverse_decomposition(
    A: np.ndarray,
    Lambda: np.ndarray,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (U, S) where U is the unique strictly upper triangular matrix
    such that

        S = Lambda - U - U^T - U A U^T   is positive definite,
        S^{-1} (I + U A)                 is lower triangular.

    Always exists for positive definite A, Lambda (Proposition 1 of
    Dolinsky-Zuk 2026).

    Strategy: try the direct backward elimination first. If it succeeds,
    extract U from the bilinear decomposition (the U is the same matrix).
    If the elimination hits a zero pivot, fall back to the robust variant
    that handles the entire SPD cone in finite arithmetic.
    """
    U, L, info = backward_triangular_elimination(A, Lambda, tol=tol)
    if info.status == "ok":
        S = Lambda - U - U.T - U @ A @ U.T
        return U, S

    # Fall back to robust variant (always returns U for SPD pairs).
    U, _, _ = backward_triangular_elimination_robust(A, Lambda)
    S = Lambda - U - U.T - U @ A @ U.T
    return U, S


def _triangular_inverse_via_newton(A: np.ndarray, Lambda: np.ndarray):
    """
    Wrap the general primal Newton solver around the 2n x 2n problem of
    Proposition 4.1 in [DZ:26]: decompose B^{-1} = Q^{-1} + C with C in the
    strictly-upper-triangular off-diagonal-block subspace S. Returns (U, S_U).

    The solver `constrained_decomposition(M, basis)` decomposes its INPUT
    matrix M as M = N^{-1} + C, so we feed it B^{-1}, not B. We use the
    closed form from the proof to avoid an extra inversion.
    """
    from constrained_decomposition_core import constrained_decomposition
    from constrained_decomposition_matrices import make_offdiag_pair_basis

    n = Lambda.shape[0]
    A_inv = np.linalg.inv(A)
    # B = [[Sigma, Sigma], [Sigma, Sigma + A]]  ==>
    # B^{-1} = [[Lambda + A^{-1}, -A^{-1}], [-A^{-1}, A^{-1}]]
    B_inv = np.block([[Lambda + A_inv, -A_inv], [-A_inv, A_inv]])

    pairs = [(i, n + j) for i in range(n) for j in range(i + 1, n)]
    basis = make_offdiag_pair_basis(2 * n, pairs)

    _, C, _ = constrained_decomposition(B_inv, basis, method="newton")
    U = np.triu(C[:n, n:], k=1)
    S = Lambda - U - U.T - U @ A @ U.T
    return U, S


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def bilinear_residual(A: np.ndarray, Lambda: np.ndarray,
                     U: np.ndarray, L: np.ndarray) -> float:
    """Max absolute entry of Lambda - (L + U + U A L)."""
    return float(np.max(np.abs(Lambda - (L + U + U @ A @ L))))


def is_strictly_upper(U: np.ndarray, tol: float = 1e-12) -> bool:
    return float(np.max(np.abs(np.tril(U)))) < tol


def is_lower(L: np.ndarray, tol: float = 1e-12) -> bool:
    return float(np.max(np.abs(np.triu(L, k=1)))) < tol


# -----------------------------------------------------------------------------
# Demos / sanity checks
# -----------------------------------------------------------------------------

def _demo_2x2():
    print("=" * 60)
    print("Demo 1: 2x2 closed-form check (paper Section 4.3)")
    print("=" * 60)
    A = np.array([[3.0, 1.0], [1.0, 4.0]])
    Lam = np.array([[2.0, 0.5], [0.5, 1.0]])
    U, L, info = backward_triangular_elimination(A, Lam)
    a, b, c = A[0, 0], A[0, 1], A[1, 1]
    p, q, r = Lam[0, 0], Lam[0, 1], Lam[1, 1]

    U01_expected = q / (1 + c * r)
    L00_expected = ((1 + c * r) * p - c * q * q) / (1 + c * r + b * q)

    assert info.status == "ok"
    assert np.isclose(U[0, 1], U01_expected)
    assert np.isclose(L[0, 0], L00_expected)
    assert np.isclose(L[1, 1], r)
    assert np.isclose(L[1, 0], q)
    print(f"  U[0,1]   = {U[0,1]:.6f}   (paper:  q/(1+cr)       = {U01_expected:.6f})")
    print(f"  L[0,0]   = {L[0,0]:.6f}   (paper:  ((1+cr)p-cq^2)/(1+cr+bq) = {L00_expected:.6f})")
    print(f"  residual = {bilinear_residual(A, Lam, U, L):.3e}")


def _demo_n1():
    print("\n" + "=" * 60)
    print("Demo 2: n=1 trivial case")
    print("=" * 60)
    A = np.array([[2.5]])
    Lam = np.array([[0.7]])
    U, L, info = backward_triangular_elimination(A, Lam)
    assert info.status == "ok"
    assert U.shape == (1, 1) and U[0, 0] == 0.0
    assert np.isclose(L[0, 0], Lam[0, 0])
    print(f"  U = {U}, L = {L}, residual = {bilinear_residual(A, Lam, U, L):.3e}")


def _demo_random(n: int = 12, seed: int = 0):
    print("\n" + "=" * 60)
    print(f"Demo 3: random SPD pair, n={n}")
    print("=" * 60)
    from constrained_decomposition_matrices import make_random_spd
    A = make_random_spd(n, seed=seed)
    Sigma = make_random_spd(n, seed=seed + 1)
    Lam = np.linalg.inv(Sigma)
    U, L, info = backward_triangular_elimination(A, Lam)
    print(f"  status = {info.status}")
    if info.status == "ok":
        print(f"  U strictly upper? {is_strictly_upper(U)}")
        print(f"  L lower?          {is_lower(L)}")
        print(f"  residual = {bilinear_residual(A, Lam, U, L):.3e}")
    else:
        print(f"  {info.detail}")


def _demo_fbm(n: int = 10, H: float = 0.7, lam: float = 0.5):
    print("\n" + "=" * 60)
    print(f"Demo 4: fBm increments + tridiagonal A, n={n}, H={H}, lambda={lam}")
    print("=" * 60)
    from constrained_decomposition_matrices import spd_fractional_BM
    Sigma = spd_fractional_BM(n, H=H, T=1.0, diff_flag=True)
    Lam = np.linalg.inv(Sigma)
    # Tridiagonal A from the restructured paper: 2*lam on diag, -lam off.
    A = 2.0 * lam * np.eye(n)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = -lam
    assert np.all(np.linalg.eigvalsh(A) > 0)
    U, L, info = backward_triangular_elimination(A, Lam)
    print(f"  status   = {info.status}")
    if info.status == "ok":
        print(f"  residual = {bilinear_residual(A, Lam, U, L):.3e}")
        print(f"  ||U||_F  = {np.linalg.norm(U):.4f}")
        print(f"  ||L||_F  = {np.linalg.norm(L):.4f}")


def _demo_singular():
    print("\n" + "=" * 60)
    print("Demo 5: 2x2 failure -- L = None but U still returned")
    print("=" * 60)
    # a=2, b=-1, c=1, p=5, q=2, r=1 so that 1 + c*r + b*q = 0.
    A = np.array([[2.0, -1.0], [-1.0, 1.0]])
    Lam = np.array([[5.0,  2.0], [ 2.0, 1.0]])
    assert np.linalg.eigvalsh(A).min() > 0
    assert np.linalg.eigvalsh(Lam).min() > 0
    U, L, info = backward_triangular_elimination(A, Lam)
    print(f"  status      = {info.status}")
    print(f"  failed_step = {info.failed_step}")
    print(f"  pivot       = {info.pivot}")
    print(f"  L is None?  = {L is None}")
    print(f"  U =\n{U}")
    # Verify U satisfies Prop 1: S_U SPD and S_U^{-1}(I + UA) lower triangular.
    S_U = Lam - U - U.T - U @ A @ U.T
    M = np.linalg.solve(S_U, np.eye(2) + U @ A)
    spd = np.all(np.linalg.eigvalsh(S_U) > 0)
    print(f"  S_U =\n{S_U}")
    print(f"  S_U is SPD:                {spd}")
    print(f"  S_U^-1 (I+UA) lower-tri:   {is_lower(M)}")
    print("  -> U is usable in the finance theorem (Theorem 1).")


def _demo_efficient_vs_basic(sizes=(20, 80, 200)):
    print("\n" + "=" * 60)
    print("Demo 6: O(n^4) basic vs O(n^3) efficient (random SPD pairs)")
    print("=" * 60)
    import time
    from constrained_decomposition_matrices import make_random_spd
    print(f"  {'n':>5}  {'basic [s]':>11}  {'fast [s]':>10}  {'speedup':>8}  {'max |U-U|':>11}  {'max |L-L|':>11}")
    for n in sizes:
        A = make_random_spd(n, seed=7)
        Lam = np.linalg.inv(make_random_spd(n, seed=8))
        t0 = time.perf_counter()
        U1, L1, info1 = backward_triangular_elimination(A, Lam)
        t_basic = time.perf_counter() - t0
        t0 = time.perf_counter()
        U2, L2, info2 = backward_triangular_elimination_efficient(A, Lam)
        t_fast = time.perf_counter() - t0
        assert info1.status == "ok" and info2.status == "ok"
        du = float(np.max(np.abs(U1 - U2)))
        dl = float(np.max(np.abs(L1 - L2)))
        speedup = t_basic / t_fast if t_fast > 0 else float("inf")
        print(f"  {n:>5}  {t_basic:>11.4f}  {t_fast:>10.4f}  {speedup:>8.1f}x  {du:>11.2e}  {dl:>11.2e}")


if __name__ == "__main__":
    _demo_2x2()
    _demo_n1()
    _demo_random(n=12, seed=0)
    _demo_fbm(n=10, H=0.7, lam=0.5)
    _demo_singular()
    _demo_efficient_vs_basic()
