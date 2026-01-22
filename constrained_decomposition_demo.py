"""
Command-line demo runner for constrained decomposition examples.

Run:
  python constrained_decomposition_demo.py --all --verbose
"""

import argparse
import os
import time
import numpy as np

from constrained_decomposition_core import *
from constrained_decomposition_matrices import *

from constrained_decomposition_viz import plot_decomposition_heatmaps, plot_block_decomposition


def time_solve(solve_fn, A, basis, **kwargs):
    t0 = time.perf_counter()
    sol = solve_fn(A, basis, **kwargs)
    t1 = time.perf_counter()
    return sol, (t1 - t0)


def solve_primal(A, basis, log_prefix="", **kwargs):
    verbose_local = kwargs.pop("verbose", globals().get("verbose", False))
    group = kwargs.pop("group", None)

    # Pop common args so they aren't passed twice via **kwargs
    method = kwargs.pop("method", "newton")
    tol = kwargs.pop("tol", 1e-8)
    max_iter = kwargs.pop("max_iter", 500)

    if group is None:
        B, C, x, info = constrained_decomposition(
            A=A,
            basis=basis,
            method=method,
            tol=tol,
            max_iter=max_iter,
            return_info=True,
            verbose=verbose_local,
            log_prefix=log_prefix,
            **kwargs,
        )
        return {"B": B, "C": C, "x": x, "solver": "primal", "info": info}

    # group-reduced solve: reduce basis only, keep A (theorem regime)
    B, C, x, basis_G, info = constrained_decomposition_group_invariant(
        A=A,
        basis=basis,
        group=group,
        solver="primal",
        method=method,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose_local,
        return_info=True,
        log_prefix=log_prefix,
        enforce_A_fixed=True,        # theorem mode: A must be G-fixed
        project_A_if_needed=False,   # do NOT silently change A in demo
        invariant_tol=1e-10,
        **kwargs,
    )
    return {"B": B, "C": C, "x": x, "solver": "primal-group", "info": info, "basis_G": basis_G}


def solve_dual(A, basis, basis_perp=None, log_prefix="", **kwargs):
    verbose_local = kwargs.pop("verbose", globals().get("verbose", False))
    tol = kwargs.pop("tol", 1e-8)
    max_iter = kwargs.pop("max_iter", 300)

    B, C, y, basis_perp_out, info = constrained_decomposition_dual(
        A=A,
        basis=basis,
        basis_perp=basis_perp,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose_local,
        return_info=True,
        log_prefix=log_prefix,
        **kwargs,
    )
    return {"B": B, "C": C, "x": y, "solver": "dual", "basis_perp": basis_perp_out, "info": info}


def solve_block_direct(A, basis, log_prefix="", **kwargs):
    """
    [DEPRECATED] Use solve_block_efficient instead.
    """
    verbose_local = kwargs.pop("verbose", globals().get("verbose", False))
    tol = kwargs.pop("tol", 1e-6)
    max_iter = kwargs.pop("max_iter", 200)
    blocks = kwargs.pop("blocks", None)
    free_pairs = kwargs.pop("free_pairs", None)

    if blocks is None:
        raise ValueError("solve_block_direct requires 'blocks' in kwargs")

    B, C, x, info = constrained_decomposition_block_direct(
        A=A,
        blocks=blocks,
        free_pairs=free_pairs,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose_local,
        return_info=True,
        log_prefix=log_prefix,
    )
    return {"B": B, "C": C, "x": x, "solver": "block-direct", "info": info}


def solve_block_efficient(A, basis, log_prefix="", **kwargs):
    """
    Efficient block-level solver for Demo III.

    Works entirely in block-parameter space with O(r³) per iteration complexity.
    Never builds O(n²) entry-level basis matrices.

    Required kwargs:
        blocks: list of index arrays (partition into r blocks)
        active_blocks: list of (i, j) tuples where C can be nonzero
    """
    verbose_local = kwargs.pop("verbose", globals().get("verbose", False))
    tol = kwargs.pop("tol", 1e-8)
    max_iter = kwargs.pop("max_iter", 200)

    # Extract blocks and active_blocks from kwargs
    blocks = kwargs.pop("blocks", None)
    active_blocks = kwargs.pop("active_blocks", None)

    if blocks is None:
        raise ValueError("solve_block_efficient requires 'blocks' in kwargs")

    B, C, x, info = constrained_decomposition_block_efficient(
        A=A,
        blocks=blocks,
        active_blocks=active_blocks,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose_local,
        return_info=True,
        log_prefix=log_prefix,
    )
    return {"B": B, "C": C, "x": x, "solver": "block-efficient", "info": info}



if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # ============================================================
    # PyCharm defaults (used when no CLI args override them)
    # ============================================================
    DEFAULT_RUN = {
        "demo1_primal_smallS": True,
        "demo2_dual_antibanded": True,
        "demo3_block_group": True,
        "demo4_banded_A_newton": True,
    }
    DEFAULT_OUTDIR = "demo_outputs"
    DEFAULT_VERBOSE = True

    # ============================================================
    # CLI parsing (overrides defaults when provided)
    # ============================================================
    parser = argparse.ArgumentParser(
        description="Constrained decomposition demos: primal / dual / group-invariant / banded."
    )
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save plots.")
    parser.add_argument("--verbose", action="store_true", help="Verbose solver output.")
    parser.add_argument("--all", action="store_true", help="Run all demos.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Comma-separated list of demos to run. Options: "
             "demo1_primal_smallS,demo2_dual_antibanded,demo3_block_group,"
             "demo4_banded_A_newton"
    )

    # demo sizes
    parser.add_argument("--n1", type=int, default=40, help="n for demo1 (general A, small S, primal).")
    parser.add_argument("--n2", type=int, default=120, help="n for demo2 (dual on banded S^perp).")
    parser.add_argument("--n3", type=int, default=50, help="n for demo3 (block-permutation group).")
    parser.add_argument("--n4", type=int, default=300, help="n for demo4/5 (banded A + banded S).")

    # parameters
    parser.add_argument("--blocks", type=int, default=5, help="Number of blocks r for demo3.")
    parser.add_argument("--bandwidth4", type=int, default=2, help="Half-bandwidth b for demo4/5 banded A and S.")
    parser.add_argument("--bandwidth2", type=int, default=1, help="Half-bandwidth b for demo2 S^perp (b=1 => tridiag).")

    args = parser.parse_args()

    # Decide which demos to run
    run_flags = dict(DEFAULT_RUN)
    if args.all:
        for k in run_flags:
            run_flags[k] = True
    if args.run is not None:
        for k in run_flags:
            run_flags[k] = False
        chosen = [s.strip() for s in args.run.split(",") if s.strip()]
        for name in chosen:
            if name not in run_flags:
                raise ValueError(f"Unknown demo '{name}'. Valid: {list(run_flags.keys())}")
            run_flags[name] = True

    outdir = args.outdir if args.outdir is not None else DEFAULT_OUTDIR
    os.makedirs(outdir, exist_ok=True)

    # IMPORTANT: solve_primal / solve_dual use the GLOBAL variable `verbose`
    globals()["verbose"] = bool(args.verbose or DEFAULT_VERBOSE)

    # ============================================================
    # Cache demo3 build (for block-efficient solver)
    # ============================================================
    _demo3_cache = {"built": False, "A": None, "blocks": None, "active_blocks": None}


    def build_demo3_illustrative_A(blocks, seed=42):
        """
        Build a block-constant SPD matrix A for Demo III that produces
        a visually clear heatmap:
        - Diagonal entries comparable to off-diagonal (not dominating)
        - Strong cross-block values where C will be INACTIVE (so B^{-1} must cover them)
        - Moderate cross-block values where C will be ACTIVE
        - Clear block structure visible in heatmap

        For 5 blocks with "band-1" active pattern:
        - Active cross-blocks: (0,1), (1,2), (2,3), (3,4) - adjacent pairs
        - Inactive cross-blocks: (0,2), (0,3), (0,4), (1,3), (1,4), (2,4)

        The values are scaled based on block sizes to ensure SPD for any n.
        """
        rng = np.random.default_rng(seed)
        n = sum(len(blk) for blk in blocks)
        r = len(blocks)
        block_sizes = [len(blk) for blk in blocks]
        max_block_size = max(block_sizes)

        A = np.zeros((n, n), dtype=float)

        # Scale values based on block sizes to ensure diagonal dominance (SPD)
        # For row i in block bi:
        #   off-diag sum = (block_sizes[bi] - 1) * within_val + sum_{bj != bi} block_sizes[bj] * cross_val
        # We need diagonal > off-diag sum with margin

        # Use relative values that scale with n
        # Target: within_val and cross_val should be small relative to diagonal
        total_other_blocks = n - max_block_size  # max cross-block contribution
        within_contrib = max_block_size - 1  # max within-block contribution

        # Set values so off-diagonal sum is ~50% of diagonal (visually clear but SPD)
        base_diag = 1.0
        within_val = 0.1 * base_diag / max(within_contrib, 1)  # ~10% from within-block
        cross_val = 0.3 * base_diag / max(total_other_blocks, 1)  # ~30% from cross-blocks

        # Varied diagonal per block for visual interest
        diag_multipliers = [1.0, 1.2, 1.0, 1.2, 1.0]

        # Within-block off-diagonal
        for bi, blk in enumerate(blocks):
            for i in blk:
                for j in blk:
                    if i != j:
                        A[i, j] = within_val

        # Cross-block values: ALL cross-blocks have the SAME value
        for bi in range(r):
            for bj in range(bi + 1, r):
                for i in blocks[bi]:
                    for j in blocks[bj]:
                        A[i, j] = cross_val
                        A[j, i] = cross_val

        # Set diagonal with variation and margin for SPD
        for bi, blk in enumerate(blocks):
            mult = diag_multipliers[bi] if bi < len(diag_multipliers) else 1.0
            for i in blk:
                A[i, i] = base_diag * mult

        return A


    def build_demo3_once():
        if not _demo3_cache["built"]:
            # Use 5 blocks with equal sizes for clean visualization
            r = 5
            block_size = args.n3 // r
            blocks3 = [list(range(i * block_size, (i + 1) * block_size)) for i in range(r)]
            # Handle remainder
            remainder = args.n3 - r * block_size
            if remainder > 0:
                blocks3[-1].extend(range(r * block_size, args.n3))

            # Build illustrative A
            A3 = build_demo3_illustrative_A(blocks3, seed=42)

            # Active pattern: most blocks active, only 2 corners inactive
            # - Diagonal blocks (i, i): all active
            # - Most cross-blocks: active
            # - Only corners (0, r-1) and (1, r-1) are INACTIVE
            # Fewer inactive blocks -> stronger contrast in B^{-1}
            active_diag = [(i, i) for i in range(r)]
            all_cross = [(i, j) for i in range(r) for j in range(i + 1, r)]
            inactive_cross = [(0, r - 1), (1, r - 1)]  # Just 2 corners inactive
            active_cross = [p for p in all_cross if p not in inactive_cross]
            active_blocks = active_diag + active_cross

            _demo3_cache["A"] = A3
            _demo3_cache["blocks"] = blocks3
            _demo3_cache["active_blocks"] = active_blocks
            _demo3_cache["built"] = True

        return _demo3_cache["A"], _demo3_cache["blocks"], _demo3_cache["active_blocks"]


    def build_demo3_efficient():
        """Build for efficient block-level solver (O(r³) per iteration)."""
        A3, blocks3, active_blocks = build_demo3_once()
        # No basis needed - solver works directly with block parameters
        return A3, None, {"blocks": blocks3, "active_blocks": active_blocks}


    # ============================================================
    # Cache demo4/5 build (same A,basis for fair Newton vs quasi compare)
    # ============================================================
    _demo45_cache = {"built": False, "A": None, "basis": None}
    def build_demo45_once():
        if not _demo45_cache["built"]:
            A4 = make_banded_spd(args.n4, b=args.bandwidth4, seed=3, diag_boost=8.0)
            basis4 = make_banded_basis_coo(args.n4, b=args.bandwidth4, include_diag=False)
            _demo45_cache["A"] = A4
            _demo45_cache["basis"] = basis4
            _demo45_cache["built"] = True
        return _demo45_cache["A"], _demo45_cache["basis"]

    # ============================================================
    # Demo specs
    # Each build returns (A, basis, solver_kwargs)
    # ============================================================
    demos = [
        {
            "name": "demo1_primal_smallS",
            "title": "Ex. I: General A + small S (primal)",
            "paper_example": "I",
            "A_type": "dense_random_spd",
            "S_type": "small explicit constraints",
            "solver": "primal-newton",
            "complexity": "O(n³ + Km²)",
            "complexity_long": "O(n³) per iteration (Cholesky) + O(m²) per Newton step",
            "get_dims": lambda n, basis, kwargs: {"n": n, "m": getattr(basis, 'm', '?')},
            "build": lambda: (
                make_random_spd(args.n1, seed=0),
                make_offdiag_pair_basis(args.n1, pairs=[(0, 1), (1, 2), (2, 3), (0, 3), (5, 7)]),
                {"method": "newton"},
            ),
            "solve": solve_primal,
            "plot_file": "demo1_primal_smallS.png",
        },
        {
            "name": "demo2_dual_antibanded",
            "title": "Ex. II: General A + large S, small S⊥ (dual)",
            "paper_example": "II",
            "A_type": "dense_random_spd",
            "S_type": "anti-banded S (implicit); S⊥ banded",
            "solver": "dual-newton",
            "complexity": "O(n³ + K(m⊥)²)",
            "complexity_long": "O(n³) per iteration (Cholesky) + O((m⊥)²) per Newton step",
            "get_dims": lambda n, basis, kwargs: {"n": n, "m⊥": getattr(kwargs.get('basis_perp'), 'm', '?')},
            "build": lambda: (
                make_random_spd(args.n2, seed=1),
                None,  # huge S not materialized
                {"basis_perp": make_banded_basis(args.n2, b=args.bandwidth2, include_diag=True)},
            ),
            "solve": solve_dual,
            "plot_file": "demo2_dual_antibanded.png",
        },
        {
            "name": "demo3_block_group",
            "title": "Ex. III: Block-permutation invariant (block-efficient)",
            "paper_example": "III",
            "A_type": "block-exchangeable (r blocks)",
            "S_type": "G-invariant block-support subspace",
            "solver": "block-efficient (O(r³) per iter)",
            "complexity": "O(Kr³)",
            "complexity_long": "O(r³) per iteration: works entirely in block-parameter space, never builds O(n²) basis",
            "get_dims": lambda n, basis, kwargs: {
                "n": n,
                "r": len(kwargs.get('blocks', [])),
                "m_G": len(kwargs.get('active_blocks', [])),
            },
            "build": build_demo3_efficient,
            "solve": solve_block_efficient,
            "plot_file": "demo3_block_group.png",
        },
        {
            "name": "demo4_banded_A_newton",
            "title": "Ex. IV: Banded A + banded S (Newton-CG)",
            "paper_example": "IV",
            "A_type": "banded_spd (bandwidth b)",
            "S_type": "banded constraints (bandwidth b)",
            "solver": "primal-newton-cg",
            "complexity": "O(Knb²)",
            "complexity_long": "O(nb²) per iteration: banded Cholesky O(nb²), CG with banded Hessian-vector products",
            "get_dims": lambda n, basis, kwargs: {"n": n, "m": getattr(basis, 'm', '?'), "b": args.bandwidth4},
            "build": lambda: (
                build_demo45_once()[0],
                build_demo45_once()[1],
                {
                    "method": "newton-cg",
                    "cholesky_backend": CholeskyBackend("banded", bandwidth=args.bandwidth4),
                },
            ),
            "solve": solve_primal,
            "plot_file": "demo4_banded_A_newton.png",
        },
    ]

    # ============================================================
    # Run loop (ONE solve call site + ONE plot call site)
    # ============================================================
    any_ran = False
    results_summary = []  # Collect results for summary table

    for spec in demos:
        if not run_flags.get(spec["name"], False):
            continue
        any_ran = True

        A, basis, solver_kwargs = spec["build"]()
        n = A.shape[0]

        # Get dimension parameters
        dims = {}
        if "get_dims" in spec:
            dims = spec["get_dims"](n, basis, solver_kwargs)

        # --- print header BEFORE solve so verbose output is contextual
        print("\n" + "=" * 72)
        print(f"[{spec['name']}] {spec['title']}")
        print(f"A_type: {spec.get('A_type', '?')}")
        print(f"S_type: {spec.get('S_type', '?')}")
        print(f"Solver: {spec.get('solver', '?')}")
        print(f"Complexity: {spec.get('complexity', '?')}")
        print(f"Dimensions: {dims}")

        # pass a prefix so iteration lines include the demo name
        solver_kwargs_run = dict(solver_kwargs)
        solver_kwargs_run["log_prefix"] = f"[{spec['name']}] "

        sol, elapsed = time_solve(spec["solve"], A, basis, **solver_kwargs_run)

        B, C = sol["B"], sol["C"]

        # reconstruction diagnostic
        recon = np.linalg.norm(A - (spd_inverse(B) + C), ord="fro")

        # choose basis for trace check and plotting:
        # - if primal: basis is the constraint basis
        # - if dual with basis=None: basis_perp exists in sol and is usable for trace/plotting
        basis_for_check = basis if basis is not None else sol.get("basis_perp", None)
        if basis_for_check is not None and hasattr(basis_for_check, "trace_with"):
            if basis is not None:
                # primal: constraints are tr(B Dk)=0
                max_trace = float(np.max(np.abs(basis_for_check.trace_with(B))))
                max_trace_label = "max|tr(BDk)|"
                residual_on = "B"
            else:
                # dual: constraints/stationarity are tr((A-B^{-1}) Ek)=0, i.e. tr(C Ek)=0
                max_trace = float(np.max(np.abs(basis_for_check.trace_with(C))))
                max_trace_label = "max|tr(CEk)|"
                residual_on = "C"
        else:
            max_trace = float("nan")
            max_trace_label = "max|trace|"
            residual_on = "B"

        info = sol.get("info", {})
        iters = info.get("iters", "?")
        backtracks = info.get("backtracks", "?")
        # Different solvers return different gradient/residual keys:
        # - primal constrained_decomposition: final_max_abs_trace (max |tr(B Dk)|)
        # - dual constrained_decomposition_dual: final_max_abs_grad
        # - blocked_newton: final_grad_norm
        grad_norm = info.get("final_max_abs_trace",
                            info.get("final_max_abs_grad",
                            info.get("final_grad_norm", None)))

        # Print detailed results
        print("\n" + "-" * 72)
        print(f"RESULTS for {spec['title']} (Example {spec.get('paper_example', '?')}):")
        print(f"  n = {n}")
        for k, v in dims.items():
            if k != 'n':
                print(f"  {k} = {v}")
        print(f"  Solver: {spec.get('solver', '?')}")
        print(f"  Complexity: {spec.get('complexity', '?')}")
        print(f"  Time: {elapsed:.4f} s")
        print(f"  Iterations: {iters}")
        print(f"  Backtracks: {backtracks}")
        if grad_norm is not None:
            print(f"  ‖∇Φ‖₂: {grad_norm:.3e}")
        print(f"  Reconstruction ‖A-(B⁻¹+C)‖_F: {recon:.3e}")
        print(f"  {max_trace_label}: {max_trace:.3e}")

        # Collect for summary
        results_summary.append({
            "example": spec.get("paper_example", "?"),
            "name": spec["name"],
            "n": n,
            "dims": dims,
            "solver": spec.get("solver", "?"),
            "complexity": spec.get("complexity", "?"),
            "time": elapsed,
            "iters": iters,
            "grad_norm": grad_norm,
            "recon": recon,
            "max_trace": max_trace,
            "info": info,  # Include solver info for m_G extraction
        })

        # Plot (use specialized plotting for demo3, generic for others)
        # Include n in filename so different runs don't overwrite each other
        base, ext = os.path.splitext(spec["plot_file"])
        plot_path = os.path.join(outdir, f"{base}_n{n}{ext}")
        if spec["name"] == "demo3_block_group":
            # Use specialized block visualization with boundaries and active/inactive highlighting
            plot_block_decomposition(
                A, B, C,
                blocks=solver_kwargs.get("blocks", []),
                active_blocks=solver_kwargs.get("active_blocks", None),
                out_file=plot_path,
                figsize=(10, 8),
                show_title=False
            )
        else:
            plot_decomposition_heatmaps(
                A, B, C, basis_for_check,
                out_file=plot_path,
                add_title=True,
                residual_on=residual_on
            )
        print(f"  Plot saved: {plot_path}")

    # ============================================================
    # Print summary table for paper (Table 2)
    # ============================================================
    if any_ran and results_summary:
        print("\n" + "=" * 80)
        print("SUMMARY TABLE FOR PAPER (Table 2)")
        print("=" * 80)
        print(f"{'Ex.':<5} {'Solver':<25} {'Complexity':<18} {'n':<6} {'K':<5} {'Time(s)':<10} {'‖∇Φ‖':<10}")
        print("-" * 80)
        for r in results_summary:
            iters_str = str(r['iters']) if r['iters'] != '?' else '?'
            grad_str = f"{r['grad_norm']:.2e}" if r['grad_norm'] is not None else "N/A"
            print(f"{r['example']:<5} {r['solver']:<25} {r['complexity']:<18} {r['n']:<6} {iters_str:<5} {r['time']:<10.4f} {grad_str:<10}")
        print("-" * 80)
        print("\nHeatmap figures saved in:", outdir)

        # ============================================================
        # Print LaTeX table rows for paper (Empirical section only)
        # Format: Examples as columns (I, II, III, IV)
        # ============================================================
        print("\n" + "=" * 80)
        print("LATEX TABLE - EMPIRICAL SECTION (copy & paste to paper)")
        print("Replace the --- placeholders in the Empirical section")
        print("=" * 80)

        # Build a dict keyed by example number for easy column access
        by_ex = {r['example']: r for r in results_summary}

        def fmt_grad(g):
            if g is None:
                return "---"
            exp = int(f"{g:.0e}".split('e')[1])
            coef = g / (10 ** exp)
            return f"${coef:.1f} \\times 10^{{{exp}}}$"

        def fmt_recon(r):
            if r is None:
                return "---"
            exp = int(f"{r:.0e}".split('e')[1])
            coef = r / (10 ** exp)
            return f"${coef:.1f} \\times 10^{{{exp}}}$"

        # Instance size row
        row_n = "Instance size &\n"
        row_n += " &\n".join([f"$n={by_ex.get(ex, {}).get('n', '---')}$" if ex in by_ex else "$n=$---"
                              for ex in ['I', 'II', 'III', 'IV']])
        row_n += " \\\\"

        # Constraint dim row (different format per example)
        dims_list = []
        for ex in ['I', 'II', 'III', 'IV']:
            if ex not in by_ex:
                dims_list.append("---")
            else:
                d = by_ex[ex]['dims']
                info_ex = by_ex[ex].get('info', {})
                if ex == 'I':
                    dims_list.append(f"$m={d.get('m', '?')}$")
                elif ex == 'II':
                    dims_list.append(f"$m^\\perp={d.get('m⊥', '?')}$")
                elif ex == 'III':
                    # For group-reduced, m_G is in solver info
                    m_G = info_ex.get('m_G', d.get('m_G', d.get('m', '?')))
                    dims_list.append(f"$m_G={m_G}$")
                elif ex == 'IV':
                    dims_list.append(f"$m={d.get('m', '?')}$,\\ $b={d.get('b', '?')}$")
        row_dim = "Constraint dim. &\n" + " &\n".join(dims_list) + " \\\\"

        # Newton iters row
        row_iters = "Newton iters &\n"
        row_iters += " & ".join([str(by_ex.get(ex, {}).get('iters', '---')) for ex in ['I', 'II', 'III', 'IV']])
        row_iters += " \\\\"

        # Time row
        row_time = "Time (s) &\n"
        row_time += " & ".join([f"{by_ex[ex]['time']:.3f}" if ex in by_ex else "---" for ex in ['I', 'II', 'III', 'IV']])
        row_time += " \\\\"

        # Gradient norm row
        row_grad = "Final $\\|\\nabla\\Phi\\|_2$ &\n"
        row_grad += " &\n".join([fmt_grad(by_ex[ex].get('grad_norm')) if ex in by_ex else "---" for ex in ['I', 'II', 'III', 'IV']])
        row_grad += " \\\\"

        # Reconstruction error row
        row_recon = "$\\|A-(B^{-1}+C)\\|_F$ &\n"
        row_recon += " &\n".join([fmt_recon(by_ex[ex].get('recon')) if ex in by_ex else "---" for ex in ['I', 'II', 'III', 'IV']])
        row_recon += " \\\\"

        print(row_n)
        print(row_dim)
        print(row_iters)
        print(row_time)
        print(row_grad)
        print(row_recon)

        # Save LaTeX table to file
        latex_content = "\n".join([row_n, row_dim, row_iters, row_time, row_grad, row_recon])
        # Create filename with n values
        n_values = f"n{args.n1}_{args.n2}_{args.n3}_{args.n4}"
        latex_file = os.path.join(outdir, f"latex_table_{n_values}.tex")
        with open(latex_file, "w") as f:
            f.write("% LaTeX table rows for constrained decomposition demo\n")
            f.write(f"% Parameters: n1={args.n1}, n2={args.n2}, n3={args.n3}, n4={args.n4}\n")
            f.write("% Copy these rows into your table environment\n\n")
            f.write(latex_content)
            f.write("\n")
        print(f"\nLaTeX table saved to: {latex_file}")

    if not any_ran:
        print("Nothing selected. Use --all or --run demo1_primal_smallS,...")
