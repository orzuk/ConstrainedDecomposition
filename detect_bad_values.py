#!/usr/bin/env python3
"""
Detect and optionally remove bad/suspicious values from results CSV.

Known relationships:
- Full >= Sum (more information = higher value)
- Full >= Markovian (more information = higher value)
- Markovian vs Sum: incomparable (sometimes one is bigger)

Bad values include:
1. Full < Sum (violates information ordering)
2. Full < Markovian (violates information ordering)
3. Sudden jumps: |val(h+0.01) - val(h)| >> typical differences
4. Isolated spikes: val(h) very different from neighbors but neighbors agree
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def detect_bad_values(results_file: str = "results/all_results.csv",
                      model: str = None,
                      n: int = None,
                      alpha: float = None,
                      h_min: float = 0.5,
                      jump_threshold: float = 5.0,
                      verbose: bool = True):
    """
    Detect rows with suspicious values.

    Parameters
    ----------
    jump_threshold : float
        Flag jumps that are this many times larger than median step size.
    h_min : float
        Only check H >= h_min.

    Returns DataFrame of bad rows.
    """
    df = pd.read_csv(results_file)

    # Convert value columns to numeric
    for col in ['value_sum', 'value_markovian', 'value_full']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Get unique (model, n, alpha) combinations
    if model is not None:
        models = [model]
    else:
        models = df['model'].unique()

    all_bad_rows = []

    for mod in models:
        mask_model = df['model'] == mod

        if mod == 'mixed_fbm':
            if alpha is not None:
                alphas = [alpha]
            else:
                alphas = df[mask_model]['alpha'].unique()
        else:
            alphas = [None]

        if n is not None:
            ns = [n]
        else:
            ns = sorted(df[mask_model]['n'].unique())

        for a in alphas:
            for n_val in ns:
                mask = mask_model & (df['n'] == n_val)
                if a is not None:
                    mask &= np.isclose(df['alpha'], a)

                subset = df[mask].copy()
                subset = subset[subset['H'] >= h_min].sort_values('H').reset_index()

                if len(subset) < 3:
                    continue

                bad_rows = check_subset(subset, n_val, a, mod, jump_threshold)
                all_bad_rows.extend(bad_rows)

    bad_df = pd.DataFrame(all_bad_rows)

    if verbose and len(bad_df) > 0:
        print(f"Found {len(bad_df)} suspicious values (H >= {h_min}):")
        print("=" * 120)
        for _, row in bad_df.iterrows():
            alpha_str = f", alpha={row['alpha']:.1f}" if pd.notna(row['alpha']) else ""
            print(f"  {row['model']} n={row['n']:4d}, H={row['H']:.2f}{alpha_str}: "
                  f"[{row['strategy']}] {row['issue']}")
            if 'values' in row and row['values']:
                print(f"      {row['values']}")
    elif verbose:
        print(f"No suspicious values detected for H >= {h_min}.")

    return bad_df


def check_subset(subset, n_val, alpha, model, jump_threshold):
    """Check a single (model, n, alpha) subset for anomalies."""
    bad_rows = []
    H_vals = subset['H'].values
    v_sum = subset['value_sum'].values
    v_mark = subset['value_markovian'].values
    v_full = subset['value_full'].values
    indices = subset['index'].values

    # Check ordering violations: Full >= Sum, Full >= Markovian
    for i in range(len(subset)):
        H = H_vals[i]
        vs, vm, vf = v_sum[i], v_mark[i], v_full[i]

        # Full < Sum (violation)
        if pd.notna(vf) and pd.notna(vs):
            if vf < vs - 0.01:  # Small tolerance
                bad_rows.append({
                    'index': indices[i], 'model': model, 'n': n_val, 'H': H,
                    'alpha': alpha, 'strategy': 'full',
                    'issue': f'Full < Sum: {vf:.4f} < {vs:.4f}',
                    'values': f'sum={vs:.4f}, mark={vm:.4f}, full={vf:.4f}'
                })

        # Full < Markovian (violation)
        if pd.notna(vf) and pd.notna(vm):
            if vf < vm - 0.01:
                bad_rows.append({
                    'index': indices[i], 'model': model, 'n': n_val, 'H': H,
                    'alpha': alpha, 'strategy': 'full',
                    'issue': f'Full < Markovian: {vf:.4f} < {vm:.4f}',
                    'values': f'sum={vs:.4f}, mark={vm:.4f}, full={vf:.4f}'
                })

    # Check for jumps and spikes in each strategy
    for col_name, col_vals, strat_name in [
        ('value_sum', v_sum, 'sum'),
        ('value_markovian', v_mark, 'markovian'),
        ('value_full', v_full, 'full')
    ]:
        # Compute differences
        valid_mask = ~np.isnan(col_vals)
        if np.sum(valid_mask) < 5:
            continue

        valid_H = H_vals[valid_mask]
        valid_vals = col_vals[valid_mask]
        valid_indices = indices[valid_mask]

        diffs = np.abs(np.diff(valid_vals))
        if len(diffs) < 3:
            continue

        median_diff = np.median(diffs)
        if median_diff < 1e-6:
            median_diff = 0.01  # Avoid division by zero

        # Detect sudden jumps
        for j in range(len(diffs)):
            if diffs[j] > jump_threshold * median_diff and diffs[j] > 0.05:
                # Could be either point j or j+1 that's bad
                # Check which one is the outlier by looking at context
                H1, H2 = valid_H[j], valid_H[j + 1]
                v1, v2 = valid_vals[j], valid_vals[j + 1]

                # Look at surrounding values to determine which is the outlier
                context_before = valid_vals[max(0, j-2):j]
                context_after = valid_vals[j+2:min(len(valid_vals), j+4)]

                if len(context_before) > 0 and len(context_after) > 0:
                    avg_before = np.mean(context_before)
                    avg_after = np.mean(context_after)

                    # If v1 is far from both contexts, it's likely the outlier
                    dist1 = min(abs(v1 - avg_before), abs(v1 - avg_after))
                    dist2 = min(abs(v2 - avg_before), abs(v2 - avg_after))

                    if dist1 > dist2 * 2:
                        bad_rows.append({
                            'index': valid_indices[j], 'model': model, 'n': n_val,
                            'H': H1, 'alpha': alpha, 'strategy': strat_name,
                            'issue': f'Sudden jump: val={v1:.4f}, next={v2:.4f}, diff={diffs[j]:.4f} (median={median_diff:.4f})',
                            'values': ''
                        })
                    elif dist2 > dist1 * 2:
                        bad_rows.append({
                            'index': valid_indices[j + 1], 'model': model, 'n': n_val,
                            'H': H2, 'alpha': alpha, 'strategy': strat_name,
                            'issue': f'Sudden jump: val={v2:.4f}, prev={v1:.4f}, diff={diffs[j]:.4f} (median={median_diff:.4f})',
                            'values': ''
                        })

        # Detect isolated spikes (val very different from both neighbors, but neighbors agree)
        for j in range(1, len(valid_vals) - 1):
            v_prev, v_curr, v_next = valid_vals[j-1], valid_vals[j], valid_vals[j+1]

            # Check if neighbors agree but current is different
            neighbor_diff = abs(v_prev - v_next)
            curr_diff = min(abs(v_curr - v_prev), abs(v_curr - v_next))

            if curr_diff > 5 * max(neighbor_diff, median_diff) and curr_diff > 0.05:
                bad_rows.append({
                    'index': valid_indices[j], 'model': model, 'n': n_val,
                    'H': valid_H[j], 'alpha': alpha, 'strategy': strat_name,
                    'issue': f'Isolated spike: val={v_curr:.4f}, neighbors={v_prev:.4f},{v_next:.4f}',
                    'values': ''
                })

    return bad_rows


def remove_bad_values(results_file: str = "results/all_results.csv",
                      bad_df: pd.DataFrame = None,
                      backup: bool = True):
    """
    Clear bad values from results CSV (set to empty string).
    Only clears the specific strategy column that was flagged.
    """
    if bad_df is None or len(bad_df) == 0:
        print("No bad values to remove.")
        return

    df = pd.read_csv(results_file)

    if backup:
        backup_file = results_file.replace('.csv', '_backup.csv')
        df.to_csv(backup_file, index=False)
        print(f"Backup saved to: {backup_file}")

    # Clear the specific column for each bad row
    strategy_to_col = {
        'sum': 'value_sum',
        'markovian': 'value_markovian',
        'full': 'value_full'
    }

    cleared = {'sum': 0, 'markovian': 0, 'full': 0}
    for _, row in bad_df.iterrows():
        idx = row['index']
        strat = row['strategy']
        col = strategy_to_col.get(strat)
        if col:
            df.loc[idx, col] = ''
            cleared[strat] += 1

    df.to_csv(results_file, index=False)
    print(f"Cleared values: sum={cleared['sum']}, markovian={cleared['markovian']}, full={cleared['full']}")
    print(f"Updated: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and remove bad values from results")
    parser.add_argument("--results", type=str, default="results/all_results.csv")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--h-min", type=float, default=0.5,
                       help="Only check H >= this value (default: 0.5)")
    parser.add_argument("--jump-threshold", type=float, default=5.0,
                       help="Flag jumps > this * median_diff (default: 5.0)")
    parser.add_argument("--remove", action="store_true",
                       help="Remove bad values (clear specific columns)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't create backup before removing")

    args = parser.parse_args()

    bad_df = detect_bad_values(
        results_file=args.results,
        model=args.model,
        n=args.n,
        alpha=args.alpha,
        h_min=args.h_min,
        jump_threshold=args.jump_threshold,
    )

    if args.remove and len(bad_df) > 0:
        print()
        remove_bad_values(
            results_file=args.results,
            bad_df=bad_df,
            backup=not args.no_backup,
        )
        print("\nRerun jobs with --incremental to recompute cleared values.")
