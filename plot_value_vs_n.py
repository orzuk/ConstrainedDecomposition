#!/usr/bin/env python3
"""
Plot investment value vs n (matrix size) for selected H values.

Creates figures with:
- X-axis: n
- Y-axis: value
- Different colors for different H values
- Different line styles for different strategies (sum, markovian, full)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for proper math rendering
from pathlib import Path


def plot_value_vs_n(
    results_file: str = "results/all_results.csv",
    model: str = "mixed_fbm",
    alpha: float = 5.0,
    H_values: list = None,
    output_dir: str = "figs",
    show_plot: bool = False,
    show_title: bool = False,
):
    """
    Plot value vs n for selected H values.

    Parameters
    ----------
    results_file : str
        Path to CSV with results.
    model : str
        Model type (mixed_fbm or fbm).
    alpha : float
        Alpha parameter (for mixed_fbm).
    H_values : list
        List of H values to plot. Default: [0.1, 0.3, 0.5, 0.7, 0.9]
    output_dir : str
        Directory for output figures.
    show_plot : bool
        If True, display plot interactively.
    """
    if H_values is None:
        H_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Load data
    df = pd.read_csv(results_file)

    # Filter by model and alpha
    mask = df['model'] == model
    if model == 'mixed_fbm':
        mask &= np.isclose(df['alpha'], alpha)

    df_filtered = df[mask].copy()

    if df_filtered.empty:
        print(f"No data found for model={model}, alpha={alpha}")
        return

    # Get available n values (only multiples of 100 for clean plots)
    n_values = sorted([n for n in df_filtered['n'].unique() if n % 100 == 0])
    print(f"Available n values: {n_values}")

    # Colors for H values: blue to red colormap
    cmap = plt.cm.coolwarm
    # Map H values to colormap positions (lowest H -> blue, highest H -> red)
    if len(H_values) > 1:
        colors = [cmap(i / (len(H_values) - 1)) for i in range(len(H_values))]
    else:
        colors = [cmap(0.5)]

    # Strategies to plot (one subplot each)
    strategies = [
        ('value_sum', 'Sum'),
        ('value_markovian', 'Markovian'),
        ('value_full', 'Full'),
    ]

    # Create figure with 3 subplots (one per strategy)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax_idx, (strategy_col, strategy_label) in enumerate(strategies):
        ax = axes[ax_idx]

        for h_idx, H in enumerate(H_values):
            # Get data for this H (with tolerance for float comparison)
            h_mask = np.isclose(df_filtered['H'], H, atol=0.005)
            df_h = df_filtered[h_mask]

            if df_h.empty:
                continue

            n_list = []
            val_list = []

            for n in n_values:
                df_n = df_h[df_h['n'] == n]
                if not df_n.empty:
                    val = df_n[strategy_col].iloc[0]
                    if pd.notna(val) and val != '':
                        try:
                            n_list.append(n)
                            val_list.append(float(val))
                        except (ValueError, TypeError):
                            pass

            if n_list:
                ax.plot(
                    n_list, val_list,
                    linestyle='-',
                    marker='o',
                    color=colors[h_idx],
                    label=f'H={H}' if ax_idx == 0 else None,
                    markersize=5,
                    linewidth=1.5,
                    alpha=0.8,
                )

        ax.set_xlabel(r'$n$', fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel(r'$v_N^*$', fontsize=12)
        ax.set_title(strategy_label, fontsize=12)
        ax.grid(True, alpha=0.3)

    # Add single legend for H values (on first subplot)
    from matplotlib.patches import Patch
    h_patches = [Patch(color=colors[i], label=f'H={H_values[i]}') for i in range(len(H_values))]
    axes[0].legend(handles=h_patches, loc='upper left', fontsize=8, framealpha=0.8)

    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h_str = "_".join([f"{h:.1f}" for h in H_values])
    alpha_str_file = f"_a{alpha}" if model == 'mixed_fbm' else ""
    out_file = output_dir / f"value_vs_n_{model}{alpha_str_file}_H_{h_str}.png"

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved figure to: {out_file}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_single_H_all_strategies(
    results_file: str = "results/all_results.csv",
    model: str = "mixed_fbm",
    alpha: float = 5.0,
    H: float = 0.5,
    output_dir: str = "figs",
    show_plot: bool = False,
    show_title: bool = False,
):
    """
    Plot value vs n for a single H value, showing all strategies clearly.
    """
    df = pd.read_csv(results_file)

    # Filter
    mask = (df['model'] == model) & np.isclose(df['H'], H, atol=0.005)
    if model == 'mixed_fbm':
        mask &= np.isclose(df['alpha'], alpha)

    df_h = df[mask].copy()

    if df_h.empty:
        print(f"No data for model={model}, alpha={alpha}, H={H}")
        return

    n_values = sorted(df_h['n'].unique())

    fig, ax = plt.subplots(figsize=(9, 6))

    strategies = [
        ('value_sum', 'Sum (upper bound)', 'C0', 's', ':'),
        ('value_markovian', 'Markovian', 'C1', 'o', '-'),
        ('value_full', 'Full information', 'C2', '^', '--'),
    ]

    for col, label, color, marker, ls in strategies:
        n_list = []
        val_list = []

        for n in n_values:
            df_n = df_h[df_h['n'] == n]
            if not df_n.empty:
                val = df_n[col].iloc[0]
                if pd.notna(val) and val != '':
                    try:
                        n_list.append(n)
                        val_list.append(float(val))
                    except (ValueError, TypeError):
                        pass

        if n_list:
            ax.plot(n_list, val_list, marker=marker, linestyle=ls, color=color,
                   label=label, markersize=8, linewidth=2)

    ax.set_xlabel(r'$n$', fontsize=14)
    ax.set_ylabel(r'$v_N^*$', fontsize=14)

    if show_title:
        alpha_str = f", α={alpha}" if model == 'mixed_fbm' else ""
        ax.set_title(f'Investment Value vs n (H={H}{alpha_str})', fontsize=14)

    ax.legend(loc='best', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_str_file = f"_a{alpha}" if model == 'mixed_fbm' else ""
    out_file = output_dir / f"value_vs_n_{model}{alpha_str_file}_H{H:.2f}.png"

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved figure to: {out_file}")

    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot investment value vs n")
    parser.add_argument("--results", type=str, default="results/all_results.csv",
                       help="Path to results CSV")
    parser.add_argument("--model", type=str, default="mixed_fbm",
                       choices=["mixed_fbm", "fbm"])
    parser.add_argument("--alpha", type=float, default=5.0,
                       help="Alpha parameter for mixed_fbm")
    parser.add_argument("--H", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9],
                       help="H values to plot (space-separated)")
    parser.add_argument("--single-H", type=float, default=None,
                       help="Plot single H value with all strategies (cleaner)")
    parser.add_argument("--output-dir", type=str, default="figs",
                       help="Output directory for figures")
    parser.add_argument("--show", action="store_true",
                       help="Show plot interactively")
    parser.add_argument("--show-title", action="store_true",
                       help="Show title on plots (default: no title)")
    parser.add_argument("--all-alpha", action="store_true",
                       help="Generate plots for all alpha values")

    args = parser.parse_args()

    if args.all_alpha and args.model == "mixed_fbm":
        alphas = [0.2, 1.0, 5.0, 10.0]
    else:
        alphas = [args.alpha]

    for alpha in alphas:
        print(f"\n=== Alpha = {alpha} ===")

        if args.single_H is not None:
            plot_single_H_all_strategies(
                results_file=args.results,
                model=args.model,
                alpha=alpha,
                H=args.single_H,
                output_dir=args.output_dir,
                show_plot=args.show,
                show_title=args.show_title,
            )
        else:
            plot_value_vs_n(
                results_file=args.results,
                model=args.model,
                alpha=alpha,
                H_values=args.H,
                output_dir=args.output_dir,
                show_plot=args.show,
                show_title=args.show_title,
            )
