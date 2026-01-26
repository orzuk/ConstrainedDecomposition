# Constrained Decomposition

Algorithms for constrained SPD matrix decomposition: given $A \succ 0$ and linear subspace $S$, find

$$A = B^{-1} + C, \quad B \in S^\perp, \quad C \in S$$

## Features

- **Primal Newton solver** for small constraint subspace $S$
- **Dual Newton solver** for small orthogonal complement $S^\perp$
- **Group-invariant reduction** for symmetric problems
- **Newton-CG with banded structure** for $O(nb^2)$ complexity
- **Finance application**: exponential utility maximization for mixed fBM

## Installation

```bash
# Clone the repository
git clone https://github.com/orzuk/ConstrainedDecomposition.git
cd ConstrainedDecomposition

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, NumPy, SciPy, Matplotlib, Pandas

## Quick Start

```bash
# Run demo with all 4 algebraic examples
python constrained_decomposition_demo.py --all --verbose

# Run finance example (mixed fBM, single H value)
python finance_example.py --model mixed_fbm --n 200 --hmin 0.7 --hmax 0.71 --hres 0.01 --alpha 5.0 --strategy both
```

## Reproducing Paper Results

This repository includes all code needed to reproduce the figures and tables in the paper.

### Figure 1: Mixed fBM Investment Value (3 strategies)

Computes optimal investment value $v_N^*$ for mixed fractional Brownian motion $S_t = W_t + \alpha B_t^H$ across Hurst parameters $H \in [0.5, 0.99]$, comparing three strategies: sum-observation, Markovian, and full-information.

```bash
python finance_example.py \
    --model mixed_fbm \
    --n 1000 \
    --hmin 0.5 \
    --hmax 1.0 \
    --hres 0.01 \
    --alpha 5.0 \
    --strategy both \
    --incremental
```

**Parameters:**
- `--n 1000`: Matrix dimension (N=500 time steps, 2N×2N covariance matrix)
- `--hmin 0.5 --hmax 1.0 --hres 0.01`: H values from 0.50 to 0.99 in steps of 0.01
- `--alpha 5.0`: Weight of fBM component
- `--strategy both`: Run both Markovian and full-information strategies (sum is always computed)
- `--incremental`: Save results after each H value (allows resuming interrupted runs)

**Output:**
- Results: `results/all_results.csv`
- Plots: `figs/mixed_fbm/`

**Runtime:** ~2-4 hours on a standard laptop (Intel i7, 32GB RAM)

To generate plots from cached results:
```bash
python finance_example.py --plot-all --plot-hmin 0.5 --plot-hmax 1.0
```

### Table 2 / Figure 2: Four Algebraic Examples (n=2000)

Demonstrates the four computational regimes: small-$m$ primal, dual formulation, group-invariant, and banded structure.

```bash
python constrained_decomposition_demo.py \
    --all \
    --verbose \
    --n1 2000 \
    --n2 2000 \
    --n3 2000 \
    --n4 2000
```

**Parameters:**
- `--n1 2000`: Example I (small S, primal Newton)
- `--n2 2000`: Example II (dual Newton-CG, tridiagonal $S^\perp$)
- `--n3 2000`: Example III (block-permutation invariant, 5 blocks)
- `--n4 2000`: Example IV (banded A and S, bandwidth b=2)

**Output:**
- LaTeX table rows: `demo_outputs/latex_table_n2000_2000_2000_2000.tex`
- Heatmap figures: `demo_outputs/demo*_n2000.png`

**Runtime:** ~6-10 minutes on a standard laptop

### Running on a Cluster (SLURM)

For large-scale computations:
```bash
# Submit demo job
./run_demo.sh -n 2000 -t 01:00:00

# Run interactively
./run_demo.sh -n 500 -i
```

## Files

| File | Description |
|------|-------------|
| `constrained_decomposition_core.py` | Main solver algorithms (Newton, Newton-CG, dual, block-efficient) |
| `constrained_decomposition_matrices.py` | Matrix construction utilities (SPD, banded, Toeplitz, fBM covariance) |
| `constrained_decomposition_viz.py` | Visualization (heatmaps, block structure plots) |
| `constrained_decomposition_demo.py` | Demo runner for the four algebraic examples |
| `finance_example.py` | Mixed fBM finance application with parallel computation |
| `toeplitz_solver.py` | Specialized Toeplitz solver exploiting 2×2 block structure |

## Citation

[Paper reference TBD]
