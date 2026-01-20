# Constrained Decomposition

Algorithms for constrained SPD matrix decomposition: given $A \succ 0$ and linear subspace $S$, find

$$A = B^{-1} + C, \quad B \in S^\perp, \quad C \in S$$

## Features

- **Primal Newton solver** for small constraint subspace $S$
- **Dual Newton solver** for small orthogonal complement $S^\perp$
- **Group-invariant reduction** for symmetric problems
- **Newton-CG with banded structure** for $O(nb^2)$ complexity
- **Finance application**: exponential utility maximization for mixed fBM

## Quick Start

```bash
# Run demo with all 4 examples
python constrained_decomposition_demo.py --all --verbose

# Run finance example
python finance_example.py --model mixed_fbm --n 100 --H 0.7 --alpha 5.0
```

## Files

- `constrained_decomposition_core.py` - Main solver algorithms
- `constrained_decomposition_matrices.py` - Matrix construction utilities
- `constrained_decomposition_viz.py` - Visualization
- `constrained_decomposition_demo.py` - Demo runner for paper examples
- `finance_example.py` - Mixed fBM finance application
- `toeplitz_solver.py` - Toeplitz matrix utilities

## Citation

[Paper reference TBD]
