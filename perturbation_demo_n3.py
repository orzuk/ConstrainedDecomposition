"""
3x3 example where the algorithm fails at step k=1, so L[1,1] (not L[0,0])
is the entry that blows up under perturbation. L[2,2] remains finite
because the bottom step sets L[2,2] = Lambda[2,2] unconditionally.
"""
import numpy as np
import sys
sys.path.insert(0, "/mnt/c/Code/Github/ConstrainedDecomposition")
from triangular_decomposition import backward_triangular_elimination

# Block-diagonal embedding: rows/cols 1,2 carry the 2x2 obstruction.
A = np.array([[1.0,  0.0,  0.0],
              [0.0,  2.0, -1.0],
              [0.0, -1.0,  1.0]])
Lam = np.array([[1.0,  0.0,  0.0],
                [0.0,  5.0,  2.0],
                [0.0,  2.0,  1.0]])
E = np.array([[0.0,  0.0,  0.0],
              [0.0,  1.0,  0.5],
              [0.0,  0.5,  2.0]])     # perturbation supported on the failing block

print(f"A = {A.tolist()}")
print(f"Lambda = {Lam.tolist()}")
print(f"E = {E.tolist()}\n")

print("A_eps = A + eps * E,  Lambda fixed")
for eps in [0.1, 0.01, 0.001, 0.0001]:
    A_eps = A + eps * E
    U, L, info = backward_triangular_elimination(A_eps, Lam)
    print(f"  eps={eps:<7}  status={info.status:8}")
    print(f"     diag(L) = {[round(L[i,i], 4) for i in range(3)]}")
