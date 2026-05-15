import numpy as np

A   = np.array([[2.0, -1.0], [-1.0, 1.0]])
Lam = np.array([[5.0,  2.0], [ 2.0, 1.0]])
E   = np.array([[1.0,  0.5], [ 0.5, 2.0]])     # all 4 entries nonzero
print(f"A = {A.tolist()},  Lambda = {Lam.tolist()},  E = {E.tolist()}\n")

def UL(A, Lam):
    (a, b), (_, c) = A
    (p, q), (_, r) = Lam
    pivot = 1 + c*r + b*q
    U = np.array([[0, q/(1+c*r)], [0, 0]])
    L = np.array([[((1+c*r)*p - c*q*q)/pivot, 0], [q, r]])
    return U, L, pivot

print("Case 1: A_eps = A + eps * E")
for eps in [0.1, 0.01, 0.001, 0.0001]:
    U, L, pi = UL(A + eps*E, Lam)
    print(f"  eps={eps:<7}  pivot={pi:<7.4f}  U={U.flatten().round(5).tolist()}  L={L.flatten().round(4).tolist()}")

print("\nCase 2: Lambda_eps = Lambda + eps * E")
for eps in [0.1, 0.01, 0.001, 0.0001]:
    U, L, pi = UL(A, Lam + eps*E)
    print(f"  eps={eps:<7}  pivot={pi:<7.4f}  U={U.flatten().round(5).tolist()}  L={L.flatten().round(4).tolist()}")
