import numpy as np
import sympy as sp
from sympy.physics.quantum import TensorProduct

def angle_givens_for(theta, alpha, beta):
    u1 = sp.exp(sp.I * alpha) * sp.cos(theta)
    u2 = sp.exp(sp.I * beta) * sp.sin(theta)
    G2 = [[u1, u2],
        [-sp.conjugate(u2),         sp.conjugate(u1)]]
    return G2

def construct_unitary(diag, theta, alpha, beta):
    n = len(diag)

    D = sp.Matrix(np.diag([sp.exp(sp.I * d) for d in diag]))
    # sp.pprint(D)
    G_list = []

    theta = list(reversed(theta))
    alpha = list(reversed(alpha))
    beta = list(reversed(beta))

    index = 0
    for j in range(n):
        for i in range(j + 1, n):
            G = sp.eye(n)
            G2 = angle_givens_for(theta[index], alpha[index], beta[index])
            G[j, j] = G2[0][0]
            G[j, i] = G2[0][1]
            G[i, j] = G2[1][0]
            G[i, i] = G2[1][1]
            # G[np.ix_([j, i], [j, i])] = G2
            G_list.append(sp.Matrix(G))
            # sp.pprint(G)
            # print('\n')
            index += 1

    U_reconstructed = D.copy()
    for G in reversed(G_list):
        U_reconstructed = U_reconstructed * G

    return sp.conjugate(U_reconstructed)

def create_smaller_unitaries():
    unitaries = []
    for l in "ABCDEF":
        phis = sp.symbols(f"φ_{l}_1 φ_{l}_2 φ_{l}_3 φ_{l}_4", real=True)
        thetas = sp.symbols(f"θ_{l}_1 θ_{l}_2 θ_{l}_3 θ_{l}_4 θ_{l}_5 θ_{l}_6", real=True)
        alphas = sp.symbols(f"a_{l}_1 a_{l}_2 a_{l}_3 a_{l}_4 a_{l}_5 a_{l}_6", real=True)
        betas = sp.symbols(f"b_{l}_1 b_{l}_2 b_{l}_3 b_{l}_4 b_{l}_5 b_{l}_6", real=True)
        U = construct_unitary(phis, thetas, alphas, betas)
        unitaries.append(U)

    return unitaries

def create_large_unitary():
    phis = sp.symbols("φ_:8",  real=True)
    thetas = sp.symbols("θ_:28",  real=True)
    alphas = sp.symbols("a_:28", real=True)
    betas = sp.symbols("b_:28", real=True)
    U = construct_unitary(phis, thetas, alphas, betas)

    return U

A, B, C, D, E, F = create_smaller_unitaries()

I_2 = sp.eye(2)

U_A = sp.Matrix(TensorProduct(I_2, A))
U_B = sp.Matrix(TensorProduct(B, I_2))
U_C = sp.Matrix(TensorProduct(I_2, C))
U_D = sp.Matrix(TensorProduct(D, I_2))
U_E = sp.Matrix(TensorProduct(I_2, E))
U_F = sp.Matrix(TensorProduct(F, I_2))


U = create_large_unitary()

sp.pprint(U[0, 0])
sp.pprint(sp.simplify((U_F * U_E * U_D * U_C * U_B * U_A)[0, 0]))

# symbolic_unitary = construct_unitary(phis, thetas, alphas, betas)

# values = {phis[0]: 0, phis[1] : 0, phis[2]: 0, phis[3]: 0, thetas[0]: 1.0654775643968202, thetas[1]: 0.6849586735331349, thetas[2]: 0.5580323953621804, thetas[3]: 0.9133819536221928, thetas[4]: 0.4374891714014266
#  , thetas[5]: 0.18481411217628455, alphas[0]: 0, alphas[1]: 0, alphas[2]: -sp.pi, alphas[3]: 0, alphas[4]: 0, alphas[5]: -sp.pi, betas[0]: 0, betas[1]: -sp.pi,
#  betas[2]: -sp.pi, betas[3]: 0, betas[4]: -sp.pi, betas[5]: -sp.pi}
