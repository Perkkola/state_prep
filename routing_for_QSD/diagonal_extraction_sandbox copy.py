import numpy as np
from utils import generate_U, extract_two_from_kron
import math

sigma_y = np.array([[0, -1j],
                    [1j, 0]])

cnot_1_2 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])

cnot_2_1 = np.array([[1, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0]])

E = np.array([[1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0],
              [0, 0, 1j / np.sqrt(2), 1 / np.sqrt(2)],
              [0, 0, 1j / np.sqrt(2), -1 / np.sqrt(2)],
              [1 / np.sqrt(2), -1j / np.sqrt(2), 0, 0]])

# E = np.array([[1 / np.sqrt(2), 0, 0, 1j / np.sqrt(2)],
#               [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
#               [0, -1 / np.sqrt(2), 1j / np.sqrt(2), 0],
#               [1 / np.sqrt(2), 0, 0, -1j / np.sqrt(2)]])

# E = np.array([[1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0],
#               [1 / np.sqrt(2), -1j / np.sqrt(2), 0, 0],
#               [0, 0, 1j / np.sqrt(2), -1 / np.sqrt(2)],
#               [0, 0, 1j / np.sqrt(2), 1 / np.sqrt(2)]
#               ])

I = np.eye(2)

sigma_y_kron_2 = np.kron(sigma_y, sigma_y)

def project_to_SU4(U):
    detU = np.linalg.det(U)

    assert detU != 0, "Matrix is not unitary!"

    return U / detU ** (1 / 4)

def gamma_map(u):
    assert len(u) == 4
    return u @ sigma_y_kron_2 @ u.T @ sigma_y_kron_2

def rz(angle):
    return np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])

def rx(angle):
    return np.array([
        [np.cos(angle / 2), -1j * np.sin(angle / 2)],
        [-1j * np.sin(angle / 2), np.cos(angle / 2)]
    ])

def orthogonal_congruence_diagonalize(S, tol_eig=1e-8):
    """
    Given a complex symmetric 4x4 matrix S = S.T,
    returns real orthogonal A and diagonal D (complex phases)
    such that D = A.T @ S @ A (up to numerical tolerance).

    Handles degenerate eigenvalues in Re(S) by block-diagonalization
    of Im(S) inside degenerate subspaces.
    """
    assert S.shape == (4,4)
    assert np.allclose(S, S.T, atol=1e-10), "S must be symmetric (S==S.T)"

    # Real/imag parts (real symmetric)
    R = np.real(S)
    ImS = np.imag(S)

    # 1) Eigendecompose real symmetric R: use eigh to get real orthonormal Q
    eigvals, Q = np.linalg.eigh(R)   # Q columns are eigenvectors (real)

    # 2) Group eigenvalues into degenerate clusters
    clusters = []
    used = np.zeros(len(eigvals), dtype=bool)
    for i in range(len(eigvals)):
        if used[i]:
            continue
        # find indices j where eigvals[j] ~= eigvals[i]
        idx = [j for j in range(len(eigvals)) if abs(eigvals[j] - eigvals[i]) < tol_eig]
        for j in idx:
            used[j] = True
        clusters.append(idx)

    # 3) For each cluster of size > 1, diagonalize ImS restricted to that subspace
    Q_final = Q.copy()
    for cluster in clusters:
        if len(cluster) == 1:
            continue
        # form basis vectors for this cluster
        cols = cluster
        subQ = Q[:, cols]   # shape (4, k)
        # Project ImS into this subspace: B = subQ.T @ ImS @ subQ  (real symmetric)
        B = subQ.T @ ImS @ subQ
        # B is real symmetric; diagonalize it to get an orthonormal transform U_block
        bvals, Ublock = np.linalg.eigh(B)
        # Replace the columns subQ @ Ublock into Q_final
        Q_final[:, cols] = subQ @ Ublock

    # Q_final should be real orthogonal
    # enforce orthonormality numerically (e.g. via QR) if needed
    # small re-orthonormalization:
    U, _, Vh = np.linalg.svd(Q_final)
    Q_final = U @ Vh   # now orthonormal and det = +/-
    # enforce det=+1 by flipping sign of first column if needed
    if np.linalg.det(Q_final) < 0:
        Q_final[:, 0] = -Q_final[:, 0]

    A = Q_final   # real orthogonal

    # 4) compute diagonal via congruence
    D = A.T @ S @ A

    # zero-out tiny off-diagonals
    off = D.copy()
    for i in range(4):
        for j in range(4):
            if i != j and abs(off[i, j]) < 1e-10:
                off[i, j] = 0.0
    D = off

    return A, D

random_U_prime = project_to_SU4(generate_U(2))

U = random_U_prime @ cnot_1_2

# U = project_to_SU4(U)

M = gamma_map(U.T).T


t_1 = M[0][0]
t_2 = M[1][1]
t_3 = M[2][2]
t_4 = M[3][3]

psi = np.atan2(np.imag(t_1 + t_2 + t_3 + t_4), np.real(t_1 + t_4 - t_3 - t_2))
Delta = cnot_1_2 @ np.kron(I, rz(psi)) @ cnot_1_2
gamma_U_Delta = gamma_map(U @ Delta)
eigvals = np.linalg.eigvals(gamma_U_Delta)
angles = [np.angle(eigval) for eigval in eigvals if np.angle(eigval) >= 0]

assert len(angles) >= 2, "Need positive r and s"

theta = (angles[0] + angles[1]) / 2
phi = (angles[0] - angles[1]) / 2

E_dgr = np.conjugate(E).T

U_E = (E_dgr @ U @ np.kron(I, rz(psi)) @ cnot_1_2 @ E)
S_U = U_E @ (U_E.T)

kernel = cnot_1_2 @ np.kron(rz(theta), rx(phi)) @ cnot_1_2
k_E = (E_dgr @ kernel @ E)
S_k = k_E @ (k_E.T)


A_U, D_U = orthogonal_congruence_diagonalize(S_U)
B_k, D_k = orthogonal_congruence_diagonalize(S_k)


C = np.conjugate(k_E).T @ B_k.T @ A_U @ U_E




A_tilde = E @ A_U @ E_dgr   # should be a ⊗ b
C_tilde = E @ C @ E_dgr     # should be c ⊗ d

a, b = extract_two_from_kron(A_tilde)
c, d = extract_two_from_kron(C_tilde)

def reconstruction_error_up_to_global_phase(X, Y):
    """
    Return Frobenius norm of X - e^{i phi} Y minimized over phi.
    Equivalent to norm(X - ( <X,Y> / |<X,Y>| ) * Y).
    """
    # flatten and compute inner product
    x = X.ravel()
    y = Y.ravel()
    inner = np.vdot(y, x)  # note vdot(y,x) = conj(y)^T x
    if abs(inner) < 1e-16:
        # orthogonal-ish; can't align a phase
        return np.linalg.norm(X - Y)
    phase = inner / abs(inner)
    return np.linalg.norm(X - phase * Y)


