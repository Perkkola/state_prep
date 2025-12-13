import numpy as np
from utils import generate_U, extract_two_from_kron, clean_matrix, orthogonal_congruence_diagonalize
from scipy.optimize import linear_sum_assignment
import math
from pennylane.math import partial_trace

sigma_y = np.array([[0, -1j],
                    [1j, 0]])


xi = np.exp(1j * np.pi / 4)
cnot_1_2 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
cnot_1_2 = cnot_1_2 * xi

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

random_U_prime = project_to_SU4(generate_U(2))

U = random_U_prime @ cnot_1_2

# U = project_to_SU4(U)

M = gamma_map(U.T).T


t_1 = M[0][0]
t_2 = M[1][1]
t_3 = M[2][2]
t_4 = M[3][3]

psi = np.atan2(np.imag(t_1 + t_2 + t_3 + t_4), np.real(t_1 + t_4 - t_3 - t_2)) # Swap t_4 and t_2
Delta = cnot_1_2 @ np.kron(I, rz(psi)) @ cnot_1_2
gamma_U_Delta = gamma_map(U @ Delta)
eigvals = np.linalg.eigvals(gamma_U_Delta)

angles = [np.angle(eigval) for eigval in eigvals if np.angle(eigval) >= 0]

assert len(angles) >= 2, "Need positive r and s"



theta = (angles[0] + angles[1]) / 2
phi = (angles[0] - angles[1]) / 2

E_dgr = np.conjugate(E).T

U_E = (E_dgr @ U @ cnot_1_2 @ np.kron(I, rz(psi)) @ cnot_1_2 @ E) # Insert cnot_1_2 after U seems to work
S_U = U_E @ (U_E.T)

kernel = cnot_1_2 @ np.kron(rx(theta + np.pi), rz(phi)) @ cnot_1_2 # Add + np.pi to get correct eigenvalues
k_E = (E_dgr @ kernel @ E)
S_k = k_E @ (k_E.T)

A_U, D_U = orthogonal_congruence_diagonalize(S_U)
B_k, D_k = orthogonal_congruence_diagonalize(S_k)


C = np.conjugate(k_E).T @ B_k @ A_U.T @ U_E

u = A_U @ B_k.T @ k_E @ C
v = B_k @ A_U.T @ U_E @ C.T

A_tilde = E @ B_k @ A_U.T @ E_dgr
C_tilde = E @ C.T @ E_dgr

#Try taking partial trace and the transform back from the magic basis

# random_rz = rz(0.123)
# random_rx = rx(0.456)

# print(random_rz)
# print(random_rx)

# kron_prod = np.kron(random_rz, random_rx)

# a = partial_trace(kron_prod, indices=[1])
# b = partial_trace(kron_prod, indices=[0])

# def project_to_SU2(U):
#     detU = np.linalg.det(U)
#     return U / detU**0.5

# a = project_to_SU2(a)
# b = project_to_SU2(b)
# print("/////////////////")
# print(a)
# print(b)
# exit()

# a = partial_trace(A_tilde, indices=[0])
# b = partial_trace(A_tilde, indices=[1])


# print(a)
# print(b)
# a, b = extract_two_from_kron(E @ B_k @ A_U.T @ E_dgr) 
# c, d = extract_two_from_kron(E @ C.T @ E_dgr)

# print(a)
# print(b)
# exit()

a = partial_trace(A_tilde, indices=[1])
b = partial_trace(A_tilde, indices=[0])
c = partial_trace(C_tilde, indices=[1])
d = partial_trace(C_tilde, indices=[0])


def project_to_SU2(U):
    detU = np.linalg.det(U)
    return U / detU**0.5


a = project_to_SU2(a)
b = project_to_SU2(b)
c = project_to_SU2(c)
d = project_to_SU2(d)

phase = np.linalg.det(a)
a = a / phase**0.5
b = b * phase**0.5

phase = np.linalg.det(c)
c = c / phase**0.5
d = d * phase**0.5

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

recon = np.kron(a, b) @ kernel @ np.kron(c, d)
U_lhs = U @ cnot_1_2 @ np.kron(I, rz(psi)) @ cnot_1_2

print(recon)
print(U_lhs)


print(reconstruction_error_up_to_global_phase(recon, U_lhs))

