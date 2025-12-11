import numpy as np
from utils import generate_U, extract_angles_from_eigvals
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

I = np.eye(2)

sigma_y_kron_2 = np.kron(sigma_y, sigma_y)

def project_to_SU4(U):
    detU = np.linalg.det(U)
    return U / detU ** (1 / 4)

def gamma_map(u):
    assert len(u) == 4
    return u @ sigma_y_kron_2 @ u.T @ sigma_y_kron_2

def rz(angle):
    return np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])

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

eigval, eigvec = np.linalg.eig(gamma_U_Delta)
print(eigval)
print(np.trace(gamma_U_Delta))