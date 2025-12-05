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

E = np.array([[1 / math.sqrt(2), 1j / math.sqrt(2), 0, 0],
              [0, 0, 1j / math.sqrt(2), 1 / math.sqrt(2)],
              [0, 0, 1j / math.sqrt(2), -1 / math.sqrt(2)],
              [1 / math.sqrt(2), -1j / math.sqrt(2), 0, 0]])

I = np.eye(2)

sigma_y_kron_2 = np.kron(sigma_y, sigma_y)
psi = 0.123


def gamma_2(u):
    assert len(u) == 4
    return u @ sigma_y_kron_2 @ u.T @ sigma_y_kron_2

def rz(angle):
    return np.array([[math.cos(angle / 2) - 1j*math.sin(angle / 2), 0],
                        [0, math.cos(angle / 2) + 1j*math.sin(angle / 2)]])

random_u = generate_U(2)
eigval, eigvec = np.linalg.eig(random_u)
angles = list(extract_angles_from_eigvals(eigval))
angles = angles[:3]

alpha = (angles[0] + angles[1]) / 2
beta = (angles[0] + angles[2]) / 2
delta = (angles[1] + angles[2]) / 2
print(angles)