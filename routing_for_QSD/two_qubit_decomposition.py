import numpy as np
from utils import orthogonal_congruence_diagonalize, get_zyz_angles
from pennylane.math import partial_trace
from collections import deque
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import Aer

sigma_y = np.array([[0, -1j],
                    [1j, 0]])


xi = np.exp(1j * np.pi / 4)
cnot_1_2 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])
cnot_1_2 = cnot_1_2 * xi


E = np.array([[1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0],
              [0, 0, 1j / np.sqrt(2), 1 / np.sqrt(2)],
              [0, 0, 1j / np.sqrt(2), -1 / np.sqrt(2)],
              [1 / np.sqrt(2), -1j / np.sqrt(2), 0, 0]])

I = np.eye(2)

sigma_y_kron_2 = np.kron(sigma_y, sigma_y)

def print_circ_unitary(qc):
    qc = qc.copy()
    qc.save_unitary()
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)

    result = simulator.run(qc).result()
    unitary = result.get_unitary(qc)
    U = np.asarray(unitary) / np.linalg.det(unitary) ** (1 / 4)
    print("Circuit unitary:\n", U.round(5))

def project_to_SU4(U):
    detU = np.linalg.det(U)

    assert detU != 0, "Matrix is not unitary!"

    return U / detU ** (1 / 4)

def project_to_SU2(U):
    detU = np.linalg.det(U)
    return U / detU ** (1 / 2)

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

def ry(angle):
    return np.array([
        [np.cos(angle / 2), -np.sin(angle / 2)],
        [np.sin(angle / 2), np.cos(angle / 2)]
    ])

def extract_diagonal(u, source):
    # print(u)
    random_U_prime = project_to_SU4(u)

    U = random_U_prime

    M = gamma_map(U.T).T

    t_1 = M[0][0]
    t_2 = M[1][1]
    t_3 = M[2][2]
    t_4 = M[3][3]

    psi = np.atan2(np.imag(t_1 + t_2 + t_3 + t_4), np.real(t_1 + t_4 - t_3 - t_2)) # Swap t_4 and t_2
    Delta =  cnot_1_2 @ np.kron(I, rz(psi)) @ cnot_1_2
    gamma_U_Delta = gamma_map(U @ Delta)
    eigvals = np.linalg.eigvals(gamma_U_Delta)

    angles = [np.angle(eigval) for eigval in eigvals if np.angle(eigval) >= 0]

    assert len(angles) >= 2, "Need positive r and s"

    theta = (angles[0] + angles[1]) / 2
    phi = (angles[0] - angles[1]) / 2

    E_dgr = np.conjugate(E).T

    U_E = (E_dgr @ U @ Delta @ E)
    S_U = U_E @ (U_E.T)

    kernel = cnot_1_2 @ np.kron(rx(theta + np.pi), rz(phi)) @ cnot_1_2 # Add + np.pi to get correct eigenvalues
    k_E = (E_dgr @ kernel @ E)
    S_k = k_E @ (k_E.T)

    A_U, D_U = orthogonal_congruence_diagonalize(S_U)
    B_k, D_k = orthogonal_congruence_diagonalize(S_k)


    C = np.conjugate(k_E).T @ B_k @ A_U.T @ U_E

    A_tilde =  E @ A_U @ B_k.T @ E_dgr
    C_tilde = E @ C @ E_dgr

    a = partial_trace(A_tilde, indices=[1])
    b = partial_trace(A_tilde, indices=[0])
    c = partial_trace(C_tilde, indices=[1])
    d = partial_trace(C_tilde, indices=[0])

    a = project_to_SU2(a)
    b = project_to_SU2(b)
    c = project_to_SU2(c)
    d = project_to_SU2(d)

    a = -a if np.round(np.kron(a, b)[0][0], 12) == -np.round(A_tilde[0][0], 12) else a
    c = -c if np.round(np.kron(c, d)[0][0], 12) == -np.round(C_tilde[0][0], 12) else c

    recon = np.kron(-a, b) @ kernel @ np.kron(c, d) @ cnot_1_2 @ np.kron(I, rz(-psi)) @ cnot_1_2
    # U_lhs = U 
    diag_u = cnot_1_2 @ np.kron(I, rz(-psi)) @ cnot_1_2


    a_1, a_2, a_3 = get_zyz_angles(a)
    b_1, b_2, b_3 = get_zyz_angles(b)
    c_1, c_2, c_3 = get_zyz_angles(c)
    d_1, d_2, d_3 = get_zyz_angles(d)

    diag_gates = deque()
    diag_gates.append((source, 1))
    diag_gates.append(('RZ', -psi, 1))
    diag_gates.append((source, 1))


    two_cnot_unitary_gates = deque()
    two_cnot_unitary_gates.append(('RZ', c_3, source))
    two_cnot_unitary_gates.append(('RY', c_2, source))
    two_cnot_unitary_gates.append(('RZ', c_1, source))
    two_cnot_unitary_gates.append(('RZ', d_3, 1))
    two_cnot_unitary_gates.append(('RY', d_2, 1))
    two_cnot_unitary_gates.append(('RZ', d_1, 1))
    two_cnot_unitary_gates.append((source, 1))
    two_cnot_unitary_gates.append(('RZ', phi, 1))
    two_cnot_unitary_gates.append(('RX', theta + np.pi, source))
    two_cnot_unitary_gates.append((source, 1))
    two_cnot_unitary_gates.append(('RZ', a_3, source))
    two_cnot_unitary_gates.append(('RY', a_2, source))
    two_cnot_unitary_gates.append(('RZ', a_1, source))
    two_cnot_unitary_gates.append(('RZ', b_3, 1))
    two_cnot_unitary_gates.append(('RY', b_2, 1))
    two_cnot_unitary_gates.append(('RZ', b_1, 1))

    # qc = QuantumCircuit(2)
    # qc.cx(0, 1)
    # qc.rz(-psi, 1)
    # qc.cx(0, 1)

    # qc.rz(c_3, 1)
    # qc.ry(c_2, 1)
    # qc.rz(c_1, 1)
    # qc.rz(d_3, 0)
    # qc.ry(d_2, 0)
    # qc.rz(d_1, 0)
    # qc.cx(0, 1)
    # qc.rz(phi, 1)
    # qc.rx(theta + np.pi, 0)
    # qc.cx(0, 1)
    # qc.rz(a_3, 1)
    # qc.ry(a_2, 1)
    # qc.rz(a_1, 1)
    # qc.rz(b_3, 0)
    # qc.ry(b_2, 0)
    # qc.rz(b_1, 0)

    # print(qc)
    # print_circ_unitary(qc)
    # print(U)
    # # print(recon)
    # exit()

    return diag_gates, diag_u, two_cnot_unitary_gates
