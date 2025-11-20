from collections import deque
import numpy as np
from scipy.linalg import cossin
from scipy.linalg import polar
from numpy.linalg import eigh, pinv
from scipy.linalg import sqrtm
import sys
import math
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def normalize(data):
    sq = 0
    for el in data:
        sq += np.square(el)

    sq = np.sqrt(sq)
    arr = data / sq
    return arr

def grey_code(dist, half):
    upper_bound = (2 ** (dist - 1) + 1) if half else 2 ** dist + 1

    for i in range(1, upper_bound):
        highest_index_diff = 0
        for j in range(dist):
            if (i >> j) & 1 != ((i - 1) >> j) & 1: highest_index_diff = j + 1
        grey_gate_queue.append("RZ")
        grey_gate_queue.append((dist - highest_index_diff, dist))

        grey_state_queue.append(grey_state[dist])
        grey_state[dist] ^= grey_state[dist - highest_index_diff]

def get_grey_gates(dist, half = False, all_gates = True, state_queue = False):
    global grey_gate_queue
    global grey_state
    global grey_state_queue

    grey_gate_queue = deque()
    grey_state_queue = deque()
    grey_state = {q: 1 << q for q in range(dist + 1)}
    
    grey_code(dist, half)

    long_range_gates = []
    for gate in grey_gate_queue:
        if gate == "RZ": continue
        if gate[1] - gate[0] > 1 and not all_gates: long_range_gates.append(gate)
        elif all_gates: long_range_gates.append(gate)
    
    if state_queue:
        return long_range_gates, grey_state_queue
    else:
        return long_range_gates


def generate_random_rz_multiplexer_unitary(num_qubits):
    arr = normalize(np.array([2 * x - 1 for x in np.random.random_sample((2 ** num_qubits) ** 2)]))
    mat_A = arr.reshape((2 ** num_qubits, 2 ** num_qubits))
    U, _, _ = np.linalg.svd(mat_A)

    sub_mat_dim = num_qubits - 1
    Z = np.zeros((2 ** sub_mat_dim, 2 ** sub_mat_dim))

    u, _, _ = cossin(U, p= 2 ** sub_mat_dim, q= 2 ** sub_mat_dim)

    u_1 = u[:2 ** sub_mat_dim, :2 ** sub_mat_dim]
    u_2 = u[2 ** sub_mat_dim:, 2 ** sub_mat_dim:]

    u_1_u_2_dgr = u_1 @ np.conj(u_2.T)

    eigval, _ = np.linalg.eig(u_1_u_2_dgr)

    if -1 or 1 in eigval: diag = np.diag(eigval)
    else: diag = np.diag([np.sqrt(x) for x in eigval])

    block_diag = np.block([[diag, Z],
                            [Z, np.conj(diag.T)]])
    return block_diag


def generate_U(num_qubits):
    arr = normalize(np.array([2 * x - 1 for x in np.random.random_sample((2 ** num_qubits) ** 2)]))
    mat_A = arr.reshape((2 ** num_qubits, 2 ** num_qubits))
    U, _, _ = np.linalg.svd(mat_A)
    return U

def simultaneous_diagonalization(s_x, s_y, tol=1e-12):
    eigvals, v = eigh(s_x)
    # sort ascending for determinism
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    v = v[:, idx]

    # group degenerate eigenvalues
    groups = []
    current = [0]
    for i in range(1, len(eigvals)):
        if abs(eigvals[i] - eigvals[i-1]) < tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    v_final = v.copy()
    diag_s_y = np.zeros_like(eigvals, dtype=float)

    for group in groups:
        if len(group) == 1:
            k = group[0]
            diag_s_y[k] = np.real_if_close((v[:, k].conj().T @ s_y @ v[:, k]))
            continue
        # refine basis inside degenerate block
        basis = v[:, group]                # n x m
        s_y_sub = basis.conj().T @ s_y @ basis  # m x m Hermitian
        evals_sub, U_sub = eigh(s_y_sub)
        v_final[:, group] = basis @ U_sub
        diag_s_y[group] = np.real_if_close(evals_sub)


    diag_s_x = np.real_if_close(eigvals)
    # clip small negative eigenvalues due to rounding
    diag_s_x = np.clip(diag_s_x, 0.0, None)
    diag_s_y = np.real(diag_s_y)
    return v_final, diag_s_x, diag_s_y

def compute_csd(U, tol=1e-12):
    n = U.shape[0] // 2

    X = U[:n, :n]
    Y = U[:n, n:]
    W = U[n:, :n]
    Z = U[n:, n:]

    # Hermitian PSD factors (more stable than polar for our decomposition)
    XX = X @ X.conj().T
    YY = Y @ Y.conj().T
    s_x = sqrtm(XX)
    s_y = sqrtm(YY)
    # enforce Hermitian (numerical)
    s_x = 0.5 * (s_x + s_x.conj().T)
    s_y = 0.5 * (s_y + s_y.conj().T)

    # unitary factors: Ux, Uy such that X = s_x @ Ux, Y = s_y @ Uy
    # Use pseudo-inverse for s_x,s_y in case of singular values (stable)
    s_x_pinv = pinv(s_x)
    s_y_pinv = pinv(s_y)
    u_x = s_x_pinv @ X
    u_y = s_y_pinv @ Y

    v, diagSx, diagSy = simultaneous_diagonalization(s_x, s_y, tol=tol)
    sigma_x = np.diag(diagSx)
    delta_y = np.diag(diagSy)

    M0 = Z @ u_y.conj().T @ s_x - W @ u_x.conj().T @ s_y
    M = M0 @ v

    u = np.block([
        [v, np.zeros((n, n), dtype=U.dtype)],
        [np.zeros((n, n), dtype=U.dtype), M]
    ])
    cs = np.block([[sigma_x, delta_y],
                   [-delta_y, sigma_x]])
    vh = np.block([
        [v.conj().T @ u_x, np.zeros((n, n), dtype=U.dtype)],
        [np.zeros((n, n), dtype=U.dtype), v.conj().T @ u_y]
    ])


    return u, cs, vh
def extract_single_qubit_unitaries(mat):
    half = len(mat) // 2
    for i in range(half):
        j = int((f"{{:0>{int(math.log2(half))}b}}".format(i))[::-1], 2) #Don't ask
        first = mat[j][j]
        second = mat[half + j][half + j]
        yield np.diag([first, second])

def extract_angles(unitaries):
    for unitary in unitaries:
        value = unitary[0][0]

        if value > 1: value = 1
        elif value < -1: value = -1

        ang = math.acos(np.real(value))

        if np.round(math.sin(ang), 5) == np.round(np.imag(value), 5): ang = -ang
        yield ang

def clean_matrix(M):
    M = M.copy()
    for i in range(len(M)):
        for j in range(len(M)):
            if np.abs(M[i][j]) < 1e-10: M[i][j] = float(0.0)
            if np.abs(np.imag(M[i][j])) < 1e-10: M[i][j] = np.real(M[i][j])
            M[i][j] = '{0:.8}'.format(M[i][j])
    return M