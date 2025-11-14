from collections import deque
import numpy as np
from scipy.linalg import cossin
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

def extract_single_qubit_unitaries(mat):
    half = len(mat) // 2
    for i in range(half):
        j = int((f"{{:0>{int(math.log2(half))}b}}".format(i))[::-1], 2) #Don't ask
        first = mat[j][j]
        second = mat[half + j][half + j]
        yield np.diag([first, second])

def extract_angles(mat):
    half = len(mat) // 2
    for i in range(half):
        j = int((f"{{:0>{int(math.log2(half))}b}}".format(i))[::-1], 2)
        value = mat[i][i]
        if value > 1: value = 1
        elif value < -1: value = -1
        yield math.acos(np.real(value))

def clean_matrix(M):
    M = M.copy()
    for i in range(len(M)):
        for j in range(len(M)):
            if np.abs(M[i][j]) < 1e-10: M[i][j] = float(0.0)
            if np.abs(np.imag(M[i][j])) < 1e-10: M[i][j] = np.real(M[i][j])
            M[i][j] = '{0:.8}'.format(M[i][j])
    return M