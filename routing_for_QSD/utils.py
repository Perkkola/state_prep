from collections import deque
import numpy as np
from scipy.linalg import cossin
from scipy.linalg import polar
from numpy.linalg import eigh, pinv
from scipy.linalg import sqrtm
import sys
import math
import multiprocessing
from multiprocessing import shared_memory

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



def generate_U(num_qubits):
    arr = normalize(np.array([math.cos(2 * np.pi * x - np.pi) + 1j*math.sin(2 * np.pi * x - np.pi) for x in np.random.random_sample((2 ** num_qubits) ** 2)]))
    # arr = normalize(np.array([x for x in np.random.random_sample((2 ** num_qubits) ** 2)]))
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
    diag_s_y = np.real_if_close(diag_s_y)
    return v_final, diag_s_x, diag_s_y

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

    # unitary factors: u_x, u_y such that X = s_x @ u_x, Y = s_y @ u_y
    # Use pseudo-inverse for s_x,s_y in case of singular values (stable)
    s_x_pinv = pinv(s_x)
    s_y_pinv = pinv(s_y)
    u_x = s_x_pinv @ X
    u_y = s_y_pinv @ Y

    v, diag_s_x, diag_s_y = simultaneous_diagonalization(s_x, s_y, tol=tol)
    sigma_x = np.diag(diag_s_x)
    delta_y = np.diag(diag_s_y)

    M0 = Z @ u_y.conj().T @ s_x - W @ u_x.conj().T @ s_y
    M = M0 @ v

    u = np.block([[v, np.zeros((n, n), dtype=U.dtype)],
                [np.zeros((n, n), dtype=U.dtype), M]])
    
    cs = np.block([[sigma_x, delta_y],
                   [-delta_y, sigma_x]])
    
    vh = np.block([[v.conj().T @ u_x, np.zeros((n, n), dtype=U.dtype)],
                    [np.zeros((n, n), dtype=U.dtype), v.conj().T @ u_y]])


    return u, cs, vh

def is_unitary(U):
    UU = clean_matrix(U @ np.conj(U.T))
    return np.allclose(UU, np.eye(len(UU)))

def generate_random_rz_multiplexer_unitary(num_qubits):
    U = generate_U(num_qubits)

    sub_mat_dim = num_qubits - 1
    Z = np.zeros((2 ** sub_mat_dim, 2 ** sub_mat_dim))

    u, _, _ = compute_csd(U)

    u_1 = u[:2 ** sub_mat_dim, :2 ** sub_mat_dim]
    u_2 = u[2 ** sub_mat_dim:, 2 ** sub_mat_dim:]

    u_1_u_2_dgr = u_1 @ np.conj(u_2.T)

    eigval, _ = np.linalg.eig(u_1_u_2_dgr)

    if -1 or 1 in eigval: diag = np.diag(eigval)
    else: diag = np.diag([np.sqrt(x) for x in eigval])

    block_diag = np.block([[diag, Z],
                            [Z, np.conj(diag.T)]])
    return block_diag

def generate_random_rz_multiplexer_unitary_fast(num_qubits):
    num_rzs = 2 ** (num_qubits - 1)
    rzs =  []
    phis = []
    for _ in range(num_rzs):
        phi = np.random.random() * 2 * np.pi - np.pi
        phis.append(phi)
        rzs.append(math.cos(phi) - 1j*math.sin(phi))

    for phi in phis:
        rzs.append(math.cos(phi) + 1j*math.sin(phi))

    return np.diag(rzs)

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

        if np.round(math.sin(ang), 10) == np.round(np.imag(value), 10): ang = -ang
        yield ang

def extract_angles_from_eigvals(eigvals):
    for value in eigvals:
        ang = math.acos(np.real(value))

        if np.round(math.sin(ang), 10) == np.round(np.imag(value), 10): ang = -ang
        yield ang

def random_angles(n):
    return [np.random.random() * 2 * np.pi - np.pi for _ in range(n)]

def clean_matrix(M):
    M = M.copy()
    for i in range(len(M)):
        for j in range(len(M)):
            if np.abs(M[i][j]) < 1e-12: M[i][j] = float(0.0)
            if np.abs(np.imag(M[i][j])) < 1e-12: M[i][j] = np.real(M[i][j])
            M[i][j] = '{0:.12}'.format(M[i][j])
    return M

def _möttönen_transformation(start, stop, n, num_controls, global_angles, transformed_angles_name):
    existing_transformed_angles = shared_memory.SharedMemory(name=transformed_angles_name)
    transformed_angles = np.ndarray((n,), buffer=existing_transformed_angles.buf)

    power = math.pow(2, -num_controls)
    for i in range(start, stop):
        temp = 0
        g_m = i ^ (i >> 1)
        for j in range(n):
            dot_product = bin(g_m & j).count('1') % 2
            temp +=  -global_angles[j]  if dot_product == 1 else global_angles[j]
        transformed_angles[i] = power * temp * 2

    existing_transformed_angles.close()

def möttönen_transformation(multiplexer_angles):
        global_angles = np.array(multiplexer_angles)
        n = len(multiplexer_angles)
        transformed_angles = np.zeros(n)
        num_controls = int(math.log2(n))

        if num_controls >= 12:
            shm_transformed_angles = shared_memory.SharedMemory(create=True, size=transformed_angles.nbytes)
            new_shm_transformed_angles = np.ndarray(transformed_angles.shape, dtype=transformed_angles.dtype, buffer=shm_transformed_angles.buf)
            new_shm_transformed_angles[:] = transformed_angles[:]

            jobs = []
            cores = multiprocessing.cpu_count()

            for i in range(cores):
                start = int((n / cores) * i)
                stop = min(int((n / cores) * (i + 1)), n)
                p = multiprocessing.Process(target = _möttönen_transformation, args=(start, stop, n, num_controls, global_angles, shm_transformed_angles.name))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()
            
            transformed_angles = new_shm_transformed_angles.copy()
            shm_transformed_angles.close()
            shm_transformed_angles.unlink()
        else:
            power = math.pow(2, -num_controls)
            for i in range(n):
                temp = 0
                g_m = i ^ (i >> 1)
                for j in range(n):
                    dot_product = bin(g_m & j).count('1') % 2
                    temp +=  -global_angles[j]  if dot_product == 1 else global_angles[j]
                transformed_angles[i] = power * temp * 2
        return transformed_angles