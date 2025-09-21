import numpy as np

# --- Step 1: Helpers ---
def random_unitary(n=4, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(X)
    phases = np.diag(R) / np.abs(np.diag(R))
    Q = Q * phases.conj()  # Fix determinant phases
    return Q

def complex_givens_for(a, b):
    """Return 2x2 Givens matrix G such that G @ [a; b] = [r; 0]"""
    if np.abs(b) == 0:
        return np.eye(2, dtype=complex), np.abs(a)
    r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
    u1 = a / r
    u2 = b / r
    G2 = np.array([[np.conj(u1), np.conj(u2)],
                   [-u2,         u1]], dtype=complex)
    return G2, r

def angle_givens_for(theta, psi):
    u1 = np.cos(theta)
    u2 = np.exp(1j * psi) * np.sin(theta)
    G2 = np.array([[np.conj(u1), np.conj(u2)],
                   [-u2,         u1]], dtype=complex)
    return G2


def givens_decomposition_left(U):
    """Left multiplication by Givens to zero out below-diagonal entries."""
    n = U.shape[0]
    A = U.copy().astype(complex)
    G_list = []
    for j in range(n - 1):
        for i in range(n - 1, j, -1):
            a = A[i - 1, j]
            b = A[i, j]
            G2, r = complex_givens_for(a, b)
            G = np.eye(n, dtype=complex)
            G[np.ix_([i - 1, i], [i - 1, i])] = G2
            print(G)
            A = G @ A
            G_list.append(G)
    return G_list, A  # A will be diagonal for unitary input

# --- Step 2: Decompose U ---
def right_givens_decomposition(U):
    # Decompose U^T with left Givens
    G_list_T, R_T = givens_decomposition_left(U.T)
    # D is diagonal phases
    D = np.diag(np.diag(R_T.T))
    # Right factors are conjugates of G_list_T in reverse order
    G_right_list = [np.conj(G) for G in reversed(G_list_T)]
    return D, G_right_list

def extract_theta_psi(block):
    """Given 2x2 unitary block, return theta (rad) and psi (rad)."""
    u1 = block[1, 1]
    u2 = -block[1, 0]
    cmplx = 0
    # if np.imag(u1) > 1e-10: cmplx = 1

    theta = np.sign(u1) * np.acos(np.real(u1))

    # print(u1)
    theta = np.atan2(np.abs(u2), np.abs(u1))
    psi = np.angle(u2) - np.angle(u1)
    # normalize psi to (-pi, pi]
    psi = (psi + np.pi) % (2*np.pi) - np.pi
    return theta, psi, cmplx

def get_givens_angles(G_list):
    thetas = []
    psis = []
    cmplxs = []
    for G in G_list:
        rows = sorted(set(np.where(np.abs(G - np.eye(len(U))) > 1e-10)[0]))
        block = G[np.ix_(rows, rows)]
        theta, psi, cmplx = extract_theta_psi(block)
        thetas.append(theta)
        psis.append(psi)
        cmplxs.append(cmplx)
    return thetas, psis, cmplxs

def get_diag_angles(D):
    D = np.round(D, 6)
    return [np.acos(np.real(D[i][i])) if np.abs(np.real(D[i][i])) <= 1 else np.acos(np.sign(np.real(D[i][i]))) for i in range(len(D))]

def construct_unitary(diag, theta, psi):
    n = len(diag)

    D = np.diag([np.exp(1j * d) for d in diag])
    G_list = []

    print('ANGLE\n')
    index = 0
    for j in range(n):
        for i in range(j + 1, n):
            G = np.eye(n, dtype=complex)
            G2 = angle_givens_for(theta[index], psi[index])
            G[np.ix_([j, i], [j, i])] = G2
            G_list.append(G)
            print(G)
            index += 1

    for j in range(n - 1):
        for i in range(n - 1, j, -1):
            G2 = angle_givens_for(heta, b)
            G = np.eye(n, dtype=complex)
            G[np.ix_([i - 1, i], [i - 1, i])] = G2

    U_reconstructed = D.copy()
    for G in reversed(G_list):
        U_reconstructed = U_reconstructed @ G

    # print(U_reconstructed)


# --- Step 3: Example run ---
if __name__ == "__main__":
    U = random_unitary(4, seed=42)
    D, G_list = right_givens_decomposition(U)

    # Verify reconstruction
    U_reconstructed = D.copy()
    for G in G_list:
        U_reconstructed = U_reconstructed @ G

    print("Max reconstruction error:",
          np.max(np.abs(U - U_reconstructed)))
    
    
    diag_angles = get_diag_angles(D)
    thetas, psis = get_givens_angles(G_list)

    construct_unitary(diag_angles, thetas, psis)

    # Show results
    print("\nDiagonal D:")
    print(np.round(D, 6))
    for idx, G in enumerate(G_list, start=1):
        # find the active 2x2 block
        rows = sorted(set(np.where(np.abs(G - np.eye(4)) > 1e-10)[0]))
        block = G[np.ix_(rows, rows)]
        print(f"\nG_{idx} acts on indices {rows}:")
        print(np.round(block, 6))
