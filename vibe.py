import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
# --- Step 1: Helpers ---
def random_unitary(n=4, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(X)
    phases = np.diag(R) / np.abs(np.diag(R))
    Q = Q * phases.conj()
    return Q

def complex_givens_for(a, b):
    if np.abs(b) == 0:
        return np.eye(2, dtype=complex), np.abs(a)
    r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
    u1 = a / r
    u2 = b / r
    G2 = np.array([[np.conj(u1), np.conj(u2)],
                   [-u2,         u1]], dtype=complex)
    return G2, r

def givens_decomposition_left(U):
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
    return G_list, A

def right_givens_decomposition(U):
    G_list_T, R_T = givens_decomposition_left(U.T)
    D = np.diag(np.diag(R_T.T))
    G_right_list = [np.conj(G) for G in reversed(G_list_T)]
    return D, G_right_list

def extract_theta_psi(block):
    """Given 2x2 unitary block, return theta (rad) and psi (rad)."""
    u1 = block[1, 1]
    u2 = -block[1, 0]
    theta = np.arctan2(np.abs(u2), np.abs(u1))
    psi = np.angle(u2) - np.angle(u1)
    # normalize psi to (-pi, pi]
    psi = (psi + np.pi) % (2*np.pi) - np.pi
    return theta, psi

# --- Step 2: Example run ---
if __name__ == "__main__":
    U = random_unitary(4, seed=42)
#     U = np.array([[-0.54409413, -0.10171701, -0.25889078,  0.79157488],
#  [-0.39583923,  0.59434125, -0.58362529, -0.38658932],
#  [ 0.7315188,   0.35371355, -0.41023319,  0.41409624],
#  [-0.11024121,  0.71505164,  0.65120336,  0.22908997]])
    D, G_list = right_givens_decomposition(U)

    # Verify reconstruction
    U_reconstructed = D.copy()
    for G in G_list:
        U_reconstructed = U_reconstructed @ G
    print("Max reconstruction error:",
          np.max(np.abs(U - U_reconstructed)))

    print("\nDiagonal D:")
    print(np.round(D, 6))

    for idx, G in enumerate(G_list, start=1):
        rows = sorted(set(np.where(np.abs(G - np.eye(4)) > 1e-10)[0]))
        block = G[np.ix_(rows, rows)]
        theta, psi = extract_theta_psi(block)
        print(f"\nG_{idx} acts on indices {rows}:")
        print(np.round(block, 6))
        print(f"  θ = {theta:.6f} rad ({np.degrees(theta):.4f}°)")
        print(f"  ψ = {psi:.6f} rad ({np.degrees(psi):.4f}°)")
