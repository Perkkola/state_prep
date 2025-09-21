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

def angle_givens_for(theta, alpha, beta):
    u1 = np.exp(1j * alpha) * np.cos(theta)
    u2 = np.exp(1j * beta) * np.sin(theta)
    G2 = np.array([[u1, u2],
                   [-np.conj(u2),         np.conj(u1)]], dtype=complex)
    return G2

def givens_decomposition_left(U):
    n = U.shape[0]
    A = U.copy().astype(complex)
    G_list = []
    for j in range(n):
        for i in range(j + 1, n):
            a = A[j, j]
            b = A[i, j]
            G2, r = complex_givens_for(a, b)
            G = np.eye(n, dtype=complex)
            G[np.ix_([j, i], [j, i])] = G2
            
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


    # theta = np.sign(u1) * np.acos(np.real(u1))

    theta = np.atan2(np.abs(u2), np.abs(u1))
    alpha = np.angle(u1)
    beta = np.angle(u2)

    # normalize psi to (-pi, pi]
    # psi = (psi + np.pi) % (2*np.pi) - np.pi
    return theta, alpha, beta

def get_givens_angles(G_list):
    thetas = []
    alphas = []
    betas = []
    for G in G_list:
        rows = sorted(set(np.where(np.abs(G - np.eye(len(U))) > 1e-10)[0]))
        block = G[np.ix_(rows, rows)]
        theta, alpha, beta = extract_theta_psi(block)
        thetas.append(theta)
        alphas.append(alpha)
        betas.append(beta)
    return thetas, alphas, betas

def get_diag_angles(D):
    D = np.round(D, 6)
    return [np.acos(np.real(D[i][i])) if np.abs(np.real(D[i][i])) <= 1 else np.acos(np.sign(np.real(D[i][i]))) for i in range(len(D))]

def construct_unitary(diag, theta, alpha, beta):
    n = len(diag)

    D = np.diag([np.exp(1j * d) for d in diag])
    G_list = []

    theta = list(reversed(theta))
    alpha = list(reversed(alpha))
    beta = list(reversed(beta))

    print('ANGLE\n')
    index = 0
    for j in range(n):
        for i in range(j + 1, n):
            G = np.eye(n, dtype=complex)
            G2 = angle_givens_for(theta[index], alpha[index], beta[index])
            G[np.ix_([j, i], [j, i])] = G2
            G_list.append(G)

            # print(G)
            index += 1

    U_reconstructed = D.copy()
    for G in reversed(G_list):
        U_reconstructed = U_reconstructed @ G

    return np.conj(U_reconstructed)



# --- Step 2: Example run ---
if __name__ == "__main__":
    U = random_unitary(4, seed=42)
    # print(U)
    print("Reconsructed\n")
    U = np.array([[-0.54409413, -0.10171701, -0.25889078,  0.79157488],
 [-0.39583923,  0.59434125, -0.58362529, -0.38658932],
 [ 0.7315188,   0.35371355, -0.41023319,  0.41409624],
 [-0.11024121,  0.71505164,  0.65120336,  0.22908997]])
    U= np.array([[ 0.62923587, -0.27810894,  0.06225542 , 0.32819821,  0.21411186, -0.26067223,
  -0.16945149 , 0.52213037],
 [-0.12832786, -0.72115181 , 0.06330487 ,-0.25546329  ,0.1811286,  -0.15270451,
   0.58015675, -0.03866454],
 [ 0.14146621, 0.33503048 , 0.63506461,  0.14609694, -0.17964295 , 0.2131513,
   0.56863418,  0.20503807],
 [-0.16206701 , 0.04719766 , 0.46284018 ,-0.12025008  ,0.7869089,   0.20527736,
  -0.27689368 ,-0.06921729],
 [ 0.25624259 , 0.38011453 ,-0.34262361, -0.67547194,  0.23405039 , 0.00414033,
   0.21956521 , 0.33644285],
 [ 0.23268996 ,-0.28038369, -0.23106108  ,0.05510674 ,-0.02434206 , 0.89985678,
  -0.00166854 , 0.02183679],
 [ 0.0135536  , 0.25475662 ,-0.4235493,   0.54214077 , 0.46218213, -0.0468658,
   0.43173278, -0.24372693],
 [-0.64909721 ,-0.01092904, -0.15625139 , 0.19494701 , 0.02944304 , 0.09593373,
   0.0091941  , 0.71132259]])
    D, G_list = right_givens_decomposition(U)


    # print(D)
    # Verify reconstruction
    U_reconstructed = D.copy()
    for G in G_list:
        U_reconstructed = U_reconstructed @ G
    print("Max reconstruction error:",
          np.max(np.abs(U - U_reconstructed)))

    print(U_reconstructed)

    diag_angles = get_diag_angles(D)
    thetas, alphas, betas = get_givens_angles(G_list)

    print(f"Diag: {len(diag_angles)}")
    print(f"Thetas: {len(thetas)}")
    print(f"Alphas: {alphas}")
    print(f"Betas: {betas}")

    U_reconstructed_2 = construct_unitary(diag_angles, thetas, alphas, betas)
    print("Max reconstruction error:",
          np.max(np.abs(U - U_reconstructed_2)))
    
    print(U_reconstructed_2)
