import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.linalg import det
from utils import generate_U
# --- 1. SETUP GATES AND UTILS ---
# SU(4) CNOT (Det = 1)
# Phase xi s.t. xi^4 = -1 => xi = exp(i*pi/4)
xi = np.exp(1j * np.pi / 4)
cnot_std = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
CNOT = xi * cnot_std 

I = np.eye(2)
# Standard Paulis
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
Sy = np.array([[0, -1j], [1j, 0]])
SySy = np.kron(Sy, Sy)

# Magic Basis E
E = np.array([[1, 1j, 0, 0],
              [0, 0, 1j, 1],
              [0, 0, 1j, -1],
              [1, -1j, 0, 0]]) / np.sqrt(2)

def gamma_map(u):
    return u @ SySy @ u.T @ SySy

def rx(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], 
                     [-1j*np.sin(theta/2), np.cos(theta/2)]])

def rz(theta):
    return np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2)])

def project_to_SU4(U):
    detU = np.linalg.det(U)

    assert detU != 0, "Matrix is not unitary!"

    return U / detU ** (1 / 4)
def orthogonal_congruence_diagonalize(S):
    # Decomposes symmetric S = A @ D @ A.T where A is orthogonal
    # Returns A, D
    R = np.real(S)
    ImS = np.imag(S)
    eigvals, Q = np.linalg.eigh(R)
    
    # Handle degeneracies (simplified block diag)
    # This is a critical step for stability.
    # For this specific problem, we often have distinct eigenvalues, 
    # but a robust implementation handles clusters. 
    # (Using the user's previous implementation logic is fine, 
    # or a simplified one if spectra are simple).
    # Here we assume simple spectrum for brevity, or rely on 'eigh'
    # sorting. For production, full block diag is needed.
    
    # We will refine Q to diagonalize ImS in degenerate subspaces
    # but for now let's try the direct approach which works for random U.
    
    A = Q
    D = A.T @ S @ A
    return A, D

# --- 2. MAIN ALGORITHM ---

# Step A: Normalize Input
# (Assuming 'random_U_prime' and 'cnot_1_2' exist in your scope)
# For testing, let's generate a dummy one if needed, or use yours.
# U_prime = random_U_prime / (det(random_U_prime)**0.25)
# U = U_prime @ CNOT

# --- START: Paste this block after defining U ---
# Ensure U is SU(4)
random_U_prime = project_to_SU4(generate_U(2))

U = random_U_prime @ CNOT
U = U / (np.linalg.det(U)**0.25)

# Step B: Find Psi
M = gamma_map(U.T).T
t = np.diag(M)
# Swapped pairing (0+1, 2+3) vs (0+3, 1+2)
# Try the user's successful swap: t2 <-> t4 logic
num = np.imag(t[0] + t[1] + t[2] + t[3])
den = np.real(t[0] + t[3] - t[2] - t[1])
psi = np.arctan2(num, den)

# Step C: Target Spectrum
Delta = CNOT @ np.kron(I, rz(psi)) @ CNOT
g_target = gamma_map(U @ Delta)
target_evals = np.linalg.eigvals(g_target)
target_angles = np.sort(np.angle(target_evals))

# Step D: OPTIMIZE Theta, Phi
# We solve for theta, phi that match the spectrum.
def objective(params):
    th, ph = params
    # Kernel with Non-Commuting Gates (Rx on Control, Rz on Target)
    # CNOT is Control-0, Target-1
    K = CNOT @ np.kron(rx(th), rz(ph)) @ CNOT
    g_K = gamma_map(K)
    k_evals = np.linalg.eigvals(g_K)
    k_angles = np.sort(np.angle(k_evals))
    return np.sum((k_angles - target_angles)**2)

# Initial guess from analytical formula
# Heuristic mapping
r = target_angles[-1]
s = target_angles[-2]
x0 = [(r+s)/2, (r-s)/2]

res = minimize(objective, x0, method='Nelder-Mead', tol=1e-6)
theta_opt, phi_opt = res.x

print(f"Optimization Success: {res.success}")
print(f"Spectrum Error: {res.fun:.2e}")

# Step E: Construct Matrices
kernel = CNOT @ np.kron(rx(theta_opt), rz(phi_opt)) @ CNOT

k_E = np.conjugate(E).T @ kernel @ E
S_k = k_E @ k_E.T

U_E = np.conjugate(E).T @ U @ np.kron(I, rz(psi)) @ CNOT @ E
S_U = U_E @ U_E.T

A_U, D_U = orthogonal_congruence_diagonalize(S_U)
B_k, D_k = orthogonal_congruence_diagonalize(S_k)

# Step F: Robust Alignment
def align_eigenvalues_robust(D_tar, D_ker, B_ker):
    d_tar = np.diag(D_tar)
    d_ker = np.diag(D_ker)
    # Match phases
    cost = np.abs(np.angle(d_tar)[:, None] - np.angle(d_ker)[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    P = np.eye(4)[:, col_ind]
    return B_ker @ P

B_k_aligned = align_eigenvalues_robust(D_U, D_k, B_k)

# Step G: Compute C
C = np.conjugate(k_E).T @ B_k_aligned @ A_U.T @ U_E

# Step H: Determinant Correction
det_C = np.linalg.det(C)
if np.real(det_C) < 0:
    print("Det is -1. Flipping one eigenvector column to fix...")
    # Flip the sign of the first column of B_k_aligned
    B_k_aligned[:, 0] *= -1
    # Recompute C
    C = np.conjugate(k_E).T @ B_k_aligned @ A_U.T @ U_E

# --- VERIFICATION ---
print("\nFinal Results:")
print(f"Is C Real? {np.allclose(np.imag(C), 0, atol=1e-5)}")
print(f"Is C Orth? {np.allclose(C @ C.T, np.eye(4), atol=1e-5)}")
print(f"Det C: {np.linalg.det(C):.4f}")

if np.allclose(np.imag(C), 0, atol=1e-5) and np.allclose(C @ C.T, np.eye(4), atol=1e-5):
    print("\nSUCCESS! Matrix C is in SO(4).")
    print("You can now decompose C into local gates (a x b) using KAK/polar decomposition.")

print(C)