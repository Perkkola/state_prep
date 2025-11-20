import numpy as np
from numpy.linalg import eigh, pinv
from scipy.linalg import sqrtm

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
