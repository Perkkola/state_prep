import numpy as np
from utils import generate_U

class BlockZXZ(object):
    def __init__(self, U):
        self.U = U

    def demultiplex(self, u_1, u_2):
        block_len = len(u_1)
        zeros = np.zeros((block_len, block_len))

        u_1_u_2_dgr = u_1 @ np.conj(u_2.T)

        eigval, eigvec = np.linalg.eig(u_1_u_2_dgr)

        diag = np.diag([np.sqrt(x) for x in eigval])
        V = eigvec
        W = diag @ np.conj(V.T) @ u_2

        block_diag = np.block([[diag, zeros],
                                [zeros, np.conj(diag.T)]])

        return V, block_diag, W

    def compute_decomposition(self, u):
        block_len = len(u) // 2
        X = u[:block_len, :block_len]
        Y = u[:block_len, block_len:]
        u_21 = u[block_len:, :block_len]
        u_22 = u[block_len:, block_len:]

        V_x, sigma_x, W_x_dgr = np.linalg.svd(X)

        S_x = V_x @ np.diag(sigma_x) @ np.conjugate(V_x).T
        U_x = V_x @ W_x_dgr

        V_y, sigma_y, W_y_dgr = np.linalg.svd(Y)

        S_y = V_y @ np.diag(sigma_y) @ np.conjugate(V_y).T
        U_y = V_y @ W_y_dgr


        C = -1j * np.conjugate(U_x).T @ U_y
        A_1 = (S_x + 1j * S_y) @ U_x
        A_2 = u_21 + u_22 @ (1j * np.conjugate(U_y).T @ U_x)

        I = np.eye(block_len)
        B = 2 * np.conjugate(A_1).T @ X - I

        V_A, block_diag_A, W_A = self.demultiplex(A_1, A_2)
        V_B, block_diag_B, W_B = self.demultiplex(I, B)
        V_C, block_diag_C, W_C = self.demultiplex(I, C)

        H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                         [1, -1]])
        

        print("Ready here")
        recon = np.kron(I, V_A) @ block_diag_A @ np.kron(I, W_A) @ np.kron(H, V_B) @ block_diag_B @ np.kron(H, W_B) @ np.kron(I, V_C) @ block_diag_C @ np.kron(I, W_C)

        print(np.allclose(recon, u))



if __name__ == "__main__":
    U = generate_U(13)
    zxz = BlockZXZ(U)
    zxz.compute_decomposition(U)