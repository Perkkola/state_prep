import numpy as np
import math
from utils import generate_U
import json
from utils import extract_single_qubit_unitaries, extract_angles, möttönen_transformation
from architecture_aware_routing import RoutedMultiplexer

class BlockZXZ(object):
    def __init__(self, U, coupling_map = None):
        self.U = U
        self.coupling_map = coupling_map

    def get_cnot_unitary(self, num_qubits, cnot):
        cnot = (cnot[1], cnot[0])
        cnot_base = np.eye(2 ** num_qubits)
        target_mask = 1 << cnot[1]
        for i in range(2 ** num_qubits):
            if (i >> cnot[0]) & 1 == 1: 
                cnot_base[i][i] = 0
                cnot_base[i][i ^ target_mask] = 1

        return cnot_base
    
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
        num_qubits = int(math.log2(len(u)))
        target_qubit = num_qubits - 1

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
        zeros = np.zeros((block_len, block_len))
        B = 2 * np.conjugate(A_1).T @ X - I

        V_A, block_diag_A, W_A = self.demultiplex(A_1, A_2)
        V_C, block_diag_C, W_C = self.demultiplex(I, C)
        B_tilde = np.block([[W_A @ V_C, zeros],
                            [zeros, W_A @ B @ V_C]])

        H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                         [1, -1]])
        

        single_qubit_unitaries_C = list(extract_single_qubit_unitaries(block_diag_C))
        angles_C = list(extract_angles(single_qubit_unitaries_C))
        transformed_angles_C = list(möttönen_transformation(angles_C))

        single_qubit_unitaries_A = list(extract_single_qubit_unitaries(block_diag_A))
        angles_A = list(extract_angles(single_qubit_unitaries_A))
        transformed_angles_A = möttönen_transformation(angles_A)

        routed_multiplexer = RoutedMultiplexer(multiplexer_angles=transformed_angles_C, coupling_map=self.coupling_map)
        _, gates_C = routed_multiplexer.map_grey_gates_to_arch()
        gates_A = routed_multiplexer.reverse_and_replace_mapped_angles(transformed_angles_A)

        while True:
            popped_gate = gates_C.pop()
            if popped_gate[0] == "RZ":
                gates_C.append(popped_gate)
                break

            gates_A.popleft()
            unitary = self.get_cnot_unitary(num_qubits, popped_gate)
            if popped_gate[1] == target_qubit: unitary = np.kron(I, H) @ unitary @ np.kron(I, H) #Position of H might vary

            B_tilde = unitary @ B_tilde @ unitary
        
        B_11 = B_tilde[:block_len, :block_len]
        B_22 = B_tilde[block_len:, block_len:]

        V_B, block_diag_B, W_B = self.demultiplex(B_11, B_22)


        recon = np.kron(I, V_A) @ block_diag_A @ np.kron(I, W_A) @ np.kron(H, V_B) @ block_diag_B @ np.kron(H, W_B) @ np.kron(I, V_C) @ block_diag_C @ np.kron(I, W_C)

        print(np.allclose(recon, u))



if __name__ == "__main__":
    with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

    U = generate_U(3)

    zxz = BlockZXZ(U, coupling_map=fake_garnet)
    # print(zxz.get_cnot_unitary(2, (1, 0)))
    # exit()
    zxz.compute_decomposition(U)