import numpy as np
import math
from utils import generate_U
import json
from utils import extract_single_qubit_unitaries, extract_angles, möttönen_transformation
from architecture_aware_routing import RoutedMultiplexer
from collections import deque
from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from a_star import BasicAStar
from two_qubit_decomposition import extract_diagonal

class BlockZXZ(object):
    def __init__(self, U, coupling_map = None):
        self.U = U
        self.coupling_map = coupling_map
        self.gate_queue = deque()
        self.two_qubit_unitary_path = None

    def get_cnot_unitary(self, num_qubits, cnot):
        cnot = (cnot[1], cnot[0])
        cnot_base = np.eye(2 ** num_qubits)
        target_mask = 1 << cnot[1]
        for i in range(2 ** num_qubits):
            if (i >> cnot[0]) & 1 == 1: 
                cnot_base[i][i] = 0
                cnot_base[i][i ^ target_mask] = 1

        return cnot_base
    
    def get_subset_of_neighbors(self, neighbors, subset_nodes):
        subset = neighbors.copy()
        for key in neighbors.copy().keys():
            if key not in subset_nodes: subset.pop(key)
            else: subset[key] = subset[key].intersection(subset_nodes)
        return subset
    
    def get_path(self, source, target):
        neighbors = self.get_subset_of_neighbors(self.routed_multiplexer.neighbors, set(self.routed_multiplexer.arch_qubits))
        AStar = BasicAStar(neighbors)
        return AStar.astar(source, target)
    
    def swap_to_adjacent(self):
        for i in range(len(self.two_qubit_unitary_path) - 2):
            q_1 = self.routed_multiplexer.arch_to_grey_map[self.two_qubit_unitary_path[i]]
            q_2 = self.routed_multiplexer.arch_to_grey_map[self.two_qubit_unitary_path[i + 1]]
            self.gate_queue.append(("SWAP", (q_1, q_2)))
    
    def decompose_two_qubit_unitary(self, u):
        source = self.routed_multiplexer.grey_to_arch_map[0]
        target = self.routed_multiplexer.grey_to_arch_map[1]

        if self.two_qubit_unitary_path == None : self.two_qubit_unitary_path = list(self.get_path(source, target))
        if len(self.two_qubit_unitary_path) - 1 > 1: self.swap_to_adjacent()
        extract_diagonal(u)
    
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

    def compute_decomposition(self, u, init = False):
        num_qubits = int(math.log2(len(u)))
        target_qubit = num_qubits - 1

        if num_qubits == 2:
            self.decompose_two_qubit_unitary(u)
            return

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
        I_2 = np.eye(2)

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
        gates_A = routed_multiplexer.replace_mapped_angles(transformed_angles_A)

        if init: self.routed_multiplexer = routed_multiplexer

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

        single_qubit_unitaries_B = list(extract_single_qubit_unitaries(block_diag_B))
        angles_B = list(extract_angles(single_qubit_unitaries_B))
        transformed_angles_B = möttönen_transformation(angles_B)

        gates_B = routed_multiplexer.replace_mapped_angles(transformed_angles_B, False)


        self.compute_decomposition(W_C)
        self.gate_queue.append(gates_C)
        self.compute_decomposition(W_B)
        self.gate_queue.append(("H", target_qubit))
        self.gate_queue.append(gates_B)
        self.gate_queue.append(("H", target_qubit))
        self.compute_decomposition(V_B)
        self.gate_queue.append(gates_A)
        self.compute_decomposition(V_A)

        qc = QuantumCircuit(num_qubits)

        while True:
            try:
                gates = self.gate_queue.popleft()
                match type(gates).__name__:
                    case 'deque':
                        for gate in list(gates):
                            if gate[0] == "RZ": qc.rz(gate[1], target_qubit)
                            else: qc.cx(gate[0], gate[1])
                    case 'ndarray':
                        U = UnitaryGate(gates)
                        qc.append(U, [x for x in range(num_qubits - 1)])
                    case 'tuple':
                        qc.h(gates[1])

            except Exception as e:
                # print(e)
                break
        # print(routed_multiplexer.arch_qubits)
        # print(qc)

        routed_multiplexer.print_circ_unitary(qc)
        print(u)
        # recon = np.kron(I_2, V_A) @ block_diag_A @ np.kron(I_2, W_A) @ np.kron(H, V_B) @ block_diag_B @ np.kron(H, W_B) @ np.kron(I_2, V_C) @ block_diag_C @ np.kron(I_2, W_C)

        # print(np.allclose(recon, u))



if __name__ == "__main__":
    with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

    U = generate_U(3)


    zxz = BlockZXZ(U, coupling_map=fake_garnet)
    # print(zxz.get_cnot_unitary(2, (0, 1)))
    # exit()
    zxz.compute_decomposition(U, init = True)