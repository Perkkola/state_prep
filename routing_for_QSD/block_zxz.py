import numpy as np
import math
import json
from utils import extract_single_qubit_unitaries, extract_angles, möttönen_transformation, generate_U, check_equivalence_up_to_phase, get_path
from architecture_aware_routing import RoutedMultiplexer
from collections import deque
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.compiler import transpile
from qiskit_aer import Aer
from two_qubit_decomposition import extract_diagonal, three_cnot_decomposition
import matplotlib.pyplot as plt

class BlockZXZ(object):
    def __init__(self, coupling_map = None):
        self.coupling_map = coupling_map
        self.gate_queue = deque()
        self.two_qubit_unitary_path = None
        self.diag = None
        self.routed_multiplexers = {}
        self.swaps_per_level = {}

        #Figure out swap sequences here and store them in memory. Also store correct multiplexers for each recursion level.

    def print_circ_unitary(self, qc):
        qc = qc.copy()
        # print(qc.count_ops())
        qc = transpile(qc, optimization_level=0, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'ry'])
        qc.save_unitary()
        simulator = Aer.get_backend('aer_simulator')
        qc = transpile(qc, simulator)

        result = simulator.run(qc).result()
        unitary = result.get_unitary(qc)
        UU = np.asarray(unitary)
        # print("Circuit unitary:\n", U)
        return UU

    def draw_circuit(self, qc, filename=None):
        # qc = transpile(qc, optimization_level=0, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'ry'])
        if filename != None:
            fig = qc.draw(output="mpl", interactive=True, filename=filename)
        else:
            fig = qc.draw(output="mpl", interactive=True)
        # plt.show()

    def get_closest_unitary(self, matrix):
        """
        Computes the closest unitary matrix to the input using SVD.
        This fixes the precision loss from copy-pasting.
        """
        V, _, Wh = np.linalg.svd(matrix)
        return V @ Wh

    def get_cnot_unitary(self, num_qubits, cnot):
        cnot = (cnot[0], cnot[1])
        cnot_base = np.eye(2 ** num_qubits)
        target_mask = 1 << cnot[1]
        for i in range(2 ** num_qubits):
            if (i >> cnot[0]) & 1 == 1: 
                cnot_base[i][i] = 0
                cnot_base[i][i ^ target_mask] = 1

        return cnot_base
    
    def swap_to_adjacent(self, path, reverse = False):
        new_source = 0 
        if reverse:
            indices = reversed(range(len(path) - 2))
        else: 
            indices = range(len(path) - 2)
        for i in indices:
            q_1 = self.original_multiplexers[0].arch_to_grey_map[path[i]]
            q_2 = self.original_multiplexers[0].arch_to_grey_map[path[i + 1]]
            new_source = q_2 
            self.gate_queue.append(("SWAP", (q_1, q_2)))

        return new_source


    def decompose_two_qubit_unitary(self, u, rightmost_unitary, leftmost_unitary):
        source = self.original_multiplexers[0].grey_to_arch_map[0]
        target = self.original_multiplexers[0].grey_to_arch_map[1]

        if self.two_qubit_unitary_path == None : self.two_qubit_unitary_path = list(get_path(self.original_multiplexers[0].neighbors, set(self.original_multiplexers[0].arch_qubits), source, target))

        new_source = self.swap_to_adjacent(self.two_qubit_unitary_path, reverse = True)

        if rightmost_unitary == False: u = self.diag @ u
        if not leftmost_unitary: 
            diag_u, gates = extract_diagonal(u, new_source)
            self.diag = diag_u
        else:
            gates = three_cnot_decomposition(u, new_source)
        # gates = three_cnot_decomposition(u, new_source)
        self.gate_queue.append(gates)

        _ = self.swap_to_adjacent(self.two_qubit_unitary_path)
    
    def demultiplex(self, u_1, u_2):
        """
        Demultiplexing procedure from the Block-ZXZ paper.
        Inputs u_1 and u_2 are from the matrix U = U_1 \oplus U_2.
        Outputs matrices V, D \oplus \dagger{D} and W
        """
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

    def compute_decomposition(self, u, init = False, rightmost_unitary = False, leftmost_unitary = False, recursion_level = 0):
        num_qubits = int(math.log2(len(u)))
        target_qubit = num_qubits - 1

        if num_qubits == 2:
            self.decompose_two_qubit_unitary(u, rightmost_unitary, leftmost_unitary)
            return

        #////////////////////////////////////////////////////////////
        #This section is the Block-ZXZ decomposition from:
        # https://arxiv.org/pdf/2403.13692

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
        #/////////////////////////////////////////////////////////////////

        single_qubit_unitaries_C = list(extract_single_qubit_unitaries(block_diag_C))
        angles_C = list(extract_angles(single_qubit_unitaries_C))
        transformed_angles_C = list(möttönen_transformation(angles_C))

        single_qubit_unitaries_A = list(extract_single_qubit_unitaries(block_diag_A))
        angles_A = list(extract_angles(single_qubit_unitaries_A))
        transformed_angles_A = möttönen_transformation(angles_A)

      
        routed_multiplexer = RoutedMultiplexer(multiplexer_angles=transformed_angles_C, coupling_map=self.coupling_map) 
        ###### Get multiplexer from class memory.
        ###### Get correct SWAP sequence also

        _, gates_C = self.routed_multiplexers[recursion_level].map_grey_gates_to_arch(set_last_two_adjacent = True, execute_only=True)
        gates_A = self.routed_multiplexers[recursion_level].replace_mapped_angles(transformed_angles_A, True)
        

        cx_gates_merged = 0
        while True:
            popped_gate = gates_C.pop()
            if popped_gate[0] == "RZ" or popped_gate[0] >= target_qubit + 1 or popped_gate[1] >= target_qubit + 1: #These CNOTs (or RZ) cannot be merged into the neighboring unitaries
                gates_C.append(popped_gate)
                break

            gates_A.popleft()
            unitary = self.get_cnot_unitary(num_qubits, popped_gate)

            if popped_gate[1] == target_qubit: unitary = np.kron(H, I) @ unitary @ np.kron(H, I) #Position of H might vary

            B_tilde = unitary @ B_tilde @ unitary
            cx_gates_merged += 1

        B_11 = B_tilde[:block_len, :block_len]
        B_22 = B_tilde[block_len:, block_len:]

        V_B, block_diag_B, W_B = self.demultiplex(B_11, B_22)

        single_qubit_unitaries_B = list(extract_single_qubit_unitaries(block_diag_B))
        angles_B = list(extract_angles(single_qubit_unitaries_B))
        transformed_angles_B = möttönen_transformation(angles_B)

        gates_B = self.routed_multiplexers[recursion_level].replace_mapped_angles(transformed_angles_B, False)


        #THIS IS REVERSED FOR A REASON
        #Reason being: we need to decompose the last two qubit unitary first and only then migrate the diagonals through the circuit
        self.compute_decomposition(V_A, rightmost_unitary = rightmost_unitary, leftmost_unitary=False, recursion_level = recursion_level + 1)

        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = True)
        self.gate_queue.append(gates_A)
        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = False)

        self.compute_decomposition(V_B, rightmost_unitary = False, leftmost_unitary=False, recursion_level = recursion_level + 1)


        self.gate_queue.append(("H", target_qubit))
        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = True)
        self.gate_queue.append(gates_B)
        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = False)
        self.gate_queue.append(("H", target_qubit))

        self.compute_decomposition(W_B, rightmost_unitary = False, leftmost_unitary=False,  recursion_level = recursion_level + 1)
        
        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = True)
        self.gate_queue.append(gates_C)
        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = False)

        self.compute_decomposition(W_C, rightmost_unitary = False, leftmost_unitary = leftmost_unitary, recursion_level = recursion_level + 1)
        

if __name__ == "__main__":
    with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

    num_qubits = 14
    U = generate_U(num_qubits)
    exit()
    
    zxz = BlockZXZ(coupling_map=fake_garnet)
    zxz.compute_decomposition(U, init = True, rightmost_unitary = True, leftmost_unitary = True)
    qubits = [QuantumRegister(1, name=f'{zxz.routed_multiplexers[0].grey_to_arch_map[i]}') for i in range(num_qubits)]
    qc = QuantumCircuit(*qubits)

    cx_count = 0

    while True:
        try:
            gates = zxz.gate_queue.pop()
            match type(gates).__name__:
                case 'deque':
                    for gate in list(gates):
                        if gate[0] == "RZ": qc.rz(gate[1], gate[2])
                        elif gate[0] == "RY": qc.ry(gate[1], gate[2])
                        elif gate[0] == "RX": qc.rx(gate[1], gate[2])
                        else:
                            cx_count += 1 
                            qc.cx(gate[0], gate[1])
                case 'tuple':
                    if type(gates[0]).__name__ == "ndarray":
                        qc.append(UnitaryGate(gates[0], gates[1]), [0, 1])
                    elif gates[0] == 'H': qc.h(gates[1])
                    elif gates[0] == "CX": qc.cx(gates[1][0], gates[1][1])
                    else:
                        cx_count += 3 
                        qc.swap(gates[1][0], gates[1][1])

        except Exception as e:
            # print(e)
            break

    print(f"CX count: {cx_count}")
    # zxz.draw_circuit(qc, "fig4.png")
    print(zxz.routed_multiplexers.get(0))
    print(zxz.routed_multiplexers.get(1))
    print(zxz.routed_multiplexers.get(2))
    print(zxz.routed_multiplexers.get(3))

    print(zxz.optimal_multiplexers.get(0))
    print(zxz.optimal_multiplexers.get(1))
    print(zxz.optimal_multiplexers.get(2))
    print(zxz.optimal_multiplexers.get(3))
    # print(zxz.swap_maps[0])
    # print(zxz.swap_maps[1])
    # print(zxz.swap_maps[2])
    recon = zxz.print_circ_unitary(qc)
    # print(recon)
    is_equiv, phase = check_equivalence_up_to_phase(U, recon)

    if is_equiv:
        recon_aligned = recon * np.conjugate(phase) # Or recon / phase
        assert np.allclose(U, recon_aligned, atol=1e-8)
        print("Assertion Passed: Matrices match exactly numerically.")
        # print(recon_aligned)


