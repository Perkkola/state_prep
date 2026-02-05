import numpy as np
import math
import json
from utils import extract_single_qubit_unitaries, extract_angles, möttönen_transformation, generate_U, check_equivalence_up_to_phase, get_path
from architecture_aware_routing import RoutedMultiplexer
from collections import deque
from qiskit.circuit import QuantumCircuit
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
        self.original_multiplexers = {}
        self.routed_multiplexers = {}
        self.optimal_multiplexers = {}
        self.swap_maps = {}
        self.swaps_per_level = {}

    def print_circ_unitary(self, qc):
        qc = qc.copy()
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
        plt.show()

    def get_closest_unitary(matrix):
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

        if init: 
            #The multiplexer is only initialized once in the first recursion step. Further recursion steps only make modifications to the original multiplexer (in order to be more efficient).
            routed_multiplexer = RoutedMultiplexer(multiplexer_angles=transformed_angles_C, coupling_map=self.coupling_map) 
            self.routed_multiplexers[0] = routed_multiplexer
            self.total_qubits = num_qubits
        else:
            self.routed_multiplexers[recursion_level].multiplexer_angles = transformed_angles_C

        

        def compute_multiplexers(B_tilde, execute_only = False):
            _, gates_C = self.routed_multiplexers[recursion_level].map_grey_gates_to_arch(set_last_two_adjacent = True, execute_only=execute_only)
            gates_A = self.routed_multiplexers[recursion_level].replace_mapped_angles(transformed_angles_A, True)

            cx_gates_merged = 0
            # while True:
            #     popped_gate = gates_C.pop()
            #     if popped_gate[0] == "RZ" or popped_gate[0] == target_qubit + 1 or popped_gate[1] == target_qubit + 1: #These CNOTs (or RZ) cannot be merged into the neighboring unitaries
            #         gates_C.append(popped_gate)
            #         break

            #     gates_A.popleft()
            #     unitary = self.get_cnot_unitary(num_qubits, popped_gate)

            #     if popped_gate[1] == target_qubit: unitary = np.kron(H, I) @ unitary @ np.kron(H, I) #Position of H might vary

            #     B_tilde = unitary @ B_tilde @ unitary
            #     cx_gates_merged += 1

            B_11 = B_tilde[:block_len, :block_len]
            B_22 = B_tilde[block_len:, block_len:]

            V_B, block_diag_B, W_B = self.demultiplex(B_11, B_22)

            single_qubit_unitaries_B = list(extract_single_qubit_unitaries(block_diag_B))
            angles_B = list(extract_angles(single_qubit_unitaries_B))
            transformed_angles_B = möttönen_transformation(angles_B)

            gates_B = self.routed_multiplexers[recursion_level].replace_mapped_angles(transformed_angles_B, False)
            current_recursion_level_cx_count = (self.routed_multiplexers[recursion_level].cx_count - cx_gates_merged) * 2 + self.routed_multiplexers[recursion_level].cx_count
            path_to_furthest = self.routed_multiplexers[recursion_level].optimal_neighborhood[self.routed_multiplexers[recursion_level].furthest_node]

            return gates_A, gates_C, V_B, gates_B, W_B, path_to_furthest, current_recursion_level_cx_count
        

        skip_swap_search = False
        execute_only = False if init else True

        if self.optimal_multiplexers.get(recursion_level) != None:
            self.routed_multiplexers[recursion_level] = self.optimal_multiplexers[recursion_level].copy() #If the optimal routing has already been found, we use it instead of computing it again
            skip_swap_search = True
        
        gates_A, gates_C, V_B, gates_B, W_B, path_to_furthest, current_recursion_level_cx_count = compute_multiplexers(B_tilde, execute_only)
        if init: 
            self.original_multiplexers[recursion_level] = self.routed_multiplexers[recursion_level].copy() #This is the first multiplexer with no swapping. This is then modified in every recursion step
        else:
            self.routed_multiplexers[recursion_level] = self.original_multiplexers[recursion_level].copy() #Reset

        optimal_cx_count = current_recursion_level_cx_count
        optimal_gates_A = gates_A
        optimal_gates_B = gates_B
        optimal_gates_C = gates_C
        optimal_V_B = V_B
        optimal_W_B = W_B

        if len(path_to_furthest) >= 3 and not skip_swap_search:
            swap_map = {x: x for x in range(self.total_qubits)}
            self.swap_maps[recursion_level] = swap_map
            self.swaps_per_level[recursion_level] = None

            for i in range(len(path_to_furthest) - 2):
                
                original_grey_to_arch = self.routed_multiplexers[recursion_level].grey_to_arch_map.copy()
                new_arch_to_grey = self.routed_multiplexers[recursion_level].arch_to_grey_map.copy()
                new_arch_to_grey_temp = self.routed_multiplexers[recursion_level].arch_to_grey_map.copy()

                for j in range(i + 1):
                    node = path_to_furthest[j]
                    swap_node_to = path_to_furthest[j + 1]
                    new_arch_to_grey[node] = new_arch_to_grey_temp[swap_node_to]
                    new_arch_to_grey[swap_node_to] = new_arch_to_grey_temp[node]
                    new_arch_to_grey_temp = new_arch_to_grey.copy()

                self.routed_multiplexers[recursion_level].root = path_to_furthest[i + 1]
                self.routed_multiplexers[recursion_level].arch_to_grey_map = new_arch_to_grey.copy()
                
                self.routed_multiplexers[recursion_level].recompute_optimal_neighborhood()

                new_grey_to_arch_map = {}
                for key, value in new_arch_to_grey.items():
                    new_grey_to_arch_map[value] = key

                self.routed_multiplexers[recursion_level].grey_to_arch_map = new_grey_to_arch_map.copy()

                gates_A, gates_C, V_B, gates_B, W_B, _, current_recursion_level_cx_count = compute_multiplexers(B_tilde, execute_only=True)
                cx_count_after_swaps = current_recursion_level_cx_count + (2 * (i + 1) * 3) * 3
                if cx_count_after_swaps < optimal_cx_count:
                    self.swap_maps[recursion_level] = {x: self.routed_multiplexers[recursion_level].arch_to_grey_map[original_grey_to_arch[x]] for x in range(num_qubits)} #This is used to map the CNOT controls and targets to correct qubits after emplying SWAP gates.
                    self.swaps_per_level[recursion_level] = path_to_furthest[:i+3] # This is used to place the SWAP gates in the correct place on the circuit.
                    self.optimal_multiplexers[recursion_level] = self.routed_multiplexers[recursion_level].copy() #Store the optimal multiplexer and swap sequence for each recursion level.
                    optimal_cx_count = current_recursion_level_cx_count
                    optimal_gates_A = gates_A
                    optimal_gates_B = gates_B
                    optimal_gates_C = gates_C
                    optimal_V_B = V_B
                    optimal_W_B = W_B

                self.routed_multiplexers[recursion_level] = self.original_multiplexers[recursion_level].copy() #Reset to original after every swap iteration
                
            new_gates_A = deque()
            new_gates_B = deque()
            new_gates_C = deque()

            for gate in optimal_gates_A:
                if gate[0] == "RZ": new_gates_A.append(("RZ", gate[1], swap_map[gate[2]]))
                else: new_gates_A.append((swap_map[gate[0]], swap_map[gate[1]]))
            for gate in optimal_gates_B:
                if gate[0] == "RZ": new_gates_B.append(("RZ", gate[1], swap_map[gate[2]]))
                else: new_gates_B.append((swap_map[gate[0]], swap_map[gate[1]]))
            for gate in optimal_gates_C:
                if gate[0] == "RZ": new_gates_C.append(("RZ", gate[1], swap_map[gate[2]]))
                else: new_gates_C.append((swap_map[gate[0]], swap_map[gate[1]]))

            optimal_gates_A = new_gates_A
            optimal_gates_B = new_gates_B
            optimal_gates_C = new_gates_C
       

        next_routed_multiplexer = self.routed_multiplexers[recursion_level].copy()
        next_num_qubits = num_qubits - 1
        next_target_qubit = target_qubit - 1
        next_root = self.routed_multiplexers[recursion_level].grey_to_arch_map[next_target_qubit]

        next_routed_multiplexer.num_qubits = next_num_qubits
        next_routed_multiplexer.num_controls = next_num_qubits - 1
        next_routed_multiplexer.root = next_root



        next_routed_multiplexer.recompute_optimal_neighborhood()
        self.routed_multiplexers[recursion_level + 1] = next_routed_multiplexer.copy()
        self.original_multiplexers[recursion_level + 1] = next_routed_multiplexer.copy()

        #THIS IS REVERSED FOR A REASON
        #Reason being: we need to decompose the last two qubit unitary first and only then migrate the diagonals through the circuit
        if num_qubits == 3 : 
            self.gate_queue.append((V_A, "V_A"))
        else:
            self.compute_decomposition(V_A, rightmost_unitary = rightmost_unitary, leftmost_unitary=False, recursion_level = recursion_level + 1)

        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = True)
        self.gate_queue.append(optimal_gates_A)
        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = False)

        if num_qubits == 3 : 
            self.gate_queue.append((optimal_V_B, "V_B"))
        else:
            self.compute_decomposition(optimal_V_B, rightmost_unitary = False, leftmost_unitary=False, recursion_level = recursion_level + 1)


        self.gate_queue.append(("H", target_qubit))
        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = True)
        self.gate_queue.append(optimal_gates_B)
        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = False)
        self.gate_queue.append(("H", target_qubit))

        if num_qubits == 3 : 
            self.gate_queue.append((optimal_W_B, "W_B"))
        else:
            self.compute_decomposition(optimal_W_B, rightmost_unitary = False, leftmost_unitary=False,  recursion_level = recursion_level + 1)


        
        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = True)
        self.gate_queue.append(optimal_gates_C)
        if self.swaps_per_level.get(recursion_level) != None: self.swap_to_adjacent(self.swaps_per_level[recursion_level], reverse = False)

        if num_qubits == 3 : 
            self.gate_queue.append((W_C, "W_C"))
        else:
            self.compute_decomposition(W_C, rightmost_unitary = False, leftmost_unitary = leftmost_unitary, recursion_level = recursion_level + 1)
        

if __name__ == "__main__":
    with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

    num_qubits = 4
    U = generate_U(num_qubits)
    zxz = BlockZXZ(coupling_map=fake_garnet)
    zxz.compute_decomposition(U, init = True, rightmost_unitary = True, leftmost_unitary = True)
    qc = QuantumCircuit(num_qubits)

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
            print(e)
            break

    print(f"CX count: {cx_count}")
    zxz.draw_circuit(qc, "fig3.png")
    print(zxz.routed_multiplexers[0])
    print(zxz.routed_multiplexers[1])
    recon = zxz.print_circ_unitary(qc)
    # print(recon)
    is_equiv, phase = check_equivalence_up_to_phase(U, recon)

    if is_equiv:
        recon_aligned = recon * np.conjugate(phase) # Or recon / phase
        assert np.allclose(U, recon_aligned, atol=1e-8)
        print("Assertion Passed: Matrices match exactly numerically.")
        # print(recon_aligned)


