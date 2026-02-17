import numpy as np
import math
import json
from utils import möttönen_transformation, generate_U, check_equivalence_up_to_phase, get_path, project_to_SU, angles_from_diag, rz, rx, ry
from architecture_aware_routing import RoutedMultiplexer
from collections import deque
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
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
        self.swap_maps = {}

        #Figure out swap sequences here and store them in memory. Also store correct multiplexers for each recursion level.

    def print_circ_unitary_manual(self, gate_queue, num_qubits):
        unitary = np.eye(2 ** num_qubits)
        for gate in gate_queue:
            if gate[0] == "RZ": gate_unitary = self.get_single_qubit_unitary(num_qubits, rz(gate[1]), gate[2])
            elif gate[0] == "RY": gate_unitary = self.get_single_qubit_unitary(num_qubits, ry(gate[1]), gate[2])
            elif gate[0] == "RX": gate_unitary = self.get_single_qubit_unitary(num_qubits, rx(gate[1]), gate[2])
            else:
                gate_unitary = self.get_cnot_unitary(num_qubits, (gate[0], gate[1]))
            unitary = gate_unitary @ unitary
        return unitary
    def print_circ_unitary(self, qc):
        qc = qc.copy()
        # print(qc.count_ops())
        # qc = transpile(qc, optimization_level=0, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'ry'])
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

    def get_single_qubit_unitary(self, num_qubits, unitary, target):
        init = 1
        I = np.eye(2)
        for _ in range(target):
            init = np.kron(I, init)
        init = np.kron(unitary, init)
        for _ in range(target + 1, num_qubits):
            init = np.kron(I, init)
        return init

    def get_cnot_unitary(self, num_qubits, cnot):
        cnot = (cnot[0], cnot[1])
        cnot_base = np.eye(2 ** num_qubits)
        target_mask = 1 << cnot[1]
        for i in range(2 ** num_qubits):
            if (i >> cnot[0]) & 1 == 1: 
                cnot_base[i][i] = 0
                cnot_base[i][i ^ target_mask] = 1

        return cnot_base
    
    def swap_to(self, multiplexer, path, reverse = False):
        if reverse:
            indices = reversed(range(len(path) - 1))
        else: 
            indices = range(len(path) - 1)
        for i in indices:
            q_1 = multiplexer.arch_to_grey_map[path[i]]
            q_2 = multiplexer.arch_to_grey_map[path[i + 1]]
            self.gate_queue.append(("SWAP", (q_1, q_2)))

    def decompose_two_qubit_unitary(self, u, rightmost_unitary, leftmost_unitary):
        if rightmost_unitary == False: u = self.diag @ u
        if not leftmost_unitary: 
            diag_u, gates = extract_diagonal(u, 0)
            self.diag = diag_u
        else:
            gates = three_cnot_decomposition(u, 0)
        # gates = three_cnot_decomposition(u, 0)
        self.gate_queue.append(gates)

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
    
    def initialize_multiplexers(self, num_qubits):
        multiplexer = RoutedMultiplexer(coupling_map=self.coupling_map, num_qubits=num_qubits)
        multiplexer.map_grey_qubits_to_arch_unitary_synth()
        self.original_multiplexer = multiplexer.copy()
        recursion_level = 0
        for i in range(num_qubits, 2, - 1):
            target = multiplexer.grey_to_arch_map[i - 1]
            path_to_root = multiplexer.optimal_neighborhood[target]

            best_cost = multiplexer.get_multiplexer_cost(i, multiplexer.arch_qubits, multiplexer.arch_to_grey_map, target)
            best_arch_to_grey = multiplexer.arch_to_grey_map.copy()
            best_swap_count = 0
            furthest = multiplexer.grey_to_arch_map[0]
            last_cnot_dist = len(list(get_path(multiplexer.neighbors, multiplexer.arch_qubits, target, furthest))) - 1
            best_cost -= ((4 * last_cnot_dist - 4) - 1) if last_cnot_dist > 1 else 1 #Last CNOT can be absorbed into the next unitary.


            arch_to_grey_copy = multiplexer.arch_to_grey_map.copy()
            swap_count = 0
            for j in range(len(path_to_root) - 1, 0, -1):
                temp = arch_to_grey_copy[path_to_root[j]]
                arch_to_grey_copy[path_to_root[j]] = arch_to_grey_copy[path_to_root[j - 1]]
                arch_to_grey_copy[path_to_root[j - 1]] = temp
                swap_count += 1
                current_cost = multiplexer.get_multiplexer_cost(i, multiplexer.arch_qubits, arch_to_grey_copy, path_to_root[j - 1])
                last_cnot_dist = len(list(get_path(multiplexer.neighbors, multiplexer.arch_qubits, path_to_root[j - 1], furthest))) - 1
                current_cost -= ((4 * last_cnot_dist - 4) - 1) if last_cnot_dist > 1 else 1 
                current_cost += swap_count * 3 * 2

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_arch_to_grey = arch_to_grey_copy.copy()
                    best_swap_count = swap_count



            
            current_level_multiplexer = multiplexer.copy()
            current_level_multiplexer.arch_to_grey_map = best_arch_to_grey
            current_level_multiplexer.grey_to_arch_map = {item: key for key, item in best_arch_to_grey.items()}
            current_level_multiplexer.root = current_level_multiplexer.grey_to_arch_map[i - 1]
            current_level_multiplexer.furthest_node = current_level_multiplexer.grey_to_arch_map[0]
            current_level_multiplexer.recompute_optimal_neighborhood()

            if best_swap_count == 0: 
                self.swaps_per_level[recursion_level] = None
                self.swap_maps[recursion_level] = None
            else: 
                self.swaps_per_level[recursion_level] = path_to_root[len(path_to_root) - 1 - best_swap_count:]
                self.swap_maps[recursion_level] = {x: current_level_multiplexer.arch_to_grey_map[multiplexer.grey_to_arch_map[x]] for x in range(i)}


            self.routed_multiplexers[recursion_level] = current_level_multiplexer

            multiplexer.num_qubits = multiplexer.num_qubits - 1
            multiplexer.num_controls = multiplexer.num_controls - 1
            value = multiplexer.grey_to_arch_map.pop(i - 1)
            multiplexer.arch_to_grey_map.pop(value)
            multiplexer.root = multiplexer.grey_to_arch_map[i - 2]
            multiplexer.optimal_neighborhood.pop(value)

            recursion_level += 1
        # exit()

    def compute_decomposition(self, u, init = False, rightmost_unitary = False, leftmost_unitary = False, recursion_level = 0):
        num_qubits = int(math.log2(len(u)))
        target_qubit = num_qubits - 1

        if init: self.initialize_multiplexers(num_qubits)

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

        angles_C = angles_from_diag(block_diag_C)
        transformed_angles_C = list(möttönen_transformation(angles_C))
        # print(transformed_angles_C)

        angles_A = angles_from_diag(block_diag_A)
        transformed_angles_A = möttönen_transformation(angles_A)

      
        routed_multiplexer = self.routed_multiplexers[recursion_level].copy()
        routed_multiplexer.multiplexer_angles = transformed_angles_C

        _, gates_C = routed_multiplexer.execute_gates(execute_only=True)
        gates_A = routed_multiplexer.replace_mapped_angles(transformed_angles_A, True)
        # print(gates_C)
        # print(gates_A)
        
        # cx_gates_merged = 0
        # while True:
        #     popped_gate = gates_C.pop()
        #     if popped_gate[0] == "RZ" or popped_gate[0] >= target_qubit + 1 or popped_gate[1] >= target_qubit + 1 or popped_gate[1] == 1 or popped_gate[0] == 1: #These CNOTs (or RZ) cannot be merged into the neighboring unitaries
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

        angles_B = angles_from_diag(block_diag_B)
        transformed_angles_B = möttönen_transformation(angles_B)

        gates_B = routed_multiplexer.replace_mapped_angles(transformed_angles_B, False)

        if self.swap_maps[recursion_level] != None:
            new_gates_A = deque()
            new_gates_B = deque()
            new_gates_C = deque()
            swap_map = self.swap_maps[recursion_level]
            for gate in gates_A:
                if gate[0] != "RZ":
                    new_gates_A.append((swap_map[gate[0]], swap_map[gate[1]]))
                else: new_gates_A.append((gate[0], gate[1], swap_map[gate[2]]))
            for gate in gates_B:
                if gate[0] != "RZ":
                    new_gates_B.append((swap_map[gate[0]], swap_map[gate[1]]))
                else: new_gates_B.append((gate[0], gate[1], swap_map[gate[2]]))
            for gate in gates_C:
                if gate[0] != "RZ":
                    new_gates_C.append((swap_map[gate[0]], swap_map[gate[1]]))
                else: new_gates_C.append((gate[0], gate[1], swap_map[gate[2]]))

            gates_A = new_gates_A
            gates_B = new_gates_B
            gates_C = new_gates_C

        #THIS IS REVERSED FOR A REASON
        #Reason being: we need to decompose the last two qubit unitary first and only then migrate the diagonals through the circuit
        self.compute_decomposition(V_A, rightmost_unitary = rightmost_unitary, leftmost_unitary=False, recursion_level = recursion_level + 1)

        if self.swaps_per_level[recursion_level] != None: self.swap_to(routed_multiplexer, self.swaps_per_level[recursion_level], reverse = True)
        # self.gate_queue.append((block_diag_A, "A", range(num_qubits)))
        if recursion_level == 0: 
            self.gate_queue.append((gates_A, "A"))

            print(np.array([block_diag_A[i][i] for i in range(len(block_diag_A))]))
            un = self.print_circ_unitary_manual(gates_A, num_qubits)
            print(np.array([un[i][i] for i in range(len(un))]))
            # exit()
        self.gate_queue.append(gates_A)
        if self.swaps_per_level[recursion_level] != None: self.swap_to(routed_multiplexer, self.swaps_per_level[recursion_level], reverse = False)

        self.compute_decomposition(V_B, rightmost_unitary = False, leftmost_unitary=False, recursion_level = recursion_level + 1)


        self.gate_queue.append(("H", target_qubit))
        if self.swaps_per_level[recursion_level] != None: self.swap_to(routed_multiplexer, self.swaps_per_level[recursion_level], reverse = True)
        self.gate_queue.append(gates_B)
        # self.gate_queue.append((block_diag_B, "B", range(num_qubits)))
        if self.swaps_per_level[recursion_level] != None: self.swap_to(routed_multiplexer, self.swaps_per_level[recursion_level], reverse = False)
        self.gate_queue.append(("H", target_qubit))

        self.compute_decomposition(W_B, rightmost_unitary = False, leftmost_unitary=False,  recursion_level = recursion_level + 1)
        
        if self.swaps_per_level[recursion_level] != None: self.swap_to(routed_multiplexer, self.swaps_per_level[recursion_level], reverse = True)
        # self.gate_queue.append((block_diag_C, "C", range(num_qubits)))
        self.gate_queue.append(gates_C)
        if self.swaps_per_level[recursion_level] != None: self.swap_to(routed_multiplexer, self.swaps_per_level[recursion_level], reverse = False)

        self.compute_decomposition(W_C, rightmost_unitary = False, leftmost_unitary = leftmost_unitary, recursion_level = recursion_level + 1)
        

if __name__ == "__main__":
    with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

    num_qubits = 6
    U = generate_U(num_qubits)
    SU, _ = project_to_SU(U, 2 ** num_qubits)
    
    zxz = BlockZXZ(coupling_map=fake_garnet)
    zxz.compute_decomposition(SU, init = True, rightmost_unitary = True, leftmost_unitary = True)
    qubits = [QuantumRegister(1, name=f'{zxz.original_multiplexer.grey_to_arch_map[i]}') for i in range(num_qubits)]
    qc = QuantumCircuit(*qubits)

    qc_test = QuantumCircuit(num_qubits)
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
                        # qc.append(UnitaryGate(gates[0], gates[1]), [0, 1])
                        qc.append(UnitaryGate(gates[0], gates[1]), gates[2])
                    elif type(gates[0]).__name__ == "deque":
                        for gate in reversed(list(gates[0])):
                            if gate[0] == "RZ": qc_test.rz(gate[1], gate[2])
                            elif gate[0] == "RY": qc_test.ry(gate[1], gate[2])
                            elif gate[0] == "RX": qc_test.rx(gate[1], gate[2])
                            else:
                                cx_count += 1 
                                qc_test.cx(gate[0], gate[1])
                    elif gates[0] == 'H': qc.h(gates[1])
                    elif gates[0] == "CX": qc.cx(gates[1][0], gates[1][1])
                    else:
                        cx_count += 3 
                        qc.swap(gates[1][0], gates[1][1])

        except Exception as e:
            print(e)
            break

    # print(f"CX count: {cx_count}")
    # zxz.draw_circuit(qc_test, "fig6_test.png")
    # print(zxz.routed_multiplexers.get(0))
    # print(zxz.routed_multiplexers.get(1))
    # print(zxz.routed_multiplexers.get(2))
    # print(zxz.routed_multiplexers.get(3))
    # st_1 = Statevector(qc)
    # unitary_circ = QuantumCircuit(num_qubits)
    # unitary_circ.append(UnitaryGate(SU), range(num_qubits))
    # st_2 = Statevector(unitary_circ)
    # print(f"Physical equality: {st_1.equiv(st_2)}")

    # recon = zxz.print_circ_unitary(qc)
    recon = zxz.print_circ_unitary(qc_test)
    # print(print(np.array([recon[i][i] for i in range(len(recon))])))
    count = 0
    for i in range(len(recon)):
        for j in range(len(recon)):
            if recon[i][j] != 0: count+=1
    print(count)
    # print(*recon, sep="\n")
    zxz.draw_circuit(qc_test)
    exit()
    is_equiv, phase = check_equivalence_up_to_phase(SU, recon)

    if is_equiv:
        recon_aligned = recon * np.conjugate(phase) # Or recon / phase
        assert np.allclose(SU, recon_aligned, atol=1e-5)
        print("Assertion Passed: Matrices match exactly numerically.")
        # print(recon_aligned)


