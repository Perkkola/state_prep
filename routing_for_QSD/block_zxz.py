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

cnot_1_2 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])

class BlockZXZ(object):
    def __init__(self, coupling_map = None):
        self.coupling_map = coupling_map
        self.gate_queue = deque()
        self.two_qubit_unitary_path = None
        self.diag = None

    def print_circ_unitary(self, qc):
        qc = qc.copy()
        qc.save_unitary()
        simulator = Aer.get_backend('aer_simulator')
        qc = transpile(qc, simulator)

        result = simulator.run(qc).result()
        unitary = result.get_unitary(qc)
        UU = np.asarray(unitary)
        # print("Circuit unitary:\n", U)
        return UU

    def draw_circuit(self, qc, filename=None):
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
        cnot = (cnot[1], cnot[0])
        cnot_base = np.eye(2 ** num_qubits)
        target_mask = 1 << cnot[1]
        for i in range(2 ** num_qubits):
            if (i >> cnot[0]) & 1 == 1: 
                cnot_base[i][i] = 0
                cnot_base[i][i ^ target_mask] = 1

        return cnot_base
    
    def swap_to_adjacent(self, reverse = False):
        new_source = 0 
        if reverse:
            indices = reversed(range(len(self.two_qubit_unitary_path) - 2))
        else: 
            indices = range(len(self.two_qubit_unitary_path) - 2)
        for i in indices:
            q_1 = self.routed_multiplexer.arch_to_grey_map[self.two_qubit_unitary_path[i]]
            q_2 = self.routed_multiplexer.arch_to_grey_map[self.two_qubit_unitary_path[i + 1]]
            new_source = q_2 
            self.gate_queue.append(("SWAP", (q_1, q_2)))

        return new_source


    def decompose_two_qubit_unitary(self, u, rightmost_unitary, leftmost_unitary):
        source = self.routed_multiplexer.grey_to_arch_map[0]
        target = self.routed_multiplexer.grey_to_arch_map[1]

        if self.two_qubit_unitary_path == None : self.two_qubit_unitary_path = list(get_path(self.routed_multiplexer.neighbors, set(self.routed_multiplexer.arch_qubits), source, target))

        new_source = self.swap_to_adjacent(reverse = True)

        if rightmost_unitary == False: u = self.diag @ u
        if not leftmost_unitary: 
            diag_u, gates = extract_diagonal(u, new_source)
            self.diag = diag_u
        else:
            gates = three_cnot_decomposition(u, new_source)
        self.gate_queue.append(gates)

        _ = self.swap_to_adjacent()
    
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

    def compute_decomposition(self, u, init = False, rightmost_unitary = False, leftmost_unitary = False):
        num_qubits = int(math.log2(len(u)))
        target_qubit = num_qubits - 1

        if num_qubits == 2:
            self.decompose_two_qubit_unitary(u, rightmost_unitary, leftmost_unitary)
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
        _, gates_C = routed_multiplexer.map_grey_gates_to_arch(set_last_two_adjacent = False)
        gates_A = routed_multiplexer.replace_mapped_angles(transformed_angles_A)

        if init: self.routed_multiplexer = routed_multiplexer
        print(gates_C)
        while True:
            popped_gate = gates_C.popleft()
            if popped_gate[0] == "RZ":
                gates_C.append(popped_gate)
                break

            gates_A.pop()
            unitary = self.get_cnot_unitary(num_qubits, popped_gate)

            if popped_gate[1] == target_qubit: unitary = np.kron(I, H) @ unitary @ np.kron(I, H) #Position of H might vary

            B_tilde = unitary @ B_tilde @ unitary

        print(B_tilde)
        B_11 = B_tilde[:block_len, :block_len]
        B_22 = B_tilde[block_len:, block_len:]

        V_B, block_diag_B, W_B = self.demultiplex(B_11, B_22)

        single_qubit_unitaries_B = list(extract_single_qubit_unitaries(block_diag_B))
        angles_B = list(extract_angles(single_qubit_unitaries_B))
        transformed_angles_B = möttönen_transformation(angles_B)

        gates_B = routed_multiplexer.replace_mapped_angles(transformed_angles_B, False)

        self.compute_decomposition(V_A, rightmost_unitary = True)
        self.gate_queue.append(gates_A)
        self.compute_decomposition(V_B)
        self.gate_queue.append(("H", target_qubit))
        self.gate_queue.append(gates_B)
        self.gate_queue.append(("H", target_qubit))
        self.compute_decomposition(W_B)
        self.gate_queue.append(gates_C)
        self.compute_decomposition(W_C, leftmost_unitary = True)

if __name__ == "__main__":
    with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

    num_qubits = 3
    # U = generate_U(num_qubits)
    U = np.array([[-0.12145768+0.00207613j, -0.32113457-0.29679609j,  0.47736931+0.02039224j, -0.21817659-0.47238552j,  0.05000757+0.19673254j,  0.08622619+0.39340325j, -0.16360049+0.06469137j, -0.22902861-0.09040058j],
 [ 0.31879968+0.08690366j,  0.21073824-0.39656049j,  0.22336564+0.14242958j,  0.17445736-0.00652138j,  0.30561294-0.09772313j,  0.57015456-0.09594228j,  0.35633014+0.03428243j,  0.14184123+0.05469723j],
 [ 0.36880988-0.04777115j,  0.08540634+0.32329978j, -0.12048505+0.25137414j, -0.12698383-0.40971248j,  0.1201848 -0.44950679j, -0.12119703+0.02088084j,  0.15988069+0.12339056j, -0.46268851+0.04113649j],
 [ 0.18149905-0.03969517j,  0.33711362+0.29644427j,  0.44663305-0.3863337j , -0.04381689-0.10305776j, -0.05250974+0.08701042j, -0.23284859+0.03998177j,  0.26711512+0.16894098j,  0.26068896-0.41071472j],
 [-0.23018696+0.40601539j, -0.21211545-0.13679973j,  0.01115239+0.16527005j, -0.22717664+0.10708209j,  0.13213307-0.04300124j, -0.40017897+0.05719932j,  0.64126169-0.17369612j,  0.05547983+0.02749252j],
 [-0.3546627 +0.04873786j,  0.31627012+0.26864316j,  0.11693279-0.08815455j,  0.10279158-0.3720337j ,  0.40222237+0.1567866j ,  0.01427172-0.18082565j, -0.06089811-0.45041062j,  0.00891598+0.32140178j],
 [ 0.26052849-0.29025084j, -0.1532354 +0.08422461j, -0.41683829+0.14745946j, -0.31996902-0.29263119j,  0.12987615+0.3216078j ,  0.06837351+0.14591191j,  0.097738  -0.12447958j,  0.51053258-0.04285834j],
 [ 0.09605175-0.44260106j, -0.1285551 -0.10376751j,  0.1230578 -0.12473628j,  0.19765844+0.25121693j,  0.53639673-0.12378883j, -0.37812523+0.25945244j, -0.04596453+0.15842672j,  0.07272894+0.29809803j]])
    zxz = BlockZXZ(coupling_map=fake_garnet)
    U = BlockZXZ.get_closest_unitary(U)
    zxz.compute_decomposition(U, init = True)
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
                    if gates[0] == 'H': qc.h(gates[1])
                    else:
                        cx_count += 3 
                        qc.swap(gates[1][0], gates[1][1])

        except Exception as e:
            break

    print(f"CX count: {cx_count}")
    zxz.draw_circuit(qc)
    recon = zxz.print_circ_unitary(qc)
    # print(recon)
    is_equiv, phase = check_equivalence_up_to_phase(U, recon)

    if is_equiv:
        recon_aligned = recon * np.conjugate(phase) # Or recon / phase
        assert np.allclose(U, recon_aligned, atol=1e-8)
        print("Assertion Passed: Matrices match exactly numerically.")
        # print(recon_aligned)
