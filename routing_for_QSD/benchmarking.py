from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UCGate, Permutation
from qiskit.compiler import transpile
from qiskit_aer import Aer, AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh, FakeCairoV2
from qiskit_ibm_runtime import QiskitRuntimeService
from pytket.extensions.qiskit import IBMQBackend
from pytket.extensions.iqm import IQMBackend
from pytket import Circuit, OpType
from pytket.circuit import MultiplexedRotationBox
from pytket.passes import DecomposeBoxes
import numpy as np
import math 
import sys
import json
import time
from collections import deque
from utils import get_grey_gates, generate_random_rz_multiplexer_unitary, extract_single_qubit_unitaries, extract_angles, clean_matrix, is_unitary, möttönen_transformation
from architecture_aware_routing import RoutedMultiplexor
from pauliopt.phase.phase_circuits import PhaseGadget, PhaseCircuit, Z, X
from pauliopt.phase.optimized_circuits import OptimizedPhaseCircuit
from pauliopt.topologies import Topology
from pauliopt.utils import pi, Angle
import matplotlib.pyplot as plt
import networkx as nx
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

IQM_TOKEN = "3BGYdJuhl8pAeYUNcJ3b9r6r2CtXuGPHdjwA1StOaGUBmnyXrB9+EKROdY4Q4p4w"
IBM_TOKEN = 'u_k47sVdjZ8QvxY5bUj-PQRM2h_CNU8jcIh6RxJG_7Jg'
IBM_INSTANCE = 'crn:v1:bluemix:public:quantum-computing:us-east:a/14c391407c20401a8dc347e17f51ba83:7d2deb88-e3a6-4315-8ca8-e936c1009807::'

class QiskitBase(object):
    def __init__(self, num_qubits, coupling_map = None, multiplexer = None, print_unitary = False, reverse = False):
        self.num_qubits = num_qubits
        self.num_controls = self.num_qubits - 1
        self.coupling_map = coupling_map
        self.multiplexer = multiplexer
        self.print_unitary = print_unitary
        self.reverse = reverse

    def random_rz_generator(self):
        for _ in range(2 ** self.num_controls):
            phi = np.random.random() * 2 * np.pi - np.pi
            yield np.array([[math.cos(phi) - 1j*math.sin(phi), 0],
                        [0, math.cos(phi) + 1j*math.sin(phi)]])

    def count_cx(self, qc):
        return qc.count_ops()['cx'] if qc.count_ops().get('cx') != None else 0

    def print_circ_unitary(self, qc):
        qc = qc.copy()
        qc.save_unitary()
        simulator = Aer.get_backend('aer_simulator')
        qc = transpile(qc, simulator)

        result = simulator.run(qc).result()
        unitary = result.get_unitary(qc)
        print("Circuit unitary:\n", np.asarray(unitary).round(5))

    def draw_circuit(self, qc):
        fig = qc.draw(output="mpl", interactive=True)
        plt.show()
        plt.close(fig)

    def draw_backend(self):
        if self.coupling_map == None:
            print("No backend to draw!")
            return

        G = nx.Graph()
        G.add_edges_from(self.coupling_map)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()
        
        


class QiskitNative(QiskitBase):
    def generate_circuit(self, mux_simp = False):
        if self.multiplexer == None:
            rz_unitaries = list(self.random_rz_generator())
        else:
            rz_unitaries = self.multiplexer

        qc = QuantumCircuit(self.num_qubits)
        ucg = UCGate(rz_unitaries, mux_simp=mux_simp)
        qc.append(ucg, [x for x in reversed(range(self.num_qubits))])


        qc = qc.decompose(reps=4)
        if self.print_unitary: self.print_circ_unitary(qc)

        if self.coupling_map != None:
            qc_opt = transpile(qc, optimization_level=3, coupling_map=self.coupling_map, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'ry'])
        else:
            qc_opt = transpile(qc, optimization_level=3, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'ry'])

        qc_opt = qc_opt.decompose(reps=4)
        return qc_opt
    



class QiskitGray(QiskitBase):
    def generate_circuit(self):

        if self.multiplexer != None:
            im_lazy = RoutedMultiplexor(multiplexer_angles=self.multiplexer, coupling_map=None, reverse=self.reverse)
            _, grey_gates = im_lazy.map_grey_gates_to_arch()
            qc = QuantumCircuit(self.num_qubits)
            for gate in list(grey_gates):
                if gate[0] == "RZ":
                    if self.reverse: qc.rz(gate[1], self.num_controls)
                    else: qc.rz(gate[1], 0)
                else:
                    if self.reverse: qc.cx(gate[0], gate[1])
                    else: qc.cx(self.num_controls - gate[0], self.num_controls - gate[1])

        else:
            grey_gates = get_grey_gates(self.num_controls, False, True)
            qc = QuantumCircuit(self.num_qubits)
            for gate in grey_gates:
                qc.rz(np.random.random() * 2 * np.pi - np.pi, self.num_controls)
                qc.cx(gate[0], gate[1])

        if self.print_unitary: self.print_circ_unitary(qc)
        qc = qc.decompose(reps=4)

        if self.coupling_map != None:
            qc = transpile(qc, optimization_level=3, coupling_map=self.coupling_map, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'ry'])

        qc = qc.decompose(reps=4)
        return qc


class SteinerSynth(object):
    def __init__(self, num_qubits, coupling_map = None, multiplexer = None, reverse= False):
        self.num_qubits = num_qubits
        self.num_controls = self.num_qubits - 1
        self.coupling_map = coupling_map
        self.multiplexer = multiplexer
        self.reverse = reverse

    def get_qiskit_circ(self, circuit, *args, **kwargs):
        draw_kwargs = {"output": "mpl", "interactive": True}
        qc = circuit.to_qiskit(*args, **kwargs)
        if qc.metadata:
            if "initial_layout" in qc.metadata:
                init_mapping = qc.metadata["initial_layout"]
                qc.compose(Permutation(qc.num_qubits, init_mapping), front=True, inplace=True)
            if "final_layout" in qc.metadata:
                final_mapping = qc.metadata["final_layout"]
                qc.compose(Permutation(qc.num_qubits, final_mapping), front=False, inplace=True)
        return qc
    
    def generate_circuit(self, grey_to_arch_map = None):
        circuit = PhaseCircuit(self.num_qubits)
        shift = True

        if self.coupling_map == None:
            topology = Topology.complete(self.num_qubits)
        else:
            shifted_coupling_map = []
            num_arch_qubits = 0

            for edge in self.coupling_map:
                if edge[0] == 0 or edge[1] == 0: 
                    shift = False
                if edge[0] > num_arch_qubits: num_arch_qubits = edge[0]
                if edge[1] > num_arch_qubits: num_arch_qubits = edge[1]
                shifted_coupling_map.append([edge[0] - 1, edge[1] - 1])
            
            if shift: self.coupling_map = shifted_coupling_map
            else: num_arch_qubits += 1

            topology = Topology(num_arch_qubits, list(self.coupling_map))
            circuit = PhaseCircuit(num_arch_qubits)

        _, state_queue = get_grey_gates(self.num_controls, state_queue=True)

        if self.multiplexer != None:
            im_lazy = RoutedMultiplexor(multiplexer_angles=self.multiplexer, coupling_map=None, reverse=self.reverse)
            _, grey_gates = im_lazy.map_grey_gates_to_arch()
            rz_gates = deque(list(filter(lambda gate: gate[0] == "RZ", grey_gates)))
        else: rz_gates = None

        for state in state_queue:
            gadget_qubits = set()
            for i in range(self.num_qubits):
                if (state >> i) & 1 == 1 and grey_to_arch_map != None: gadget_qubits.add(grey_to_arch_map[i] - 1 if shift else grey_to_arch_map[i])
                elif (state >> i) & 1 == 1: gadget_qubits.add(i)
            if rz_gates != None: circuit >>= Z(Angle(rz_gates.popleft()[1])) @ gadget_qubits
            else: circuit >>= Z(pi / 5) @ gadget_qubits
        opt = OptimizedPhaseCircuit(circuit.copy(), topology, 3, phase_method="steiner-graysynth", cx_method="permrowcol",reallocate=True)
        
        qc = self.get_qiskit_circ(opt, topology)
        return qc
    
    def count_cx(self, qc):
        return qc.count_ops()['cx'] if qc.count_ops().get('cx') != None else 0
    
    def draw_circuit(self, qc):
        fig = qc.draw(output="mpl", interactive=True)
        plt.show()
        plt.close(fig)
    

class TketBase(object):
    def __init__(self, num_qubits, backend = None, multiplexer = None, print_unitary = False, reverse=False):
        self.num_qubits = num_qubits
        self.num_controls = self.num_qubits - 1
        self.backend = backend
        self.multiplexer = multiplexer
        self.print_unitary = print_unitary
        self.reverse = reverse
        self._get_backend()

    
    def _get_backend(self):
        match self.backend:
            case "fez":
                ibm_token = IBM_TOKEN
                inst = IBM_INSTANCE
                QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=ibm_token, instance=inst, overwrite=True)
                self.backend = IBMQBackend("ibm_fez")
            case "marrakesh":
                ibm_token = IBM_TOKEN
                inst = IBM_INSTANCE
                QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=ibm_token, instance=inst, overwrite=True)
                self.backend = IBMQBackend("ibm_marrakesh")
            case "garnet":
                iqm_token = IQM_TOKEN
                self.backend = IQMBackend('garnet', api_token=iqm_token)
            case "emerald":
                iqm_token = IQM_TOKEN
                self.backend = IQMBackend('emerald', api_token=iqm_token)
            case _:
                self.backend = None

    def count_cx(self, qc):
        cx_count = qc.n_gates_of_type(OpType.CX)
        if cx_count == 0: return qc.n_gates_of_type(OpType.CZ)
        else: return cx_count

    def random_angle_generator(self):
        for i in range(2 ** self.num_controls):
            if i % 2 == 0:
                phi = np.random.random() * 2 * np.pi - np.pi
                yield phi
            else:
                yield phi
    
class TKetGray(TketBase):
    def generate_circuit(self):
        if self.multiplexer != None:
            im_lazy = RoutedMultiplexor(multiplexer_angles=self.multiplexer, coupling_map=None, reverse=self.reverse)
            _, grey_gates = im_lazy.map_grey_gates_to_arch()
            qc = Circuit(self.num_qubits)
            for gate in list(grey_gates):
                if gate[0] == "RZ":
                    if self.reverse: qc.Rz((gate[1]) / np.pi, self.num_controls)
                    else: qc.Rz((gate[1]) / np.pi, 0)
                else:
                    if self.reverse: qc.CX(gate[0], gate[1])
                    else: qc.CX(self.num_controls - gate[0], self.num_controls - gate[1])

        else:
            grey_gates = get_grey_gates(self.num_controls, False, True)
            qc = Circuit(self.num_qubits)
            for gate in grey_gates:
                qc.Rz(np.random.random() * 2 * np.pi - np.pi, self.num_controls)
                qc.CX(gate[0], gate[1])

        if self.print_unitary: print(clean_matrix(qc.get_unitary()))
        
        if self.backend != None:
            self.backend.default_compilation_pass(optimisation_level=2).apply(qc)

        return qc

class TketNative(TketBase):
    def generate_circuit(self):
        qc = Circuit(self.num_qubits)

        if self.multiplexer == None:
            angles = list(self.random_angle_generator())
        else:
            angles = self.multiplexer
            angles = [(2 * x) / np.pi for x in angles]

        multiplexor = MultiplexedRotationBox(angles, OpType.Rz)
        qc.add_gate(multiplexor, [x for x in range(self.num_qubits)][::-1])

        if self.print_unitary: print(clean_matrix(qc.get_unitary()))

        if self.backend != None:
            self.backend.default_compilation_pass(optimisation_level=2).apply(qc)

        return qc
    

if __name__ == "__main__":
    
    num_qubits = int(sys.argv[1])
    assert num_qubits > 1

    multiplexor_unitary = generate_random_rz_multiplexer_unitary(num_qubits)
    print(multiplexor_unitary)

    single_qubit_unitaries = list(extract_single_qubit_unitaries(multiplexor_unitary))
    angles = list(extract_angles(single_qubit_unitaries))
    transformed_angles = list(möttönen_transformation(angles))

    coupling_map = None
    backend = None

    match sys.argv[2] if len(sys.argv) > 2 else "":
        case "fez":
            fake_fez = FakeFez()
            coupling_map = fake_fez.coupling_map
            backend = "fez"
        case "marrakesh":
            fake_marrakesh = FakeMarrakesh()
            coupling_map = fake_marrakesh.coupling_map
            backend = "marrakesh"
        case "cairo":
            fake_cairo = FakeCairoV2()
            coupling_map = fake_cairo.coupling_map
            backend = None
        case "garnet":
            with open(f"./coupling_maps/fake_garnet.json") as f:
                coupling_map = json.load(f)
            backend = "garnet"
        case "emerald":
            with open(f"./coupling_maps/fake_emerald.json") as f:
                coupling_map = json.load(f)
            backend = "emerald"
        case _:
            coupling_map = None
            backend = None

    # print(multiplexor_unitary)
    # print("///////////////////")
    qn = QiskitNative(num_qubits, coupling_map=coupling_map, multiplexer=single_qubit_unitaries, print_unitary= True)
    qc = qn.generate_circuit(mux_simp=False)
    qn_cx_count = qn.count_cx(qc)

    qg = QiskitGray(num_qubits, coupling_map=coupling_map, multiplexer=transformed_angles, print_unitary=True, reverse=True)
    qc2 = qg.generate_circuit()
    qg_cx_count = qg.count_cx(qc2)


    proposed = RoutedMultiplexor(multiplexer_angles=transformed_angles, coupling_map=coupling_map, num_qubits=num_qubits)
    p_cx_count = proposed.execute_gates()
    qc_p = proposed.get_circuit()
    proposed.print_circ_unitary(qc_p)
    grey_to_arch_map = proposed.grey_to_arch_map

    sg = SteinerSynth(num_qubits, coupling_map, multiplexer= transformed_angles)
    qc3 = sg.generate_circuit(grey_to_arch_map=grey_to_arch_map)
    sg_cx_count = sg.count_cx(qc3)
    
    tg = TKetGray(num_qubits, backend=backend, multiplexer=transformed_angles, print_unitary=True)
    qc4 = tg.generate_circuit()
    tg_cx_count = tg.count_cx(qc4)

    tn = TketNative(num_qubits, backend=backend, multiplexer=angles, print_unitary= True)
    qc5 = tn.generate_circuit()
    tn_cx_count = tn.count_cx(qc5)


    
    print(f"Qiskit native UCGate CX count: {qn_cx_count}")
    print(f"Qiskit with gray code multiplexor CX count: {qg_cx_count}")
    print(f"Steiner-Gray CX count: {sg_cx_count}")
    print(f"Tket with gray code CX count: {tg_cx_count}")
    print(f"Tket native Multiplexor CX count: {tg_cx_count}")
    print(f"Proposed method CX count: {p_cx_count}")