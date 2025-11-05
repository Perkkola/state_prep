from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UCGate
from qiskit.compiler import transpile
from qiskit_aer import Aer, AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
import numpy as np
import math 
import sys
import json
from utils import get_grey_gates
from architecture_aware_routing import RoutedMultiplexor

class QiskitBase(object):
    def __init__(self, num_qubits, coupling_map = None):
        self.num_qubits = num_qubits
        self.num_controls = self.num_qubits - 1
        self.coupling_map = coupling_map

    def random_rz_generator(self):
        for _ in range(2 ** self.num_controls):
            phi = np.random.random() * np.pi - 0.00123
            yield np.array([[math.cos(phi / 2) - 1j*math.sin(phi / 2), 0],
                            [0, math.cos(phi / 2) + 1j*math.sin(phi / 2)]])
    def count_cx(self, qc):
        ops = qc.count_ops()
        return ops['cx']

    def print_unitary(self, qc):
        qc.save_unitary()
        simulator = Aer.get_backend('aer_simulator')
        qc = transpile(qc, simulator)

        result = simulator.run(qc).result()
        unitary = result.get_unitary(qc)
        print("Circuit unitary:\n", np.asarray(unitary).round(5))


class QiskitNative(QiskitBase):
    def generate_circuit(self, mux_simp = False, routed = False):
        random_rzs = list(self.random_rz_generator())
        qc = QuantumCircuit(self.num_qubits)
        ucg = UCGate(random_rzs, mux_simp=mux_simp)
        qc.append(ucg, [x for x in range(self.num_qubits)])

        if not routed:
            qc_opt = transpile(qc, optimization_level=3)
        else:
            assert self.coupling_map != None
            qc_opt = transpile(qc, optimization_level=3, coupling_map=self.coupling_map)

        qc_opt = qc_opt.decompose(reps=4)
        return qc_opt
    



class QiskitGray(QiskitBase):
    def generate_circuit(self, routed = False):
        grey_gates = get_grey_gates(self.num_controls, False, True)
        qc = QuantumCircuit(self.num_qubits)
        for gate in grey_gates:
            qc.rz(np.random.random() * np.pi - 0.00123, self.num_controls)
            qc.cx(gate[0], gate[1])

        if routed:
            assert self.coupling_map != None
            qc = transpile(qc, optimization_level=3, coupling_map=self.coupling_map)

        qc = qc.decompose(reps=4)
        return qc




if __name__ == "__main__":
    
    num_qubits = int(sys.argv[1])
    assert num_qubits > 1

    coupling_map = None

    match sys.argv[2] if len(sys.argv) > 2 else "":
        case "cairo":
            fake_cairo = FakeCairoV2()
            coupling_map = fake_cairo.coupling_map
        case "garnet":
            with open(f"coupling_maps/fake_garnet.json") as f:
                fake_garnet = json.load(f)
            coupling_map = fake_garnet
        case _:
            coupling_map = None

    qn = QiskitNative(num_qubits, coupling_map)
    qc = qn.generate_circuit(mux_simp=True, routed=True)
    qn_cx_count = qn.count_cx(qc)

    qg = QiskitGray(num_qubits, coupling_map)
    qc2 = qg.generate_circuit(routed=True)
    qg_cx_count = qg.count_cx(qc2)

    proposed = RoutedMultiplexor(coupling_map=coupling_map, num_qubits=num_qubits)
    p_cx_count = proposed.execute_gates()

    print(f"Qiskit native UCGate CX count: {qn_cx_count}")
    print(f"Qiskit with grey code multiplexor CX count: {qg_cx_count}")
    print(f"Proposed method CX count: {p_cx_count}")