from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from utils import extract_single_qubit_unitaries, extract_angles, möttönen_transformation, generate_U, check_equivalence_up_to_phase, get_path
from qiskit.compiler import transpile
import json


with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

num_qubits = 6

U = generate_U(num_qubits)

qc = QuantumCircuit(num_qubits)
qc.append(UnitaryGate(U), [x for x in range(num_qubits)])
qc = transpile(qc, optimization_level=3, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'ry'], coupling_map=fake_garnet)
print(qc.count_ops())


