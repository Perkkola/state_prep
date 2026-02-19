from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from utils import extract_single_qubit_unitaries, extract_angles, möttönen_transformation, generate_U, check_equivalence_up_to_phase, get_path
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
from qiskit.compiler import transpile
from qiskit.synthesis.unitary.qsd import qs_decomposition
import json


with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

fake_cairo = FakeCairoV2()

for num_qubits in range(3, 9):
        cumulative = 0
        for _ in range(5):
                U = generate_U(num_qubits)
                qc = qs_decomposition(U, opt_a1=True, opt_a2=True)
                # qc = QuantumCircuit(num_qubits)
                # qc.append(UnitaryGate(U), [x for x in range(num_qubits)])
                qc = transpile(qc, optimization_level=3, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'ry'], coupling_map=fake_garnet)
                cumulative += qc.count_ops().get('cx')
        average = cumulative / 5
        print(f"Num qubits: {num_qubits}, average CX count: {average}")

