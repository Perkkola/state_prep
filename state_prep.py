import sys
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate



np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def normalize(data):
    sq = 0
    for el in data:
        sq += np.square(el)

    sq = np.sqrt(sq)
    arr = data / sq
    return arr



def prepare_state(data):
    upper_circ_qubit_count = (qubit_count - qubit_count % 2) // 2
    lower_circ_qubit_count = (qubit_count + qubit_count % 2) // 2
    upper_circ = QuantumRegister(upper_circ_qubit_count)
    lower_circ = QuantumRegister(lower_circ_qubit_count)
    qc = QuantumCircuit(upper_circ, lower_circ)

    mat_A = data.reshape((2 ** lower_circ_qubit_count, 2 **  upper_circ_qubit_count))
    U, S, V = np.linalg.svd(mat_A)
    V_t = V.transpose()

    qc.prepare_state(S, upper_circ)

    half_trunc = qubit_count // 2
    for i in range(half_trunc):
        qc.cx(i, i + half_trunc)

    U_gate = UnitaryGate(U)
    V_gate = UnitaryGate(V_t)

    qc.append(V_gate, upper_circ)
    qc.append(U_gate, lower_circ)

    state_vec = Statevector(qc)
    
    print(state_vec)



qubit_count = 4
arr = normalize(np.array([2 * x - 1 for x in np.random.random_sample(2 ** qubit_count)]))
prepare_state(arr)

print(arr)