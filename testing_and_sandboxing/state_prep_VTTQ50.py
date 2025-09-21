import sys
import numpy as np
import math

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate


from normalize_data import normalize

np.set_printoptions(threshold=sys.maxsize)


arr = normalize(np.array([2 * x - 1 for x in np.random.random_sample(2 ** 5)]))
# arr = normalize(np.array([x  for x in range(2 ** 4)]))

arr = np.array([-1.56543837e-01, -3.08838235e-01,  2.85190674e-01 ,-1.20806053e-01,
  2.84993658e-02 ,-1.90403154e-01, -2.42676345e-01, -2.72982189e-01,
 -1.07331674e-01, -9.28381047e-02 , 4.66242485e-02 , 2.96832832e-01,
 -1.57154902e-02 ,-3.75615784e-02, -1.61318717e-01 , 1.45130168e-01,
 -2.20590318e-01 , 2.04184376e-01 , 1.85423402e-01 ,-1.00883425e-04,
 -3.51420954e-02, -1.05797101e-01 , 1.05742709e-01, -1.85286012e-01,
  1.37244458e-01,  1.24560252e-01 , 1.85222125e-01 , 2.05486102e-02,
  2.98305018e-01 , 1.88803213e-01 ,-2.79149453e-01,  1.51738596e-02])

# print(arr)

def prepare_state(data):
    qubit_count = int(np.log2(len(data)))
    #Special cases for qubit count 1, 2 and 3
    upper_circ_qubit_count = (qubit_count - qubit_count % 2) // 2
    lower_circ_qubit_count = (qubit_count + qubit_count % 2) // 2
    upper_circ = QuantumRegister(upper_circ_qubit_count)
    lower_circ = QuantumRegister(lower_circ_qubit_count)
    qc = QuantumCircuit(upper_circ, lower_circ)

    mat_A = arr.reshape((2 ** lower_circ_qubit_count, 2 **  upper_circ_qubit_count))
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

    qc.measure_all()

    sampler = Sampler()
    shots = 2 ** 14
    job = sampler.run([qc], shots=shots)
    job_result = job.result()
    counts = job_result._pub_results[0].data.meas.get_counts()

    recreated_state_vec = {k:  math.sqrt(v / shots) for k, v in counts.items()}
    sorted_state_vec = dict(sorted(recreated_state_vec.items()))
    print(sorted_state_vec)
    # print(qc)
    # state_vec = Statevector(qc)
    
    # print(state_vec)


prepare_state(arr)