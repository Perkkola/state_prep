import os
import math
import sys
import numpy as np
import random
from functools import reduce
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate
from gray_synth import synth_cnot_phase_aam
from normalize_data import normalize

np.set_printoptions(threshold=sys.maxsize)


arr = normalize(np.array([2 * x - 1 for x in np.random.random_sample(2 ** 4)]))
# arr = normalize(np.array([x  for x in range(2 ** 4)]))

print(arr)

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

    qc_opt = transpile(qc, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'x', 'ry'], optimization_level=0)
    print(qc_opt)
    state_vec = Statevector(qc_opt)

    # print(qc)
    # state_vec = Statevector(qc)
    
    print(state_vec)


prepare_state(arr)

# print("QISKIT ///////////////////////////////////")
# qiskit_circ = QuantumCircuit(4)
# qiskit_circ.initialize(arr)

# state_vec_qiskit = Statevector(qiskit_circ)
# qc_opt_qiskit = transpile(qiskit_circ, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'x', 'ry'], optimization_level=3)

# print(state_vec_qiskit)
# print(qc_opt_qiskit)
# exit()
