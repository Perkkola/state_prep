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
from gray_synth import synth_cnot_phase_aam
from normalize_data import normalize

np.set_printoptions(threshold=sys.maxsize)

qubits = 4

arr = normalize(np.array([x for x in range(2 ** 4)]))

mat_A = arr.reshape((qubits, qubits))
U, S, Vh = np.linalg.svd(mat_A)
print(arr)
print(mat_A)
print(U)
print(S)
print(Vh)
exit()


def prepare_state(data):
    qubit_count = int(np.log2(len(data)))
    #Special cases for qubit count 1, 2 and 3
    half_plus_modulo = qubit_count // 2 + qubit_count % 2
    upper_circ_qb_count = half_plus_modulo if half_plus_modulo % 2 == 0 else qubit_count // 2
    lower_circ_qb_count = half_plus_modulo if half_plus_modulo % 2 == 1 else qubit_count // 2
    upper_circ = QuantumRegister(upper_circ_qb_count)
    lower_circ = QuantumRegister(lower_circ_qb_count)
    qc = QuantumCircuit(upper_circ, lower_circ)

    # 1. Generate Scmidt decomposition and calculate the coefficients on the upper half
    # 2. Apply CNOTS from upper circ to lower circ
    # 3. Apply Unitary transformation to upper and lower circ
    print(qc)




prepare_state(arr)
# exit()

# qc_opt = QuantumCircuit(2)

# qc_opt.h([x for x in range(2)])
# qc_opt.ry(4.18815+4.46963, 1)
# qc_opt.cx(0, 1)
# qc_opt.ry(4.18815-4.46963, 1)
# qc_opt.cx(0, 1)

# qc_opt.ry(3.3242, 0)
# qc_opt.cx(1, 0)
# qc_opt.ry(-3.3242, 0)
# qc_opt.cx(1, 0)



# qc_opt = transpile(qc_opt, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'x', 'ry'], optimization_level=0)
# state_vec_qc_opt = Statevector(qc_opt)
# print(qc_opt)
# print(state_vec_qc_opt)
