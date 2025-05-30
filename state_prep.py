import os
import math
import sys
import numpy as np
import random
from functools import reduce
import pandas as pd
import time

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import normalize
from scipy.linalg import cossin
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate


from gray_synth import synth_cnot_phase_aam
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

    print(U)
    print(V_t)

    qc.prepare_state(S, upper_circ)

    half_trunc = qubit_count // 2
    for i in range(half_trunc):
        qc.cx(i, i + half_trunc)

    U_gate = UnitaryGate(U)
    V_gate = UnitaryGate(V_t)

    qc.append(V_gate, upper_circ)
    qc.append(U_gate, lower_circ)

    # qc_opt = transpile(qc, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'x', 'ry'], optimization_level=0)
    # print(qc_opt)
    # state_vec = Statevector(qc_opt)

    # print(qc)
    # state_vec = Statevector(qc)
    
    # print(state_vec)


# prepare_state(arr)

U = [[ 0.62923587, -0.27810894,  0.06225542 , 0.32819821,  0.21411186, -0.26067223,
  -0.16945149 , 0.52213037],
 [-0.12832786, -0.72115181 , 0.06330487 ,-0.25546329  ,0.1811286,  -0.15270451,
   0.58015675, -0.03866454],
 [ 0.14146621, 0.33503048 , 0.63506461,  0.14609694, -0.17964295 , 0.2131513,
   0.56863418,  0.20503807],
 [-0.16206701 , 0.04719766 , 0.46284018 ,-0.12025008  ,0.7869089,   0.20527736,
  -0.27689368 ,-0.06921729],
 [ 0.25624259 , 0.38011453 ,-0.34262361, -0.67547194,  0.23405039 , 0.00414033,
   0.21956521 , 0.33644285],
 [ 0.23268996 ,-0.28038369, -0.23106108  ,0.05510674 ,-0.02434206 , 0.89985678,
  -0.00166854 , 0.02183679],
 [ 0.0135536  , 0.25475662 ,-0.4235493,   0.54214077 , 0.46218213, -0.0468658,
   0.43173278, -0.24372693],
 [-0.64909721 ,-0.01092904, -0.15625139 , 0.19494701 , 0.02944304 , 0.09593373,
   0.0091941  , 0.71132259]]

V_t = [[-0.54409413, -0.10171701, -0.25889078,  0.79157488],
 [-0.39583923,  0.59434125, -0.58362529, -0.38658932],
 [ 0.7315188,   0.35371355, -0.41023319,  0.41409624],
 [-0.11024121,  0.71505164,  0.65120336,  0.22908997]]

u, cs, vdh = cossin(V_t, p=2, q=2)

print(u)
print(cs)
print(vdh)
# print("QISKIT ///////////////////////////////////")
# qiskit_circ = QuantumCircuit(4)
# qiskit_circ.initialize(arr)

# state_vec_qiskit = Statevector(qiskit_circ)
# qc_opt_qiskit = transpile(qiskit_circ, basis_gates=['cx', 'h', 'x', 'rz', 'rx', 'x', 'ry'], optimization_level=3)

# print(state_vec_qiskit)
# print(qc_opt_qiskit)
# exit()
