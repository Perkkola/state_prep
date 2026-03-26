import statistics
import numpy as np
theoretical_cnot_count = [19, 95, 423, 1783, 7319, 29655, 119383, 479063, 1919319]

iqm_qubit = {
    "proposed": [27, 135, 617, 2599, 10585, 42801, 172025, 689873, 2762999],
    "tket": [31, 147, 651, 2728, 11237, 45505],
    "qiskit": [29, 183, 903, 3796, 15606, 61386, 242271, 957936 ],
    "pennylane": [40, 221, 963, 4033, 16517]
}

ibm_qubit = {
    "proposed": [27, 145, 681, 2873, 11761, 49521, 194697, 757939, 3020169],
    "tket": [35, 157, 768, 3684, 15429, 64508],
    "qiskit": [31, 177, 833, 3658, 15047, 60748, 244311, 979867 ],
    "pennylane": [37, 221, 1053, 4573, 19037]
}

iqm_runtime = {
    "proposed": [0.0, 0.02, 0.11, 0.35, 0.87, 3.75, 16.51, 72.2, 345.94],
    "tket": [3.4, 9.1, 41.1, 149.6, 418.7, 961.1, ],
    "qiskit": [5.54, 0.1, 0.62, 2.31, 7.36, 26.8, 134.5, 599.4],
    "pennylane": [0.0, 2.9, 10.6, 123.2, 1100.3]
}

ibm_runtime = {
    "proposed": [0.01, 0.03, 0.08, 0.23, 0.92, 4.02, 17.1, 75.9, 334.67],
    "tket": [18.8, 27.1, 61.0, 195.7, 538.5, 1224.8, ],
    "qiskit": [2.93, 3.03, 3.23, 5.1, 12.23, 31.21, 132.6, 617.2 ],
    "pennylane": [9.1, 4.3, 14.4, 104.3, 1374.3 ]
}

garnet_ratios = []
for theoretical, proposed in zip(theoretical_cnot_count, iqm_qubit["proposed"]):
    print(f"CNOT count overhead of proposed method compared to theoretical on IQM Garnet: {(proposed ) / theoretical}")
    garnet_ratios.append((proposed ) / theoretical)
print("\n")

marrakesh_ratios = []
for theoretical, proposed in zip(theoretical_cnot_count, ibm_qubit["proposed"]):
    print(f"CNOT count overhead of proposed method compared to theoretical on IBM Marrakesh: {(proposed ) / theoretical}")
    marrakesh_ratios.append((proposed ) / theoretical)

print(f"Ratio growth factor for IQM Garnet: {statistics.mean(garnet_ratios[:])} +- {np.std(garnet_ratios[:])}")
print(f"Ratio growth factor for IBM Marrakesh: {statistics.mean(marrakesh_ratios[:])} +- {np.std(marrakesh_ratios[:])}")
exit()

for key, value in iqm_qubit.items():
    if key == "proposed": continue
    s = 0
    for index in range(len(value)):
        proposed_count = iqm_qubit["proposed"][index]
        if value[index] == 2 ** 31: continue
        print(f"Proposed CNOT decrease compared to {key} on IQM Garnet for {index + 3} qubits: {(value[index] - proposed_count) / value[index]}")
        s += (value[index] - proposed_count) / value[index]
    average_decrease = s / len(value)
    print(f"Average CNOT decrease with on IQM Garnet: {average_decrease}")

print("\n")
for key, value in ibm_qubit.items():
    if key == "proposed": continue
    s = 0
    for index in range(len(value)):
        proposed_count = ibm_qubit["proposed"][index]
        if value[index] == 2 ** 31: continue
        print(f"Proposed CNOT decrease compared to {key} on IBM Marrakesh for {index + 3} qubits: {(value[index] - proposed_count) / value[index]}")
        s += (value[index] - proposed_count) / value[index]
        average_decrease = s / len(value)
    print(f"Average CNOT decrease with on IBM Marraqkesh: {average_decrease}")

print("\n")
for key, value in iqm_runtime.items():
    if key == "proposed": continue
    s = 0
    for index in range(len(value)):
        proposed_count = ibm_runtime["proposed"][index] if ibm_runtime["proposed"][index] != 0 else 1e-3
        comparison_count = value[index] if value[index] != 0 else 1e-3
        if comparison_count == 2 ** 31: continue
        print(f"Runtime speedup compared to {key} on IQM Garnet for {index + 3} qubits: {(comparison_count) / proposed_count}")
        s += (comparison_count) / proposed_count
    average_decrease = s / len(value)
    print(f"Average speedup with on IQM Garnet: {average_decrease}")

print("\n")
for key, value in ibm_runtime.items():
    if key == "proposed": continue
    s = 0
    for index in range(len(value)):
        proposed_count = ibm_runtime["proposed"][index] if ibm_runtime["proposed"][index] != 0 else 1e-3
        comparison_count = value[index] if value[index] != 0 else 1e-3
        if comparison_count == 2 ** 31: continue
        print(f"Runtime speedup compared to {key} on IBM Marrakesh for {index + 3} qubits: {(comparison_count) / proposed_count}")
        s += (comparison_count) / proposed_count
    average_decrease = s / len(value)
    print(f"Average speedup with on IBM Marrakesh: {average_decrease}")