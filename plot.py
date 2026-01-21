import matplotlib.pyplot as plt
import numpy as np

qiskit_native = [2, 6, 16, 43, 88, 235, 335, 983, 1486, 3758, 6457, 13024, 25527, 50235, 96813]
qiskit_gray = [2, 4, 8, 16, 41, 79, 161, 333, 635, 1233, 2499, 5162, 10086, 20197, 39551]
steiner_gray = [2, 8, 25, 39, 91, 193, 377, 877, 1516, 3041, 6137, 12176, 24407, 49056, 100960]
tket_gray = [2, 4, 8, 16, 38, 79, 155, 307, 611, 1219, 2642, 5278]
tket_native = [2, 4, 8, 16, 38, 79, 155, 307, 611, 1219, 2642, 5278]
proposed = [2, 4, 8, 16, 36, 72, 144, 288, 576, 1152, 2304, 4608, 9226, 18450, 36898]

qubits_1 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
qubits_2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

difference = [0, 0, 0, 0, 2, 7, 11, 19, 35, 67, 195, 554, 860, 1747, 2653]

fig, ax = plt.subplots()

p = ax.bar(qubits_2, difference, label=[str(item) for item in difference])
ax.bar_label(p)

# plt.yscale("log")
# plt.plot(qubits_2, qiskit_native, label='qiskit_native')
# plt.plot(qubits_2, qiskit_gray, label= 'qiskit_gray')
# plt.plot(qubits_2, steiner_gray, label='steiner_gray')
# plt.plot(qubits_1, tket_native, label='tket_native')
# plt.plot(qubits_1, tket_gray, label='tket_gray')
# plt.plot(qubits_2, proposed, label='proposed')

# plt.legend(loc='best')
plt.xlabel("Amount of qubits")
plt.ylabel("Difference in CNOT count with the next best method")
# plt.ylabel("CNOT count")
plt.show()