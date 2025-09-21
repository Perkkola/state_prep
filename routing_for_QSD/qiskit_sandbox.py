import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator

# Construct quantum circuit without measure
circ = QuantumCircuit(3)
circ.rz(0.123 / 4, 2)
circ.cx(0, 2)
circ.rz(-0.123 / 4, 2)
circ.cx(1, 2)
circ.rz(0.123 / 4, 2)
circ.cx(0, 2)
circ.rz(-0.123 / 4, 2)
circ.cx(1, 2)
print(circ)
circ.save_unitary()


simulator = Aer.get_backend('aer_simulator')
circ = transpile(circ, simulator)


result = simulator.run(circ).result()
unitary = result.get_unitary(circ)
print("Circuit unitary:\n", np.asarray(unitary).round(5))

circ = QuantumCircuit(3)
circ.rz(0.123 / 4, 2)
circ.cx(2, 1)
circ.rz(-0.123 / 4, 1)
circ.cx(1, 0)
circ.rz(0.123 / 4, 0)
circ.cx(0, 1)
circ.cx(2, 1)
circ.rz(-0.123 / 4, 1)
circ.cx(1, 0)
circ.cx(2, 1)
circ.swap(0, 1)

print(circ)
circ.save_unitary()
circ = transpile(circ, simulator)


result = simulator.run(circ).result()
unitary = result.get_unitary(circ)
print("Circuit unitary:\n", np.asarray(unitary).round(5))