import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator

# Construct quantum circuit without measure
circ = QuantumCircuit(2)
circ.cp(0.123, 0, 1)
circ.save_unitary()


simulator = Aer.get_backend('aer_simulator')
circ = transpile(circ, simulator)


result = simulator.run(circ).result()
unitary = result.get_unitary(circ)
print("Circuit unitary:\n", np.asarray(unitary).round(5))

circ = QuantumCircuit(3)
circ.p(0.123 / 4, 2)
circ.cx(0, 2)
circ.p(-0.123 / 4, 2)
circ.cx(1, 2)
circ.p(0.123 / 4, 2)
circ.cx(0, 2)
circ.p(-0.123 / 4, 2)
circ.cx(1, 2)

# circ.p(0.123 / 4, 0)
# circ.p(0.123 / 4, 1)
# circ.p(0.123 / 4, 2)



# circ.p(0.123 / 2, 0)
circ.save_unitary()

circ = transpile(circ, simulator)
result = simulator.run(circ).result()
unitary = result.get_unitary(circ)
print("Circuit unitary:\n", np.asarray(unitary).round(5))