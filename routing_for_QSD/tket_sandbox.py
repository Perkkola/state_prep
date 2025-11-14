from pytket import Circuit, OpType
from pytket.extensions.qiskit import IBMQBackend
from pytket.extensions.iqm import IQMBackend
from qiskit_ibm_runtime import QiskitRuntimeService


# ibm_token = 'u_k47sVdjZ8QvxY5bUj-PQRM2h_CNU8jcIh6RxJG_7Jg'
inst = 'crn:v1:bluemix:public:quantum-computing:us-east:a/14c391407c20401a8dc347e17f51ba83:7d2deb88-e3a6-4315-8ca8-e936c1009807::'

# QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=ibm_token, instance=inst)

circ = Circuit(3) # Define a circuit to be compiled to the backend
circ.CX(0, 1)
circ.H(1)
circ.Rx(0.42, 1)
circ.S(1)
circ.CX(0, 2)
circ.CX(2, 1)
circ.Z(2)
circ.Y(1)
circ.CX(0, 1)
circ.CX(2, 0)
circ.measure_all()

# backend = IBMQBackend('ibm_fez') # Initialise backend for an IBM device
# ['ibm_fez', 'ibm_torino', 'ibm_marrakesh']
backend = IBMQBackend("ibm_fez")
print(backend.backend_info.architecture.to_dict())
exit()
backend = IQMBackend('garnet', api_token="3BGYdJuhl8pAeYUNcJ3b9r6r2CtXuGPHdjwA1StOaGUBmnyXrB9+EKROdY4Q4p4w")

# backendinfo_list = IBMQBackend.available_devices(instance=inst)
# print([backend.device_name for backend in backendinfo_list])

print("Total gate count before compilation =", circ.n_gates)
print("CX count before compilation =",  circ.n_gates_of_type(OpType.CX))

    # Now apply the default_compilation_pass at different levels of optimisation.


test_circ = circ.copy()
backend.default_compilation_pass(optimisation_level=2).apply(test_circ)
assert backend.valid_circuit(test_circ)
print("Gates", test_circ.n_gates)
print("CXs", test_circ.n_gates_of_type(OpType.CZ))