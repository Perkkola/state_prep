from scipy.stats import unitary_group
import numpy as np
SEED = 42
IQM_TOKEN = "3BGYdJuhl8pAeYUNcJ3b9r6r2CtXuGPHdjwA1StOaGUBmnyXrB9+EKROdY4Q4p4w"

def generate_random_unitary(n_qubits: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + n_qubits)
    return unitary_group.rvs(2**n_qubits, random_state=rng)

def _qiskit_synthesise_bare(U: np.ndarray, n: int):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate

    qc = QuantumCircuit(n)
    qc.append(UnitaryGate(U), range(n))
    return transpile(
        qc,
        basis_gates=["cx", "u", "id"],
        optimization_level=3,
        seed_transpiler=SEED,
    )

def run_tket(U, n, edges, n_phys, arch_name) -> int:
    from pytket.extensions.qiskit import qiskit_to_tk
    from pytket.architecture import Architecture
    from pytket.passes import (
        DecomposeBoxes,
        FullPeepholeOptimise,
        DefaultMappingPass,
        SequencePass,
        AutoRebase,
    )
    from pytket.extensions.qiskit import IBMQBackend
    from pytket.extensions.iqm import IQMBackend
    from pytket import Circuit, OpType
    from pytket.circuit import MultiplexedRotationBox
    from pytket.passes import DecomposeBoxes
    from qiskit_ibm_runtime import QiskitRuntimeService
    from pytket.circuit.display import render_circuit_jupyter
    from pytket.utils import Graph




    if arch_name == "iqm_garnet":
        iqm_token = IQM_TOKEN
        backend = IQMBackend('garnet', api_token=iqm_token)
    else:
        ibm_token = IBM_TOKEN
        inst = IBM_INSTANCE
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=ibm_token, instance=inst, overwrite=True)
        backend = IBMQBackend("ibm_marrakesh")
    # 1. Get bare Qiskit circuit & convert
    qc_base = _qiskit_synthesise_bare(U, n)
    tk_circ = qiskit_to_tk(qc_base)
    backend.default_compilation_pass(optimisation_level=2).apply(tk_circ)

    # cx_count = tk_circ.n_gates_of_type(OpType.CX)
    # cx_count += tk_circ.n_gates_of_type(OpType.CZ)

    cx_count = tk_circ.n_gates_of_type(OpType.CX)

    # 5. Sanity: count ALL 2-qubit gates to confirm rebase worked
    all_2q = sum(
        1 for cmd in tk_circ.get_commands()
        if len(cmd.qubits) == 2
    )
    if all_2q != cx_count:
        print(f"    ⚠ TKet rebase incomplete: {cx_count} CX but "
              f"{all_2q} total 2q gates  (delta = {all_2q - cx_count})")
        cx_count = all_2q  # conservative: count all 2q gates

    print(U)
    print(tk_circ.get_unitary())


num_qubits = 4
U = generate_random_unitary(num_qubits)
run_tket(U, num_qubits, None, None, "iqm_garnet")