from collections import deque
import multiprocessing
import math
from qiskit.circuit import QuantumCircuit, Parameter
import matplotlib.pyplot as plt
import sys

num_qubits = int(sys.argv[1]) #mapping num_qubits - 1 to the last physical qubit on LNN arch
num_controls = num_qubits - 1

assert num_controls >= 1

pp_terms = set([x for x in range(2 ** num_controls, 2 ** num_qubits)])
# discovered_pp_terms = set([2 ** num_controls])

len_pp_terms = 2 ** num_controls

# state = {q: 1 << q for q in range(num_qubits)}
# gate_queue = deque()
# gate_queue.append("RZ")

def grey_code(g):
    # state = s.copy()
    gates = g.copy()

    for i in range(1, len_pp_terms + 1):
        highest_index_diff = 0
        for j in range(num_controls):
            if (i >> j) & 1 != ((i - 1) >> j) & 1: highest_index_diff = j + 1
        # state[num_controls] = state[num_controls - highest_index_diff] ^ state[num_controls]
        gates.append("RZ")
        gates.append((num_controls - highest_index_diff, num_controls))

    return gates

def cancel_or_append(cnot):
    global state
    global gate_queue
    global discovered_pp_terms

    barrier_found = False
    barrier_type = "U"
    prev_gate = gate_queue.pop()

    if prev_gate == "Barrier U":
        barrier_type = "U"
        barrier_found = True
        prev_gate = gate_queue.pop()

    if prev_gate == "Barrier C":
        barrier_type = "C"
        barrier_found = True
        prev_gate = gate_queue.pop()


    if prev_gate != cnot:
        gate_queue.append(prev_gate)
        if barrier_found: gate_queue.append(f"Barrier {barrier_type}")
        gate_queue.append(cnot)
    elif barrier_found: gate_queue.append(f"Barrier {barrier_type}")
    
    if cnot[1] == num_controls and state[num_controls] in pp_terms and state[num_controls] not in discovered_pp_terms:
        discovered_pp_terms.add(state[num_controls])
        gate_queue.append("RZ")

def long_range_cnot(dist):
    global state
    offset = num_controls - dist
    
    for i in range(dist):
        i_f = i + offset
        state[i_f + 1] = state[i_f + 1] ^ state[i_f]
        cancel_or_append((i_f, i_f+1))
    for j in range(dist - 1, 0, -1):
        j_f = j + offset
        state[j_f] = state[j_f] ^ state[j_f - 1]
        cancel_or_append((j_f - 1, j_f))
    for k in range(1, dist):
        k_f = k + offset
        state[k_f + 1] = state[k_f + 1] ^ state[k_f]
        cancel_or_append((k_f, k_f + 1))
    for l in range(dist - 1, 1, -1):
        l_f = l + offset
        state[l_f] = state[l_f] ^ state[l_f - 1]
        cancel_or_append((l_f - 1, l_f))

def reverse_long_range_cnot(dist):
    global state
    offset = num_controls - dist

    for l in reversed(range(dist - 1, 1, -1)):
        l_f = l + offset
        state[l_f] = state[l_f] ^ state[l_f - 1]
        cancel_or_append((l_f - 1, l_f))
    for k in reversed(range(1, dist)):
        k_f = k + offset
        state[k_f + 1] = state[k_f + 1] ^ state[k_f]
        cancel_or_append((k_f, k_f + 1))
    for j in reversed(range(dist - 1, 0, -1)):
        j_f = j + offset
        state[j_f] = state[j_f] ^ state[j_f - 1]
        cancel_or_append((j_f - 1, j_f))
    for i in reversed(range(dist)):
        i_f = i + offset
        state[i_f + 1] = state[i_f + 1] ^ state[i_f]
        cancel_or_append((i_f, i_f + 1))

def route_multiplexor(long_range_grey_gates):
    global state
    global gate_queue
    global discovered_pp_terms

    num_configurations = 2 ** 8
    
    for conf in range(num_configurations):
        discovered_pp_terms = set([2 ** num_controls])
        state = {q: 1 << q for q in range(num_qubits)}
        gate_queue = deque()
        gate_queue.append("RZ")


        long_range_cnot(1)
        gate_queue.append("Barrier C")
        long_range_cnot(2)
        gate_queue.append("Barrier C")
        long_range_cnot(3)
        gate_queue.append("Barrier C")
        reverse_long_range_cnot(2)
        gate_queue.append("Barrier U")
        long_range_cnot(4)
        gate_queue.append("Barrier C")
        long_range_cnot(2)
        gate_queue.append("Barrier C")
        reverse_long_range_cnot(3)
        gate_queue.append("Barrier U")
        long_range_cnot(2)
        gate_queue.append("Barrier C")
        long_range_cnot(5)
        gate_queue.append("Barrier C")
        reverse_long_range_cnot(2)
        gate_queue.append("Barrier U")
        long_range_cnot(3)
        gate_queue.append("Barrier C")
        reverse_long_range_cnot(2)
        gate_queue.append("Barrier U")
        
        long_range_cnot(1)

    circuit_length = len([gate for gate in gate_queue if gate != "RZ" and gate != "Barrier C" and gate != "Barrier U"])
    print(f"Found {len(discovered_pp_terms)}/{len(pp_terms)} phase polynomial terms.")
    print(f"Circuit length: {circuit_length}")
    return gate_queue

def find_optimal(start, stop, long_range_gates, core_index, return_dict):
    global state
    global gate_queue
    global discovered_pp_terms

    optimal_circuits = []
    optimal_circuit_len = (2 ** num_qubits) * (2 ** num_qubits)

    for conf in range(start, stop):
        discovered_pp_terms = set([2 ** num_controls])
        state = {q: 1 << q for q in range(num_qubits)}
        gate_queue = deque()
        gate_queue.append("RZ")

        long_range_cnot(1)
        gate_queue.append("Barrier C")
        for i, gate in enumerate(long_range_gates[:int(len(long_range_gates) / 2)]):
            if (conf >> i) & 1 == 1:
                long_range_cnot(gate[1] - gate[0])
                gate_queue.append("Barrier C")
            else:
                reverse_long_range_cnot(gate[1] - gate[0])
                gate_queue.append("Barrier U")

        for i, gate in enumerate(long_range_gates[:int(len(long_range_gates) / 2)]):
            if (conf >> i) & 1 == 0:
                long_range_cnot(gate[1] - gate[0])
                gate_queue.append("Barrier C")
            else:
                reverse_long_range_cnot(gate[1] - gate[0])
                gate_queue.append("Barrier U")

        long_range_cnot(1)

        # draw_circuit(gate_queue)
        if len(discovered_pp_terms) != len(pp_terms): continue

        circuit_length = len([gate for gate in gate_queue if gate != "RZ" and gate != "Barrier C" and gate != "Barrier U"])
        if circuit_length < optimal_circuit_len:
            optimal_circuits = []
            # optimal_circuits.append(conf)
            optimal_circuits.append(gate_queue.copy())
            optimal_circuit_len = circuit_length
        elif circuit_length == optimal_circuit_len:
            # optimal_circuits.append(conf)
            optimal_circuits.append(gate_queue.copy())
        else: continue

    return_dict[core_index] = (optimal_circuits, optimal_circuit_len)

def draw_circuit(index, gates):
    qc = QuantumCircuit(num_qubits)
    phi = Parameter("θ")

    for gate in gates:
        if gate == "RZ":
            qc.rz(phi, num_controls)
        elif gate == "Barrier C":
            qc.barrier(label="C")
        elif gate == "Barrier U":
            qc.barrier(label="U")
        else:
            qc.cx(gate[0], gate[1])
    
    # fig = qc.draw(output="mpl", interactive=True, filename=f"./circuits/grey/optimal_circuit_{num_controls}-ctrl_{index}.png")
    fig = qc.draw(output="mpl", interactive=True)
    plt.show()


if __name__ == "__main__":
    # gates = route_multiplexor()
    # draw_circuit(0, gates)
    # exit()

    grey_gate_queue = deque()
    grey_gates = grey_code(grey_gate_queue)

    long_range_gates = []
    for gate in grey_gates:
        if gate == "RZ": continue

        if gate[1] - gate[0] > 1: long_range_gates.append(gate)

    num_configurations = 2 ** (len(long_range_gates) / 2)

    # route_multiplexor(long_range_gates)
    # exit()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    cores = multiprocessing.cpu_count()

    for index in range(cores):
        start = int(math.ceil(num_configurations / cores) * index)
        stop = int(min(math.ceil(num_configurations / cores) * (index + 1), num_configurations))
        p = multiprocessing.Process(target=find_optimal, args=(start, stop, long_range_gates, index, return_dict))
        # print(f"[{start}, {stop}]")
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    optimal_circuits = []
    optimal_circuit_len = (2 ** num_qubits) * (2 ** num_qubits)

    for item in return_dict.items():
        circuits = item[1][0]
        circuit_len = item[1][1]

        if circuit_len < optimal_circuit_len:
            optimal_circuits = circuits
            optimal_circuit_len = circuit_len
        elif circuit_len == optimal_circuit_len:
            optimal_circuits.extend(circuits)
     

    print(f"Long range cnots in circuit: {len(long_range_gates)}")
    print(f"Found {len(optimal_circuits)} optimal circuits.")
    print(f"Optimal circuit length: {optimal_circuit_len}")
    for index, circuit in enumerate(optimal_circuits):
        draw_circuit(index, circuit)

