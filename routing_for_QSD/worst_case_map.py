from collections import deque
import multiprocessing
import math
from qiskit.circuit import QuantumCircuit, Parameter
import matplotlib.pyplot as plt
import sys

num_qubits = int(sys.argv[1]) #mapping num_qubits - 1 to the last physical qubit on LNN arch
num_controls = num_qubits - 1

assert num_controls >= 1

dist_two_bit_map = {2: deque([1]), 3: deque([1]), 4: deque([0, 1]), 5: deque([1, 1, 0, 1]), 6: deque([0, 0, 0, 0, 1, 1, 0, 1])}


def grey_code(dist, half):
    upper_bound = (2 ** (dist - 1) + 1) if half else 2 ** dist + 1

    for i in range(1, upper_bound):
        highest_index_diff = 0
        for j in range(dist):
            if (i >> j) & 1 != ((i - 1) >> j) & 1: highest_index_diff = j + 1
        grey_gate_queue.append("RZ")
        grey_gate_queue.append((dist - highest_index_diff, dist))

def get_long_range_grey_gates(dist, half):
    global grey_gate_queue
    grey_gate_queue = deque()

    grey_code(dist, half)

    long_range_gates = []
    for gate in grey_gate_queue:
        if gate == "RZ": continue
        if gate[1] - gate[0] > 1: long_range_gates.append(gate)
    
    return long_range_gates

def cancel_or_append(cnot, size):
    global state
    global gate_queue
    global discovered_pp_terms
    global pp_terms

    prev_gate = gate_queue.pop()

    if prev_gate != cnot:
        gate_queue.append(prev_gate)
        gate_queue.append(cnot)
    
    if cnot[1] == size and state[size] in pp_terms and state[size] not in discovered_pp_terms:
        discovered_pp_terms.add(state[size])
        gate_queue.append("RZ")

def long_range_cnot(dist, size):
    global state
    offset = size - dist
    
    for i in range(dist):
        i_f = i + offset
        state[i_f + 1] = state[i_f + 1] ^ state[i_f]
        cancel_or_append((i_f, i_f+1), size)
    for j in range(dist - 1, 0, -1):
        j_f = j + offset
        state[j_f] = state[j_f] ^ state[j_f - 1]
        cancel_or_append((j_f - 1, j_f), size)
    for k in range(1, dist):
        k_f = k + offset
        state[k_f + 1] = state[k_f + 1] ^ state[k_f]
        cancel_or_append((k_f, k_f + 1), size)
    for l in range(dist - 1, 1, -1):
        l_f = l + offset
        state[l_f] = state[l_f] ^ state[l_f - 1]
        cancel_or_append((l_f - 1, l_f), size)

def reverse_long_range_cnot(dist, size):
    global state
    offset = size - dist

    for l in reversed(range(dist - 1, 1, -1)):
        l_f = l + offset
        state[l_f] = state[l_f] ^ state[l_f - 1]
        cancel_or_append((l_f - 1, l_f), size)
    for k in reversed(range(1, dist)):
        k_f = k + offset
        state[k_f + 1] = state[k_f + 1] ^ state[k_f]
        cancel_or_append((k_f, k_f + 1), size)
    for j in reversed(range(dist - 1, 0, -1)):
        j_f = j + offset
        state[j_f] = state[j_f] ^ state[j_f - 1]
        cancel_or_append((j_f - 1, j_f), size)
    for i in reversed(range(dist)):
        i_f = i + offset
        state[i_f + 1] = state[i_f + 1] ^ state[i_f]
        cancel_or_append((i_f, i_f + 1), size)

def long_range_leg(dist, state, size):
    offset = size - dist
    for i in range(dist):
        i_f = i + offset
        state[i_f + 1] = state[i_f + 1] ^ state[i_f]

    return state

def reverse_long_lange_leg(dist, state, size):
    dist = dist - 1
    return long_range_cnot(dist, state, size)

def route_multiplexor():
    global state
    global gate_queue
    global discovered_pp_terms
    global pp_terms

    long_range_bit_map = {2: deque([1])}

    for i in range(num_controls):
        long_range_gates = get_long_range_grey_gates(i + 1, True)
        
        pp_terms = set([x for x in range(2 ** (i + 1), 2 ** (i + 2))])
        discovered_pp_terms = set([2 ** (i + 1)])
        state = {q: 1 << q for q in range(i + 2)}
        gate_queue = deque()
        gate_queue.append("RZ")

        bit_map = deque()

        long_range_cnot(1, i + 1)
        for gate in long_range_gates:
            gate_index = gate[1] - gate[0]
            orientation = long_range_bit_map[gate_index].pop()
            long_range_bit_map[gate_index].appendleft(orientation)
            bit_map.append(orientation)

            if orientation == 1:
                long_range_cnot(gate_index, i + 1)
            else:
                reverse_long_range_cnot(gate_index, i + 1)

        for gate in long_range_gates:
            gate_index = gate[1] - gate[0]
            if bit_map.popleft() == 0 or i == 1:
                long_range_cnot(gate_index, i + 1)
            else:
                reverse_long_range_cnot(gate_index, i + 1)
        
        for j in range(i + 1, 0, -1):
            if j == 1: 
                if i <= 4:
                    offset = 1 if i <= 4 else 0
                    long_range_bit_map[2] = dist_two_bit_map[i + 1 + offset]
                    print(long_range_bit_map[2])
                else: 
                    # print(f"Bits: {i % 2}, amount: {2 ** (i - 2)}")
                    for _ in range(2 ** (i - 2)):
                        long_range_bit_map[2].appendleft(i % 2)
            else:
                long_range_bit_map[j + 1] = long_range_bit_map[j]
                print(long_range_bit_map[j + 1])

        print("//////////////////////////")
            
        long_range_cnot(1, i + 1)

    circuit_length = len([gate for gate in gate_queue if gate != "RZ" and gate != "Barrier C" and gate != "Barrier U"])
    print(f"Found {len(discovered_pp_terms)}/{len(pp_terms)} phase polynomial terms.")
    print(f"Circuit length: {circuit_length}")
    # draw_circuit(0, gate_queue)
    return gate_queue

def find_optimal(long_range_gates):
    global state
    global gate_queue
    global discovered_pp_terms
    global first_half_terms
    global first_half
    global pp_terms

    pp_terms = set([x for x in range(2 ** num_controls, 2 ** num_qubits)])
    first_half = True
    first_half_terms = [2, 3]

    optimal_circuits = []
    optimal_circuit_len = (2 ** num_qubits) * (2 ** num_qubits)

    num_configurations = 2 ** 16

    for n in range(num_configurations):
        conf = int(f"1{(n >> 0) & 1}011{(n >> 1) & 1}010{(n >> 2) & 1}011{(n >> 3) & 1}011{(n >> 4) & 1}011{(n >> 5) & 1}010{(n >> 6) & 1}011{(n >> 7) & 1}010{(n >> 8) & 1}011{(n >> 9) & 1}010{(n >> 10) & 1}011{(n >> 11) & 1}011{(n >> 12) & 1}011{(n >> 13) & 1}010{(n >> 14) & 1}011{(n >> 15) & 1}011010001010100010001000101010001011011101010111011010001011011011", 2)

        discovered_pp_terms = set([2 ** num_controls])
        state = {q: 1 << q for q in range(num_qubits)}
        gate_queue = deque()
        gate_queue.append("RZ")

        long_range_cnot(1, num_controls)
        gate_queue.append("Barrier C")
        for i, gate in enumerate(long_range_gates):
            if (conf >> i) & 1 == 1:
                long_range_cnot(gate[1] - gate[0], num_controls)
                gate_queue.append("Barrier C")
            else:
                reverse_long_range_cnot(gate[1] - gate[0], num_controls)
                gate_queue.append("Barrier U")

        for i, gate in enumerate(long_range_gates):
            if (conf >> i) & 1 == 0:
                long_range_cnot(gate[1] - gate[0], num_controls)
                gate_queue.append("Barrier C")
            else:
                reverse_long_range_cnot(gate[1] - gate[0], num_controls)
                gate_queue.append("Barrier U")

        long_range_cnot(1, num_controls)

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

    return optimal_circuits, optimal_circuit_len

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
    
    fig = qc.draw(output="mpl", interactive=True, filename=f"./circuits/cnot_at_end/optimal_circuit_{num_controls}-ctrl_{index}.png")
    # fig = qc.draw(output="mpl", interactive=True)
    plt.show()


if __name__ == "__main__":


    long_range_gates = get_long_range_grey_gates(num_controls, True)

    route_multiplexor()
    exit()
    optimal_circuits, optimal_circuit_len = find_optimal(long_range_gates)



    print(f"Long range cnots in circuit: {len(long_range_gates)}")
    print(f"Found {len(optimal_circuits)} optimal circuits.")
    print(f"Optimal circuit length: {optimal_circuit_len}")
    for index, circuit in enumerate(optimal_circuits):
        draw_circuit(0, circuit)

