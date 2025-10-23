from collections import deque
from qiskit.circuit import QuantumCircuit, Parameter
import matplotlib.pyplot as plt
import sys
import os
from functools import reduce
import json

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

def save_json(bit_map, file):
    json_bitmap = {}

    for key, value in bit_map.items():
        json_bitmap[str(key)] = reduce(lambda x, y: str(x) + str(y), value, "")

    with open(file, 'w', encoding='utf-8') as f:
        json.dump(json_bitmap, f, ensure_ascii=False, indent=4)

def find_prev_best():
    long_range_bit_map = {2: deque([1])}
    prev_best_template_index = 0
    for i in range(num_controls, 1, -1):
        if i <= num_controls and os.path.exists(f"patterns/{i}-ctrl.json"):
            with open(f"patterns/{i}-ctrl.json") as f:
                bit_map = {}
                prev_best_template_index = i - 1
                json_bitmap = json.load(f)

                for key, value in json_bitmap.items():
                    bit_map[int(key)] = deque([int(x) for x in value])

                long_range_bit_map = bit_map
                break

    return long_range_bit_map, prev_best_template_index

def route_multiplexor():
    global state
    global gate_queue
    global discovered_pp_terms
    global pp_terms

    long_range_bit_map, prev_best_template_index = find_prev_best()


    for i in range(prev_best_template_index, num_controls):
        if i >= 1 and not os.path.exists(f"patterns/{i+1}-ctrl.json"):
            save_json(long_range_bit_map, f"patterns/{i+1}-ctrl.json")

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
                    long_range_bit_map[2] = dist_two_bit_map[i + 2]
                else: 
                    for _ in range(2 ** (i - 2)):
                        long_range_bit_map[2].appendleft(i % 2)
            else:
                long_range_bit_map[j + 1] = long_range_bit_map[j].copy()

        
        long_range_cnot(1, i + 1)

    circuit_length = len([gate for gate in gate_queue if gate != "RZ"])
    print(f"Found {len(discovered_pp_terms)}/{len(pp_terms)} phase polynomial terms.")
    print(f"Circuit length: {circuit_length}")
    return gate_queue


def draw_circuit(gates):
    qc = QuantumCircuit(num_qubits)
    phi = Parameter("θ")

    for gate in gates:
        if gate == "RZ":
            qc.rz(phi, num_controls)
        else:
            qc.cx(gate[0], gate[1])
    
    # fig = qc.draw(output="mpl", interactive=True, filename=f"./circuits/cnot_at_end/optimal_circuit_{num_controls}-ctrl_{index}.png")
    fig = qc.draw(output="mpl", interactive=True)
    plt.show()


if __name__ == "__main__":
    gates = route_multiplexor()
    # draw_circuit(gates)

