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