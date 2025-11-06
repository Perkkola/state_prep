from collections import deque

def grey_code(dist, half):
    upper_bound = (2 ** (dist - 1) + 1) if half else 2 ** dist + 1

    for i in range(1, upper_bound):
        highest_index_diff = 0
        for j in range(dist):
            if (i >> j) & 1 != ((i - 1) >> j) & 1: highest_index_diff = j + 1
        grey_gate_queue.append("RZ")
        grey_gate_queue.append((dist - highest_index_diff, dist))

        grey_state_queue.append(grey_state[dist])
        grey_state[dist] ^= grey_state[dist - highest_index_diff]

def get_grey_gates(dist, half = False, all_gates = True, state_queue = False):
    global grey_gate_queue
    global grey_state
    global grey_state_queue

    grey_gate_queue = deque()
    grey_state_queue = deque()
    grey_state = {q: 1 << q for q in range(dist + 1)}
    
    grey_code(dist, half)

    long_range_gates = []
    for gate in grey_gate_queue:
        if gate == "RZ": continue
        if gate[1] - gate[0] > 1 and not all_gates: long_range_gates.append(gate)
        elif all_gates: long_range_gates.append(gate)
    
    if state_queue:
        return long_range_gates, grey_state_queue
    else:
        return long_range_gates