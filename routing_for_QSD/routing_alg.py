from functools import reduce
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from a_star import BasicAStar
from collections import deque

num_qubits = 3
num_controls = num_qubits - 1
max_cnot = num_controls * (2 ** num_controls) #Hypothetical upper bound


#Represent the qubit with a floor(log num_qubits) + 1 -bit number.
#Represent the current phase polynomial term on the qubit by num_qubits -bit number.
def build_identifier(state):
    return int(reduce(lambda x, y: (x << num_qubits) | y, reversed(list(state.values()))))


graph = {"vertices": set(), "edges": set(), "neighbors": {}}
pp_states = {}
dist = deque()
vertices_with_root_dist = deque()
starting_state = {}
state_queue = deque()

for q in range(num_qubits):
    starting_state[q] = 1 << q

for term in range(2 ** num_controls, 2 ** num_qubits):
    pp_states[term] = []

start_id = build_identifier(starting_state)
state_queue.append((starting_state, 0)) #Include distance from the root



while True:
    try:
        source, source_dist_from_root = state_queue.popleft()
        source_id = build_identifier(source)

        
        if source_id in graph["vertices"] or source_dist_from_root > max_cnot / 2:
            continue

        graph["vertices"].add(source_id)
        graph["neighbors"][source_id] = []
        vertices_with_root_dist.append((source_id, source_dist_from_root))

    
        for i in range(num_qubits - 1):
            cnot_up = source.copy()
            cnot_up[i] = cnot_up[i] ^ cnot_up[i+1]
            cnot_up_id = build_identifier(cnot_up)
            

            if (cnot_up_id, source_id) in graph["edges"]:
                graph["neighbors"][source_id].append(cnot_up_id)
            elif source_dist_from_root + 1 <= max_cnot / 2:
                graph["edges"].add((source_id, cnot_up_id))
                graph["neighbors"][source_id].append(cnot_up_id)
                state_queue.append((cnot_up, source_dist_from_root + 1))
            
            cnot_down = source.copy()
            cnot_down[i+1] = cnot_down[i] ^ cnot_down[i+1]
            cnot_down_id = build_identifier(cnot_down)
 
            if (cnot_down_id, source_id) in graph["edges"]:
                graph["neighbors"][source_id].append(cnot_down_id)
            elif source_dist_from_root + 1 <= max_cnot / 2:
                graph["edges"].add((source_id, cnot_down_id))
                graph["neighbors"][source_id].append(cnot_down_id)
                state_queue.append((cnot_down, source_dist_from_root + 1))

            if source[i] >= 2 ** num_controls and source[i] < 2 ** num_qubits: pp_states[source[i]].append((source_id, source_dist_from_root))
        if source[num_qubits - 1] >= 2 ** num_controls and source[num_qubits - 1] < 2 ** num_qubits: pp_states[source[num_qubits - 1]].append((source_id, source_dist_from_root))
    except Exception as e:
        # print(e)
        break

# AStar = BasicAStar(graph["neighbors"])

# for i in range(2 ** num_controls, 2 ** num_qubits):
#     for a in pp_states[i]:
#         for j in range(2 ** num_controls + (i % (2 ** num_controls)) + 1, 2 ** num_qubits):
#             for b in pp_states[j]:
#                 id_a, a_dist_from_root = a
#                 id_b, b_dist_from_root = b

#                 if a_dist_from_root + b_dist_from_root >= max_cnot:
#                     continue

#                 path = AStar.astar(id_a, id_b)

#                 if len(list(path)) - 1 + a_dist_from_root + b_dist_from_root > max_cnot:
#                     continue

#                 dist.append((a, b, len(list(path)) - 1))

# for term in pp_terms:
#     for q in num_qubits:
#         for v in graph["vertices"]:
#             pass


graph["vertices"] = list(graph["vertices"])
graph["edges"] = list(graph["edges"])
print(len(graph["vertices"]))
print(len(graph["edges"]))
# print(graph["neighbors"][273])
print(len(dist))

print(pp_states)
exit()
s = 0

for key in pp_states.keys():
    s += len(pp_states[key])

print(s)


# print(vertices_with_root_dist)
# print(pp_states[2 ** num_controls])
# print(pp_states[2 ** num_controls + 1])
# exit()



G = nx.Graph()
G.add_nodes_from(graph["vertices"])
G.add_edges_from(graph["edges"])

fig = plt.subplot()
nx.draw(G, with_labels=True, font_weight='bold')

plt.show()