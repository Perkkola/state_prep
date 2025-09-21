from functools import reduce
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from a_star import BasicAStar

num_qubits = 3
num_controls = num_qubits - 1
pp_terms = [x for x in range(2 ** num_controls + 1, 2 ** num_qubits)]

#Represent the qubit with a floor(log num_qubits) + 1 -bit number.
#Represent the current phase polynomial term on the qubit by num_qubits -bit number.
def build_identifier(state):
    return int(reduce(lambda x, y: (x << num_qubits) | y, reversed(list(state.values()))))


graph = {"vertices": set(), "edges": set(), "neighbors": {}}
# dist = {}
starting_state = {}

for q in range(num_qubits):
    starting_state[q] = 1 << q

id = build_identifier(starting_state)
graph["vertices"].add(id)
graph["neighbors"][id] = set()

#Create the search space
def traverse(state, id):
    for i in range(num_qubits - 1):
        cnot_up = state.copy()
        cnot_up[i] = cnot_up[i] ^ cnot_up[i+1]

        cnot_up_id = build_identifier(cnot_up)

        if cnot_up_id not in graph["vertices"]:
            graph["vertices"].add(cnot_up_id)
            graph["edges"].add((id, cnot_up_id))

            graph["neighbors"][cnot_up_id] = set()
            traverse(cnot_up, cnot_up_id)
        else:
            graph["edges"].add((id, cnot_up_id))

        graph["neighbors"][cnot_up_id].add(id)
        graph["neighbors"][id].add(cnot_up_id)

        cnot_down = state.copy()
        cnot_down[i+1] = cnot_down[i] ^ cnot_down[i+1]

        cnot_down_id = build_identifier(cnot_down)

        if cnot_down_id not in graph["vertices"]:
            graph["vertices"].add(cnot_down_id)
            graph["edges"].add((id, cnot_down_id))

            graph["neighbors"][cnot_down_id] = set()
            traverse(cnot_down, cnot_down_id)
        else:
            graph["edges"].add((id, cnot_down_id))

        graph["neighbors"][cnot_down_id].add(id)
        graph["neighbors"][id].add(cnot_down_id)

traverse(starting_state, id)

path = BasicAStar(graph["neighbors"]).astar(273, 305)

for term in pp_terms:
    for q in num_qubits:
        for v in graph["vertices"]:
            pass
print(list(path))
exit()
print(len(graph["vertices"]))
print(len(graph["edges"]))
print(graph["neighbors"][273])



G = nx.Graph()
G.add_nodes_from(graph["vertices"])
G.add_edges_from(graph["edges"])

fig = plt.subplot()
nx.draw(G, with_labels=True, font_weight='bold')

plt.show()