from collections import deque
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
import math
from functools import reduce
import json

# num_qubits = int(sys.argv[1]) #mapping num_qubits - 1 to the last physical qubit on LNN arch
# num_controls = num_qubits - 1

# assert num_controls >= 1

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


class RoutedMultiplexor(object):
    def __init__(self, multiplexor = None, coupling_map = None, num_qubits = 5):
        assert num_qubits >= 2

        self.multiplexor = multiplexor
        self.coupling_map = coupling_map
        self.num_qubits = num_qubits if self.multiplexor == None else math.log2(len(multiplexor))
        self.num_controls = self.num_qubits - 1

        self.neighbors = self.get_neighbors()
        self.vertices = self.neighbors.copy().keys()

        assert len(self.vertices) >= self.num_qubits
    
    def get_neighbors(self):
        neighbors = {}
        for edge in self.coupling_map:
            if neighbors.get(edge[0]) == None: neighbors[edge[0]] = set()
            if neighbors.get(edge[1]) == None: neighbors[edge[1]] = set()

            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])

        return neighbors
    
    #Next, find shortest paths form root to all qubits. If there are multiple shortest paths, select the one which intersects the most with other paths.

    #Then, calculate the grey gates and map the grey qubits to the architecture accordignly. The nearest qubit to the root in the grey circuit should be mapped to to the path which
    # includes the furthest node. Rest should be mapped such that we minimize the amount of long range CNOTS.

    #After that, modify the long range cnot functions so that they apply the correct cnots.

    #Then, apply the algorithm, checking whether a cnot is needed and otherwise executing long range cnots according to the grey gates.

    #Finally, make sure everything works and start benchmarking.
    
    def find_optimal_paths(self):
        optimal_neighborhoods = []
        best_dist_cost = 2 ** 31

        for root in self.vertices:
            paths = {root: [root]}
            vertices_in_neighborhood = set([root])
            dist_cost = 0
            current_dist = 1
            found_neighborhood = False
            neighborhood_size = 1

            while not found_neighborhood:
                paths_copy = paths.copy()
                current_dist_paths = list(filter(lambda path: len(path) == current_dist, paths_copy.values()))
                current_dist_paths.sort(key = lambda path: len(self.neighbors[path[-1]]), reverse=True)

                for path in current_dist_paths:
                    if found_neighborhood: break
                    tail = path[-1]

                    neighbors = self.neighbors[tail]

                    for neighbor in neighbors:
                        
                        if neighbor not in vertices_in_neighborhood:
                            path_copy = path.copy()
                            path_copy.append(neighbor)

                            paths[neighbor] = path_copy
                            vertices_in_neighborhood.add(neighbor)
                            neighborhood_size += 1
                            dist_cost += current_dist
                            if neighborhood_size >= self.num_qubits:
                                found_neighborhood = True
                                break

                current_dist += 1
                
            if dist_cost < best_dist_cost:
                optimal_neighborhoods = []
                optimal_neighborhoods.append(paths)
                best_dist_cost = dist_cost
            elif dist_cost == best_dist_cost:
                optimal_neighborhoods.append(paths)

        print(best_dist_cost)
        for neighborhood in optimal_neighborhoods:
            print(neighborhood)

    def draw_backend(self, planar = False):
        G = nx.Graph()
        G.add_edges_from(self.coupling_map)


        if planar:
            is_planar, embedding = nx.check_planarity(G)
            pos = nx.combinatorial_embedding_to_pos(embedding)
            nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=600)
            plt.show()
        else:
            fig = plt.subplot()
            nx.draw(G, with_labels=True, font_weight='bold')

            plt.show()

    def find_highest_degree_nodes(self):
        neighbors = {}
        highest_degree_nodes = {}
        highest_degree = 0
        for edge in self.coupling_map:
            if neighbors.get(edge[0]) == None: neighbors[edge[0]] = [set(), 0]
            if neighbors.get(edge[1]) == None: neighbors[edge[1]] = [set(), 0]

            n_0 = neighbors[edge[0]]
            n_1 = neighbors[edge[1]]

            n_0[0].add(edge[1])
            n_0[1] = len(n_0[0])

            n_1[0].add(edge[0])
            n_1[1] = len(n_1[0])

            for index, candidate in enumerate([n_0, n_1]):
                if candidate[1] > highest_degree:
                    highest_degree_nodes = {}
                    highest_degree_nodes[edge[index]] = candidate[0]
                    highest_degree = candidate[1]
                elif candidate[1] == highest_degree:
                    highest_degree_nodes[edge[index]] = candidate[0]
                else: continue

        print(highest_degree_nodes, highest_degree)
    
        
        # print(neighbors)

if __name__ == "__main__":
    num_qubits = int(sys.argv[1])

    with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

    fake_cairo = FakeCairoV2()

    routed_multiplexor = RoutedMultiplexor(coupling_map=fake_garnet, num_qubits=num_qubits)
    routed_multiplexor.find_optimal_paths()

    routed_multiplexor.draw_backend()