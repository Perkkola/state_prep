from collections import deque
from qiskit.circuit import QuantumCircuit, Parameter, QuantumRegister
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
from qiskit.compiler import transpile
from qiskit_aer import Aer, AerSimulator
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
import math
from functools import reduce
import json
from utils import get_grey_gates, generate_random_rz_multiplexer_unitary, extract_single_qubit_unitaries, extract_angles, clean_matrix, is_unitary
import numpy as np
from gray_synth import synth_cnot_phase_aam

class RoutedMultiplexor(object):
    def __init__(self, multiplexer_angles = None, coupling_map = None, num_qubits = 5, reverse = False):
        assert num_qubits >= 2

        self.multiplexer_angles = multiplexer_angles
        self.coupling_map = coupling_map
        self.reverse = reverse


        self.num_qubits = num_qubits if self.multiplexer_angles == None else int(math.log2(len(multiplexer_angles)) + 1)
        self.num_controls = self.num_qubits - 1

        if self.multiplexer_angles == None: 
            self.multiplexer_angles = [0.123 for x in range(2 ** self.num_controls)]
        else:
            self.möttönen_transformation()


        if self.coupling_map == None: self.coupling_map = [[x, y] for x in range(self.num_qubits) for y in range(self.num_qubits) if x != y]

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
    
    
    def find_optimal_neighborhood(self):
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

        return optimal_neighborhoods

    def map_grey_qubits_to_arch(self):
        optimal_neighborhood = self.find_optimal_neighborhood()[0]
        self.optimal_neighborhood = optimal_neighborhood.copy()
        furthest_node_path = list(optimal_neighborhood.values())[-1]

        furthest_node = furthest_node_path[-1]
        closest_node = furthest_node_path[1]
        root = furthest_node_path[0]
        self.root = root
        self.closest_node = closest_node

        grey_to_arch_map = {}
        grey_to_arch_map[self.num_controls] = root
        grey_to_arch_map[self.num_controls-1] = closest_node
        grey_to_arch_map[0] = furthest_node

        optimal_neighborhood.pop(root)
        optimal_neighborhood.pop(closest_node)

        if optimal_neighborhood.get(furthest_node) != None:
            optimal_neighborhood.pop(furthest_node)
            offset = 2
        else: 
            offset = 1

        for key, grey_key in zip(list(optimal_neighborhood.keys()), range(self.num_controls - offset, 0, -1)):
            grey_to_arch_map[grey_key] = key
        
        self.grey_to_arch_map = grey_to_arch_map

        self.arch_to_grey_map = {}

        for key, value in self.grey_to_arch_map.items():
            self.arch_to_grey_map[value] = key

    
    def cancel_or_append(self, cnot, ignore):
        prev_gate = self.gate_queue.pop()

        if prev_gate != cnot:
            self.gate_queue.append(prev_gate)
            self.gate_queue.append(cnot)
        
        if cnot[1] == self.num_controls and self.state[self.num_controls] in self.pp_terms and self.state[self.num_controls] not in self.discovered_pp_terms:
            self.discovered_pp_terms.add(self.state[self.num_controls])
            self.gate_queue.append(("RZ", self.state_to_angle_dict[self.state[self.num_controls]]))
        elif cnot[1] == self.num_controls and ignore and self.state[self.num_controls] in self.pp_terms and self.state[self.num_controls] in self.discovered_pp_terms:
            self.gate_queue.pop()
            self.state[cnot[1]] ^= self.state[cnot[0]]
    
    def long_range_cnot(self, arch_path, ignore = False):

        grey_path = list(reversed(list(map(lambda arch_qubit: self.arch_to_grey_map[arch_qubit], arch_path))))
        grey_path_dist = len(grey_path)
        for i in range(grey_path_dist - 1):
            self.state[grey_path[i + 1]] ^= self.state[grey_path[i]]
            self.cancel_or_append((grey_path[i], grey_path[i + 1]), ignore)

        for j in range(grey_path_dist - 2, 0, -1):
            self.state[grey_path[j]] ^= self.state[grey_path[j - 1]]
            self.cancel_or_append((grey_path[j - 1], grey_path[j]), ignore)

        for k in range(1, grey_path_dist - 1):
            self.state[grey_path[k + 1]] ^= self.state[grey_path[k]]
            self.cancel_or_append((grey_path[k], grey_path[k + 1]), ignore)

        for l in range(grey_path_dist - 2, 1, -1):
            self.state[grey_path[l]] ^= self.state[grey_path[l - 1]]
            self.cancel_or_append((grey_path[l - 1], grey_path[l]), ignore)

    def reset_state(self):
        for i in range(self.num_controls):
            if (self.state[self.num_controls] >> i) & 1 == 1: 
                arch_qubit = self.grey_to_arch_map[i]
                arch_path = self.optimal_neighborhood[arch_qubit]
                self.long_range_cnot(arch_path, False)

    def find_missing_terms(self, unfound_terms):
        parity_matrix = []
        for term in unfound_terms:
            temp = []
            for i in range(self.num_qubits):
                if (term >> i) & 1 == 1: temp.append(1)
                else: temp.append(0)

            parity_matrix.append(temp)

        parity_matrix = list(np.array(parity_matrix).transpose())
        for i in range(len(parity_matrix)):
            parity_matrix[i] = list(parity_matrix[i])

        angles = [0.123 for x in range(len(parity_matrix[0]))]
        qc = synth_cnot_phase_aam(parity_matrix, angles)

        synth_cnots = deque()
        for instruction in qc.data:
            indices = [qc.find_bit(q).index for q in instruction.qubits]
            if len(indices) > 1: synth_cnots.append((indices[0], indices[1]))

        for gate in synth_cnots:
            ctrl_qubit = gate[0]
            arch_qubit = self.grey_to_arch_map[ctrl_qubit]
            arch_path = self.optimal_neighborhood[arch_qubit]
            dist = len(arch_path) - 1

            ignore = False if dist > 1 else True
            self.long_range_cnot(arch_path, False)
    
    def möttönen_transformation(self):
        #todo: multithread
        transformed_angles = np.zeros(len(self.multiplexer_angles))
        for i in range(len(self.multiplexer_angles)):
            temp = 0
            g_m = i ^ (i >> 1)
            for j in range(len(self.multiplexer_angles)):
                dot_product = 0
                for k in range(self.num_controls):
                    dot_product += ((g_m >> k) & 1) * ((j >> k) & 1)
                temp += math.pow(2, -self.num_controls) * math.pow((-1), dot_product) * self.multiplexer_angles[j] * 2
            transformed_angles[i] = temp
        
        self.multiplexer_angles = transformed_angles

    def execute_gates(self):
        self.map_grey_qubits_to_arch()
        grey_gates, grey_state_queue = get_grey_gates(self.num_controls, False, True, True)

        self.pp_terms = set([x for x in range(2 ** self.num_controls, 2 ** self.num_qubits)])
        self.discovered_pp_terms = set([2 ** self.num_controls])
        self.state = {q: 1 << q for q in range(self.num_qubits)}
        self.state_to_angle_dict = dict(zip(grey_state_queue, self.multiplexer_angles))
        init_state = self.state.copy()

        self.gate_queue = deque()
        self.gate_queue.append(("RZ", self.multiplexer_angles[0]))
        for gate in grey_gates:
            ctrl_qubit = gate[0]
            arch_qubit = self.grey_to_arch_map[ctrl_qubit]
            arch_path = self.optimal_neighborhood[arch_qubit]
            dist = len(arch_path) - 1

            ignore = False if dist > 1 else True
            self.long_range_cnot(arch_path, ignore)
        
        if init_state != self.state:
            self.reset_state()

        unfound_terms = self.pp_terms - self.discovered_pp_terms

        if len(unfound_terms) > 0:
            self.find_missing_terms(unfound_terms)

        circuit_length = len([gate for gate in self.gate_queue if gate[0] != "RZ"])

        if len(self.discovered_pp_terms) != len(self.pp_terms):
            print(f"Found {len(self.discovered_pp_terms)}/{len(self.pp_terms)} phase polynomial terms.")

        if self.state != init_state:
            print("State was not reset correctly!")
        
        return circuit_length

    def map_grey_gates_to_arch(self):
        self.execute_gates()
        grey_gates = self.gate_queue.copy()
        arch_gates = deque()

        if not self.reverse:
            keys = list(self.grey_to_arch_map.keys()).copy()
            temp = {}
            for key in keys:
                temp[self.num_controls - key] = self.grey_to_arch_map.pop(key)
            self.grey_to_arch_map = temp

        while True:
            try:
                grey_gate = grey_gates.popleft()
                if grey_gate[0] != "RZ":
                    if self.reverse: arch_gate = (self.grey_to_arch_map[grey_gate[0]], self.grey_to_arch_map[grey_gate[1]])
                    else: arch_gate = (self.grey_to_arch_map[self.num_controls - grey_gate[0]], self.grey_to_arch_map[self.num_controls - grey_gate[1]])
                else:
                    arch_gate = grey_gate
                arch_gates.append(arch_gate)
            except Exception as e:
                break

        self.arch_gates = arch_gates
        return arch_gates
    
    def run(self):
        self.map_grey_gates_to_arch()



    def draw_backend(self, planar = False, filename = None):
        G = nx.Graph()
        G.add_edges_from(self.coupling_map)


        if planar:
            is_planar, embedding = nx.check_planarity(G)
            pos = nx.combinatorial_embedding_to_pos(embedding)
            nx.draw(G, pos, with_labels=True)
            if filename != None:
                plt.savefig(f"./coupling_maps/{filename}.png", format="PNG")
            plt.show()
        else:
            nx.draw(G, with_labels=True, font_weight='bold')
            if filename != None:
                plt.savefig(f"./coupling_maps/{filename}.png", format="PNG")
            plt.show()

    def draw_circuit(self, arch = False, filename = None, print_unitary = False):
        if not arch:
            qc = QuantumCircuit(self.num_qubits)
        else: 
            qubits = [QuantumRegister(1, name=f'q{i}') for i in range(1, len(self.vertices) + 1)]
            qc = QuantumCircuit(*qubits)

        gates = self.gate_queue if not arch else self.arch_gates
        

        for gate in gates:
            if gate[0] == "RZ":
                if self.reverse: qc.rz(gate[1], self.num_controls) if not arch else qc.rz(gate[1], self.root - 1)
                else: qc.rz(gate[1], 0) if not arch else qc.rz(gate[1], self.root - 1)
            else:
                if self.reverse: qc.cx(gate[0], gate[1]) if not arch else qc.cx(gate[0] - 1, gate[1] - 1)
                else: qc.cx(self.num_controls - gate[0], self.num_controls - gate[1]) if not arch else qc.cx(gate[0] - 1, gate[1] - 1)
        if filename != None:
            fig = qc.draw(output="mpl", interactive=True, filename=filename)
        else:
            fig = qc.draw(output="mpl", interactive=True)

        if print_unitary: self.print_circ_unitary(qc)
        plt.show()
    
    def print_circ_unitary(self, qc):
        qc = qc.copy()
        qc.save_unitary()
        simulator = Aer.get_backend('aer_simulator')
        qc = transpile(qc, simulator)

        result = simulator.run(qc).result()
        unitary = result.get_unitary(qc)
        print("Circuit unitary:\n", np.asarray(unitary).round(5))

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
    assert num_qubits > 1

    with open(f"coupling_maps/fake_garnet.json") as f:
        fake_garnet = json.load(f)

    fake_cairo = FakeCairoV2()


    multiplexor_unitary = generate_random_rz_multiplexer_unitary(num_qubits)

    single_qubit_unitaries = list(extract_single_qubit_unitaries(multiplexor_unitary))
    angles = list(extract_angles(single_qubit_unitaries))


    routed_multiplexor = RoutedMultiplexor(multiplexer_angles= angles, coupling_map= None, reverse=True)
    routed_multiplexor.run()
    # routed_multiplexor.draw_circuit(print_unitary=True)

    # routed_multiplexor.draw_circuit(arch=True, filename=f"./circuits/final/cairo_{num_qubits}_qubits.png")
    # routed_multiplexor.draw_circuit(arch=True)
    # print(routed_multiplexor.optimal_neighborhood)
    # routed_multiplexor.draw_backend(planar=False, filename="garnet_coupling_map")