Redo the optimal neighborhood algorithm. We actually dont want to have the first target as close to every other qubit. We want the last qubit to be as close to every other
qubit since the recursion is exponential, so we want to minimize the exponentail routing cost for the smallest components of exponential number.

Sketch for the new algorithm -> Select a target qubit. Keep adding new qubits so that the distance to every other is as small as possible. Prioritize the distance to the qubits
with the smallest index, since most of the CNOTS will be between these qubits. Iterate for every qubit and select the best one.
        -One possibility would be to add a cost equal to the cost of the multiplexer (or similar) if a candidate qubit would be selected as the next target in the iteration.
        This way, it is easy to compare cost of each qubit choise and make the best one.

With this approach we shoudn't be disconnecting the graph which makes things a lot simpler.

Additionally, look into the Qiskit circ to unitary method. I might want to redo it myself from scratch.

The whole SWAP search can probably be deleted also. I can calculate the cost of the multiplexes plus SWAPS vs not SWAPPing and subtracting the outer most CNOTs absorbed by the
unitary and determine whether to SWAP or not, similarly to the optimal neighborhood algorithm.
        -When calculating the SWAP cost, we should SWAP towards qubit 0 and also SWAP with it. It doeesn't make sense to SWAP any further since everything is as close as possible to qubit 0.

Change SWAP sequence so that it doesnt go the shortest route but from n -> n-1 -> n-2 ... -> 0
    