"State preparation paper" 


Original state prep:
https://journals.aps.org/pra/pdf/10.1103/PhysRevA.93.032318

Knill isometry, state prep optimizations in appendix

Optimal two and three qubit state prep:
https://journals.aps.org/pra/pdf/10.1103/PhysRevA.77.032320

Look into those, try to implement

Knill isometry state prep:
https://arxiv.org/abs/1003.5760

Figure our the Schmidt decomposition and the general unitary transformation, try to implement

General unitiray transformation circuit synthesis and optimization:

https://ieeexplore.ieee.org/document/1629135/ #2006 OG

Notes:
Ry and Rz multiplexors are just folded multicontrolled Ry and Rz gates with Gray code decomposition. 
Try if replacing CNOTs with CZs in the multiplexed Ry in the general unitary synthesis works.

Qiskit unitary synthesis already uses the QSD from the paper.

https://ieeexplore.ieee.org/document/9743148
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10025533
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10626814&tag=1
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10528701
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10313871

KAK1:

https://arxiv.org/abs/quant-ph/0507171

THREE QUBIT GATE FROM 6 TWO QUBITS GATES ETC:
https://arxiv.org/abs/quant-ph/9503016


Matrix product states etc.:

Minimizing entanglement entropy for enhanced quantum state preparation
https://arxiv.org/abs/2507.22562

Encoding of Matrix Product States into Quantum Circuits of One- and Two-Qubit Gates
https://arxiv.org/abs/1908.07958

Matrix product unitaries: structure, symmetries, and topological invariants

Matrix Product Representation of Locality Preserving Unitaries
https://arxiv.org/abs/1704.01943

Matrix Product State Representations
https://arxiv.org/abs/quant-ph/0608197


Logarithmic depth CNOT ladder: 
https://arxiv.org/pdf/2501.16802

Linear depth MCSU on LNN architecture:
https://www.arxiv.org/pdf/2506.00695v1

Low rank state prep (take the schmidt coefficient into account):
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10190145

Vittorio state prep:
https://arxiv.org/pdf/2505.06054

Better QSD:
https://arxiv.org/pdf/2403.13692


Steiner synth Github repo:
https://github.com/hashberg-io/pauliopt


Possibly faster method for Cosine-Sine decomposition:
https://www.sciencedirect.com/science/article/pii/S0024379518300843?via%3Dihub

For diagonal extraction:
https://arxiv.org/pdf/quant-ph/0401162 

For 3-CNOT unitary construction. Follow advice from Vivek to perform the diagonal extraction:
https://arxiv.org/pdf/quant-ph/0308033