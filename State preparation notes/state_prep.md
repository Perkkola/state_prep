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