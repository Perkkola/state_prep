import time
import subprocess
import os
import json
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

# ==========================================
# 1. Qiskit to Intel C++ Translator
# ==========================================
def qiskit_to_intel_cpp(qc: QuantumCircuit, filename="benchmark_circuit.cpp"):
    """
    Translates a Qiskit QuantumCircuit into C++ code for the Intel Quantum SDK.
    Supports basic single and two-qubit gates.
    """
    num_qubits = qc.num_qubits
    
    cpp_lines = [
        "#include <quantum.hpp>",
        "#include <iostream>\n",
        "int main() {",
        f"    iqs::QubitRegister qreg({num_qubits});\n"
    ]
    
    # Iterate through the Qiskit circuit instructions
    for instruction in qc.data:
        gate = instruction.operation
        name = gate.name.lower()
        
        # Extract the integer indices of the qubits this gate acts on
        q_indices = [qc.find_bit(q).index for q in instruction.qubits]
        
        # Single-qubit non-parameterized gates
        if name in ['h', 'x', 'y', 'z']:
            cpp_lines.append(f"    iqs::{name.upper()}(qreg[{q_indices[0]}]);")
            
        # Single-qubit parameterized rotations
        elif name in ['rx', 'ry', 'rz']:
            theta = gate.params[0]
            cpp_lines.append(f"    iqs::{name.upper()}(qreg[{q_indices[0]}], {theta});")
            
        # Two-qubit gates
        elif name == 'cx':
            cpp_lines.append(f"    iqs::CNOT(qreg[{q_indices[0]}], qreg[{q_indices[1]}]);")
        elif name == 'cz':
            cpp_lines.append(f"    iqs::CZ(qreg[{q_indices[0]}], qreg[{q_indices[1]}]);")
            
        else:
            cpp_lines.append(f"    // WARNING: Gate '{name}' translation not implemented.")
            print(f"Warning: Gate '{name}' skipped during C++ translation.")

    cpp_lines.append("\n    return 0;")
    cpp_lines.append("}")
    
    with open(filename, "w") as f:
        f.write("\n".join(cpp_lines))


# ==========================================
# 2. Qiskit Benchmarking
# ==========================================
def benchmark_qiskit(qc: QuantumCircuit, topology_edges):
    print(f"--- Benchmarking Qiskit (Optimization Level 3) ---")
    
    coupling_map = CouplingMap(topology_edges)
    
    start_time = time.time()
    optimized_qc = transpile(qc, coupling_map=coupling_map, optimization_level=3)
    compilation_time = time.time() - start_time
    
    gate_counts = optimized_qc.count_ops()
    cx_count = gate_counts.get('cx', 0)
    
    print(f"Qiskit Compilation Time: {compilation_time:.4f} seconds")
    print(f"Qiskit CNOT Count: {cx_count}\n")
    
    return compilation_time, cx_count

# ==========================================
# 3. Intel Quantum SDK (PCOAST) Benchmarking
# ==========================================
def generate_hardware_config(topology_edges, filename="hardware_config.json"):
    config = {"topology": topology_edges}
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)

def benchmark_pcoast(qc: QuantumCircuit, topology_edges):
    print(f"--- Benchmarking Intel Quantum SDK (PCOAST -O1) ---")
    cpp_filename = "benchmark_circuit.cpp"
    config_filename = "hardware_config.json"
    
    # Dynamically generate the C++ code from the input circuit
    qiskit_to_intel_cpp(qc, cpp_filename)
    generate_hardware_config(topology_edges, config_filename)
    
    compile_cmd = [
        "intel-quantum-compiler", 
        "-O1", 
        f"-mllvm", f"-hardware-config={config_filename}", 
        cpp_filename, 
        "-o", "benchmark_executable"
    ]
    
    start_time = time.time()
    try:
        process = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
        compilation_time = time.time() - start_time
        
        compiler_output = process.stdout + process.stderr
        cx_count = parse_pcoast_gate_count(compiler_output) 
        
        print(f"PCOAST Compilation Time: {compilation_time:.4f} seconds")
        print(f"PCOAST CNOT Count: {cx_count}\n")
        
    except FileNotFoundError:
        print("Error: Intel Quantum SDK compiler ('iqc') not found in PATH.")
        compilation_time, cx_count = None, None
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e.stderr}")
        compilation_time, cx_count = None, None
    finally:
        for file in [cpp_filename, config_filename]:
            if os.path.exists(file):
                os.remove(file)
            
    return compilation_time, cx_count

def parse_pcoast_gate_count(compiler_output):
    """Placeholder for LLVM IR parsing logic."""
    return 0 

# ==========================================
# 4. Execution Example
# ==========================================
if __name__ == "__main__":
    QUBITS = 4
    
    # 1. Define arbitrary topology
    LINEAR_TOPOLOGY = [[0, 1], [1, 2], [2, 3]]
    LINEAR_TOPOLOGY += [[j, i] for i, j in LINEAR_TOPOLOGY]
    
    # 2. Build an arbitrary Qiskit circuit
    custom_qc = QuantumCircuit(QUBITS)
    custom_qc.h(0)
    custom_qc.cx(0, 1)
    custom_qc.rx(3.14, 2)
    custom_qc.cx(1, 2)
    custom_qc.cz(2, 3)
    custom_qc.x(3)
    
    print(f"Starting Benchmark for custom {QUBITS}-qubit circuit...\n")
    
    # 3. Pass the circuit to both benchmarks
    benchmark_qiskit(custom_qc, LINEAR_TOPOLOGY)
    benchmark_pcoast(custom_qc, LINEAR_TOPOLOGY)