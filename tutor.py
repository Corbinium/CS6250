"""
Grover's Algorithm Implementation on 6 Qubits
ECE 6250 - Quantum Computing

This program implements Grover's search algorithm to find a marked state
in an unsorted database of 2^6 = 64 items with quadratic speedup.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram


def create_oracle(n_qubits: int, marked_state: str) -> QuantumCircuit:
    """
    Create an oracle that marks the target state by flipping its phase.
    
    Args:
        n_qubits: Number of qubits
        marked_state: Binary string representing the state to find (e.g., '101010')
    
    Returns:
        QuantumCircuit implementing the oracle
    """
    oracle = QuantumCircuit(n_qubits, name='Oracle')
    
    # Flip qubits that are '0' in the marked state (to mark with Z gate pattern)
    for i, bit in enumerate(reversed(marked_state)):
        if bit == '0':
            oracle.x(i)
    
    # Multi-controlled Z gate: flip phase of |111...1⟩
    # Implemented as H on last qubit, MCX, H on last qubit
    oracle.h(n_qubits - 1)
    oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    oracle.h(n_qubits - 1)
    
    # Flip back the qubits we flipped earlier
    for i, bit in enumerate(reversed(marked_state)):
        if bit == '0':
            oracle.x(i)
    
    return oracle


def create_diffusion_operator(n_qubits: int) -> QuantumCircuit:
    """
    Create the Grover diffusion operator (inversion about the mean).
    
    The diffusion operator is: D = 2|s⟩⟨s| - I
    where |s⟩ is the uniform superposition state.
    
    Args:
        n_qubits: Number of qubits
    
    Returns:
        QuantumCircuit implementing the diffusion operator
    """
    diffusion = QuantumCircuit(n_qubits, name='Diffusion')
    
    # Apply H gates to transform to computational basis
    diffusion.h(range(n_qubits))
    
    # Apply X gates to all qubits
    diffusion.x(range(n_qubits))
    
    # Multi-controlled Z gate (phase flip of |000...0⟩ after X gates = |111...1⟩)
    diffusion.h(n_qubits - 1)
    diffusion.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffusion.h(n_qubits - 1)
    
    # Apply X gates again
    diffusion.x(range(n_qubits))
    
    # Apply H gates to return to superposition basis
    diffusion.h(range(n_qubits))
    
    return diffusion


def grovers_algorithm(n_qubits: int, marked_state: str, num_iterations: int = None) -> QuantumCircuit:
    """
    Construct the complete Grover's algorithm circuit.
    
    Args:
        n_qubits: Number of qubits
        marked_state: Binary string of the state to search for
        num_iterations: Number of Grover iterations (optimal if None)
    
    Returns:
        Complete quantum circuit for Grover's algorithm
    """
    # Calculate optimal number of iterations if not specified
    if num_iterations is None:
        N = 2 ** n_qubits
        num_iterations = int(np.floor(np.pi / 4 * np.sqrt(N)))
    
    print(f"Number of qubits: {n_qubits}")
    print(f"Search space size: {2**n_qubits}")
    print(f"Marked state: |{marked_state}⟩")
    print(f"Number of Grover iterations: {num_iterations}")
    
    # Create the main circuit
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Step 1: Initialize uniform superposition
    qc.h(range(n_qubits))
    qc.barrier()
    
    # Create oracle and diffusion operator
    oracle = create_oracle(n_qubits, marked_state)
    diffusion = create_diffusion_operator(n_qubits)
    
    # Step 2: Apply Grover iterations
    for i in range(num_iterations):
        # Apply oracle
        qc.compose(oracle, inplace=True)
        qc.barrier()
        
        # Apply diffusion operator
        qc.compose(diffusion, inplace=True)
        qc.barrier()
    
    # Step 3: Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc


def run_grovers_algorithm(n_qubits: int = 6, marked_state: str = None, shots: int = 1024):
    """
    Run Grover's algorithm and display results.
    
    Args:
        n_qubits: Number of qubits (default: 6)
        marked_state: State to search for (random if None)
        shots: Number of measurement shots
    """
    # Generate a random marked state if not provided
    if marked_state is None:
        marked_state = format(np.random.randint(0, 2**n_qubits), f'0{n_qubits}b')
    
    # Validate marked state
    if len(marked_state) != n_qubits or not all(b in '01' for b in marked_state):
        raise ValueError(f"Marked state must be a {n_qubits}-bit binary string")
    
    print("=" * 50)
    print("GROVER'S ALGORITHM SIMULATION")
    print("=" * 50)
    
    # Create the Grover circuit
    qc = grovers_algorithm(n_qubits, marked_state)
    
    # Print circuit depth info
    print(f"Circuit depth: {qc.depth()}")
    print(f"Total gates: {qc.size()}")
    print("=" * 50)
    
    # Simulate the circuit
    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # Analyze results
    print("\nMeasurement Results (top 10):")
    print("-" * 30)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for state, count in sorted_counts[:10]:
        probability = count / shots * 100
        marker = " <-- TARGET" if state == marked_state else ""
        print(f"|{state}⟩: {count:4d} ({probability:5.1f}%){marker}")
    
    # Calculate success probability
    success_count = counts.get(marked_state, 0)
    success_probability = success_count / shots * 100
    
    print("-" * 30)
    print(f"\nTarget state |{marked_state}⟩ found with probability: {success_probability:.1f}%")
    
    # Theoretical success probability
    N = 2 ** n_qubits
    optimal_iterations = int(np.floor(np.pi / 4 * np.sqrt(N)))
    theta = np.arcsin(1 / np.sqrt(N))
    theoretical_prob = np.sin((2 * optimal_iterations + 1) * theta) ** 2 * 100
    print(f"Theoretical success probability: {theoretical_prob:.1f}%")
    
    # Compare with classical random search
    classical_prob = 1 / N * 100
    print(f"Classical random guess probability: {classical_prob:.2f}%")
    print(f"Quantum speedup factor: ~{np.sqrt(N):.1f}x")
    
    return qc, counts

def draw_circuit_diagram(n_qubits: int = 6, marked_state: str = '101010'):
    """
    Draw a simplified version of the Grover circuit for visualization.
    """
    # Create a smaller circuit for visualization (1 iteration)
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize superposition
    qc.h(range(n_qubits))
    qc.barrier(label='Init')
    
    # One Grover iteration
    oracle = create_oracle(n_qubits, marked_state)
    diffusion = create_diffusion_operator(n_qubits)
    
    qc.compose(oracle, inplace=True)
    qc.barrier(label='Oracle')
    
    qc.compose(diffusion, inplace=True)
    qc.barrier(label='Diffusion')
    
    qc.measure(range(n_qubits), range(n_qubits))
    
    # Draw the circuit
    print("\nCircuit Diagram (1 Grover iteration):")
    print(qc.draw(output='text', fold=120))
    
    return qc


if __name__ == "__main__":
    # Configuration
    N_QUBITS = 6
    MARKED_STATE = '110010'  # The state we want to find (can be any 6-bit string)
    SHOTS = 2048
    
    print("\n" + "=" * 50)
    print("QUANTUM SEARCH WITH GROVER'S ALGORITHM")
    print("=" * 50 + "\n")
    
    # Draw the circuit structure
    draw_circuit_diagram(N_QUBITS, MARKED_STATE)
    
    print("\n")
    
    # Run Grover's algorithm
    circuit, results = run_grovers_algorithm(
        n_qubits=N_QUBITS,
        marked_state=MARKED_STATE,
        shots=SHOTS
    )
