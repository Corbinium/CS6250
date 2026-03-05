import qiskit
import matplotlib.pyplot as plt
from qiskit_experiments.library.characterization.t2hahn import T2Hahn

qubit = 0
conversion_factor = 1e-6 # our delay will be in micro-sec
delays = list(range(0, 51, 5) )
# Round so that the delay gates in the circuit display does not have trailing 9999's
delays = [round(float(_) * conversion_factor, ndigits=6) for _ in delays]
number_of_echoes = 1

# Create a T2Hahn experiment. Print the first circuit as an example
exp1 = T2Hahn(physical_qubits=(qubit,), delays=delays, num_echoes=number_of_echoes)
print(exp1.circuits()[1])

from qiskit_experiments.test.t2hahn_backend import T2HahnBackend

estimated_t2hahn = 20 * conversion_factor
# The behavior of the backend is determined by the following parameters
backend = T2HahnBackend(
    t2hahn=[estimated_t2hahn],
    frequency=[100100],
    initialization_error=[0.0],
    readout0to1=[0.02],
    readout1to0=[0.02],
)

exp1.analysis.set_options(p0=None, plot=True)
expdata1 = exp1.run(backend=backend, shots=2000, seed_simulator=101)
expdata1.block_for_results()  # Wait for job/analysis to finish.


expdata1.figure(0).figure.savefig("t2hahn_result.png")
print(expdata1.analysis_results(dataframe=True))