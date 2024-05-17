from hamiltonian_simulation_benchmark import HamiltonianSimulation
import json 
import sys
import os 
import logging
from matplotlib.pyplot import figure

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit", "../../_common/transformers"]
import execute as ex
import metrics as metrics
import tket_optimiser as tket_optimiser  

# import precalculated data to compare against
filename = os.path.join(os.path.dirname(__file__), os.path.pardir, "_common", "precalculated_data.json")
with open(filename, 'r') as file:
    data = file.read()
precalculated_data = json.loads(data)

if __name__ == "__main__":


    for method in [1]: 
        for f in [.95]:
            for n_spins in [8]:
                for k in [10]:
                    for t in [1]:

                        high_optimisation = tket_optimiser.tket_transformer_generator(cx_fidelity=f, remove_barriers=True) 

                        print(f"Circuit metrics for method {method}, fidelity {f}, n_spins {n_spins}")

                        qc = HamiltonianSimulation(n_spins, K=k, t=t, method=method, measure_x= False)

                        compiled_qc = high_optimisation(qc, ex.backend)

