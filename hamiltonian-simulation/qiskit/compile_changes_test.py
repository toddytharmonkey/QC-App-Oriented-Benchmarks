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

# Get circuit metrics fom the circuit passed in
def get_circuit_metrics(qc):

    
    # obtain initial circuit size metrics
    qc_depth = qc.depth()
    qc_size = qc.size()
    qc_count_ops = qc.count_ops()
    qc_xi = 0
    qc_n2q = 0 
    
    # iterate over the ordereddict to determine xi (ratio of 2 qubit gates to one qubit gates)
    n1q = 0; n2q = 0
    if qc_count_ops != None:
        for key, value in qc_count_ops.items():
            if key == "measure": continue
            if key == "barrier": continue
            if key.startswith("c") or key.startswith("mc"):
                n2q += value
            else:
                n1q += value
        qc_xi = n2q / (n1q + n2q)
        qc_n2q = n2q
    
    return qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q
if __name__ == "__main__":


    for method in [1,2]: 
        for f in [.995]:
            for n_spins in [8]:
                for k in [5]:
                    for t in [1.151243815198]:

                        high_optimisation = tket_optimiser.tket_transformer_generator(cx_fidelity=f, remove_barriers=False) 

                        print(f"Circuit metrics for method {method}, fidelity {f}, n_spins {n_spins}")

                        from qiskit import transpile

                        w = 1 # strength of disorder
                        h_x = precalculated_data['h_x'][:n_spins] # precalculated random numbers between [-1, 1]
                        h_z = precalculated_data['h_z'][:n_spins]

                        qc = HamiltonianSimulation(n_spins, K=k, t=t, method=method, measure_x= False)
                        transpiled_qc = transpile(qc, ex.backend)

                        print("Printing circuit metrics: qc_tr_depth, qc_tr_size, qc_tr_count_ops, qc_tr_xi, qc_tr_n2q")
                        print(ex.transpile_for_metrics(qc))
                        print("Now displaying metrics for post compiled metrics")
                        
                        high_optimisation = tket_optimiser.tket_transformer_generator(cx_fidelity=f) 

                        compiled_qc = high_optimisation(qc, ex.backend)
                        print(ex.transpile_for_metrics(compiled_qc))

                        filename_suffix = f"method_{method}_trotter_{k}_time_{t}".replace(".","")
                        fig = figure(figsize=(20,20))
                        ax = fig.add_subplot()
                        title = ex.transpile_for_metrics(transpiled_qc)
                        ax.set_title(title)


                        transpiled_qc.draw("mpl", ax=ax, filename="transpiled_qc" + filename_suffix)

                        fig = figure(figsize=(20,20))
                        ax = fig.add_subplot()
                        title = ex.transpile_for_metrics(compiled_qc)
                        ax.set_title(title)
                        transpile(compiled_qc,ex.backend).draw("mpl", ax=ax,filename="compiled_qc"+filename_suffix)
                        print()
