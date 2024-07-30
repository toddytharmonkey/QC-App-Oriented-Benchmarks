import json
import os
import sys
import time
import numpy as np
from tqdm import tqdm

import hamlib_simulation_benchmark, hamlib_simulation_kernel, hamlib_utils
import execute
import metrics

import sys

sys.path.insert(1, "hamlib/qiskit")

execute.verbose = False
execute.verbose_time = False
hamlib_simulation_benchmark.verbose = False
hamlib_simulation_kernel.verbose = False
hamlib_utils.verbose = False

hamlib_simulation_kernel.global_U = None
hamlib_simulation_kernel.global_enc = None
hamlib_simulation_kernel.global_ratio = None
hamlib_simulation_kernel.global_rinst = None
hamlib_simulation_kernel.global_h = None
hamlib_simulation_kernel.global_pbc_val = None
min_qubits = 2
max_qubits = 10
skip_qubits = 1
num_shots = 1000

backend_id = "qasm_simulator"
# backend_id="statevector_simulator"

hub = "ibm-q"
group = "open"
project = "main"
provider_backend = None
exec_options = {}
if __name__ == "__main__":
    method = 3

    #!TODO: do something about max_circuits: max_circuits is greater for the random pauli circuits?

for hamiltonian_name in tqdm(["tfim", "heis"]):
    for pbc_val in ["pbc"]:
        # for h in [0,1,2,3,5]
        for h in [2]:  # add 4 and 6 for tfim only
            for K in [5, 10]:
                # exec_options = {"noise_model" : None} # use this line only for method 2 noiseless calculation
                hamlib_simulation_kernel.global_h = h
                hamlib_simulation_kernel.global_pbc_val = pbc_val
                print(
                    "Method, Hamiltonian, h, pbc_val = ",
                    method,
                    hamiltonian_name,
                    h,
                    pbc_val,
                )
                print("=======================================================")
                metrics.data_suffix = (
                    f"_method_{method}_{hamiltonian_name}_{pbc_val}_h_{h}"
                )
                # for method 2 noiseless calculations only
                # metrics.data_suffix = f'_method_{method}_{hamiltonian_name}_{pbc_val}_h_{h}_noiseless'
                hamlib_simulation_benchmark.run(
                    min_qubits=min_qubits,
                    max_qubits=max_qubits,
                    skip_qubits=skip_qubits,
                    max_circuits=max_circuits,
                    num_shots=num_shots,
                    method=method,
                    hamiltonian=hamiltonian_name,
                    init_state=None,
                    backend_id=backend_id,
                    provider_backend=provider_backend,
                    hub=hub,
                    group=group,
                    project=project,
                    exec_options=exec_options,
                )
