import json
import os
import sys
import time
import numpy as np
from tqdm import tqdm

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
sys.path[1:1] = ["../_common"]
sys.path[1:1] = ["../../hamiltonian-simulation/qiskit"]

import execute as ex
import metrics as metrics

# import hamlib_simulation_kernel
# from hamlib_utils import create_full_filenames, construct_dataset_name
from qiskit_aer import Aer, AerSimulator
from qiskit import transpile

from qiskit.transpiler.passes import RemoveFinalMeasurements


def construct_circuits(num_qubits):
    from hamlib_simulation_kernel import HamiltonianSimulation

    circuits = {}

    circuits["Method 1 K = 10"] = HamiltonianSimulation(
        n_spins=num_qubits,
        K=10,
        t=1,
        hamiltonian="TFIM",
        init_state="checkerboard",
        method=1,
    )

    circuits["Method 1 K = 5"] = HamiltonianSimulation(
        n_spins=num_qubits,
        K=5,
        t=1,
        hamiltonian="TFIM",
        init_state="checkerboard",
        method=1,
    )

    circuits["Method 1 K = 10 Inverse"] = (
        circuits["Method 1 K = 10"]
        .remove_final_measurements(False)
        .inverse()
        .measure_all(inplace=False)
    )

    circuits["Method 1 K = 5 Inverse"] = (
        circuits["Method 1 K = 5"]
        .remove_final_measurements(False)
        .inverse()
        .measure_all(inplace=False)
    )

    circuits["Method 1 K = 10 t=1E-9"] = HamiltonianSimulation(
        n_spins=num_qubits,
        K=10,
        t=1e-9,
        hamiltonian="TFIM",
        init_state="checkerboard",
        method=1,
    )

    circuits["Method 3 K = 5"] = HamiltonianSimulation(
        n_spins=num_qubits,
        K=5,
        t=1,
        hamiltonian="TFIM",
        init_state="checkerboard",
        method=3,
    )

    return circuits


def construct_hamsim_circuits(num_qubits):
    from hamiltonian_simulation_kernel import TfimHamiltonianKernel

    circuits = {}

    circuits["Method 1 K = 10"] = TfimHamiltonianKernel(
        num_qubits,
        K=10,
        t=1,
        hamiltonian="TFIM",
        init_state="checkerboard",
        w=1,
        hx=np.ones(100),
        hz=np.ones(100),
        use_XX_YY_ZZ_gates=False,
        method=1,
        random_pauli_flag=False,
    ).overall_circuit()

    circuits["Method 1 K = 5"] = TfimHamiltonianKernel(
        num_qubits,
        K=5,
        t=1,
        hamiltonian="TFIM",
        init_state="checkerboard",
        w=1,
        hx=np.ones(100),
        hz=np.ones(100),
        use_XX_YY_ZZ_gates=False,
        method=1,
        random_pauli_flag=False,
    ).overall_circuit()

    circuits["Method 1 K = 10 Inverse"] = (
        circuits["Method 1 K = 10"]
        .remove_final_measurements(False)
        .inverse()
        .measure_all(inplace=False)
    )

    circuits["Method 1 K = 5 Inverse"] = (
        circuits["Method 1 K = 5"]
        .remove_final_measurements(False)
        .inverse()
        .measure_all(inplace=False)
    )

    circuits["Method 1 K = 10 t=1E-9"] = TfimHamiltonianKernel(
        num_qubits,
        K=10,
        t=1e-9,
        hamiltonian="TFIM",
        init_state="checkerboard",
        w=1,
        hx=np.ones(100),
        hz=np.ones(100),
        use_XX_YY_ZZ_gates=False,
        method=1,
        random_pauli_flag=False,
    ).overall_circuit()

    circuits["Method 3 K = 5"] = TfimHamiltonianKernel(
        num_qubits,
        K=5,
        t=1,
        hamiltonian="TFIM",
        init_state="checkerboard",
        w=1,
        hx=np.ones(100),
        hz=np.ones(100),
        use_XX_YY_ZZ_gates=False,
        method=3,
        random_pauli_flag=False,
    ).overall_circuit()

    qc = (
        TfimHamiltonianKernel(
            num_qubits,
            K=1,
            t=1 / 5,
            hamiltonian="TFIM",
            init_state="checkerboard",
            w=1,
            hx=np.ones(100),
            hz=np.ones(100),
            use_XX_YY_ZZ_gates=False,
            method=1,
            random_pauli_flag=False,
        )
        .overall_circuit()
        .remove_final_measurements(False)
    )

    qc2 = TfimHamiltonianKernel(
        num_qubits,
        K=1,
        t=1 / 5,
        hamiltonian="TFIM",
        init_state="checkerboard",
        w=1,
        hx=np.ones(100),
        hz=np.ones(100),
        use_XX_YY_ZZ_gates=False,
        method=1,
        random_pauli_flag=False,
    ).create_hamiltonian()

    qc3 = (
        TfimHamiltonianKernel(
            num_qubits,
            K=1,
            t=1 / 5,
            hamiltonian="TFIM",
            init_state="checkerboard",
            w=1,
            hx=np.ones(100),
            hz=np.ones(100),
            use_XX_YY_ZZ_gates=False,
            method=1,
            random_pauli_flag=False,
        )
        .create_hamiltonian()
        .inverse()
    )

    qc.append(qc3, range(num_qubits))

    for _ in range(9):
        qc.append(qc2, range(num_qubits))
        qc.append(qc3, range(num_qubits))

    qc.measure_all()

    # print(qc)

    circuits["Method 1 K = 10 Inverse Sandwich"] = qc

    return circuits


def get_circuit_metrics(transpiled_qc):
    qc_depth, qc_size, qc_count_ops, qc_xi, qc_n2q = ex.get_circuit_metrics(
        transpiled_qc
    )
    return {
        "depth": qc_depth,
        "size": qc_size,
        "count_ops": qc_count_ops,
        "xi": qc_xi,
        "n2q": qc_n2q,
    }


def run_experiment(qc, num_shots, noise_model):
    backend = AerSimulator(noise_model=noise_model, basis_gates=ex.basis_gates_array[1])
    transpiled_qc = transpile(qc, backend, optimization_level=0)
    circuit_metrics = get_circuit_metrics(transpiled_qc)
    job = backend.run(transpiled_qc, shots=num_shots)
    result = job.result()
    counts = result.get_counts(qc)
    return counts, circuit_metrics


def save_results(data, filename="results.json"):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def calculate_fidelity(counts, correct_dist):
    return metrics.polarization_fidelity(counts, correct_dist)


if __name__ == "__main__":
    hamiltonian = "TFIM"
    init_state = "checkerboard"
    # hamlib_simulation_kernel.filename = create_full_filenames(hamiltonian)
    # hamlib_simulation_kernel.dataset_name_template = construct_dataset_name(hamlib_simulation_kernel.filename)

    min_qubits = 4
    max_qubits = 12
    num_shots = 1000000
    skip_qubits = 1
    # valid_qubits = get_valid_qubits(min_qubits, max_qubits, skip_qubits)
    #
    valid_qubits = range(2, 12)

    experiment_data = {}

    for num_qubits in tqdm(valid_qubits):
        experiment_data[num_qubits] = {"metrics": {}, "experiments": {}}

        # Construct circuits
        circuits = construct_hamsim_circuits(num_qubits)

        # Experiment for each method
        for method, circuit in tqdm(circuits.items(), leave=False):
            # With noise
            noise_model = ex.default_noise_model()
            noisy_counts, circuit_metrics = run_experiment(
                circuit, num_shots, noise_model
            )

            experiment_data[num_qubits]["metrics"][method] = circuit_metrics

            # Without noise
            noise_model = None
            correct_dist, _ = run_experiment(circuit, num_shots, noise_model)
            fidelity = calculate_fidelity(noisy_counts, correct_dist)

            # Store results
            experiment_data[num_qubits]["experiments"][method] = {
                "num_shots": num_shots,
                "noisy_counts": noisy_counts,
                "correct_dist": correct_dist,
                "fidelity": fidelity,
            }

    # Save all results to a JSON file
    save_results(experiment_data, filename="hamsim.json")

    # this codew isc urrently set to generate only the hamsim, not the hamlib results.
