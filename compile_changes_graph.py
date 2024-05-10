from qiskit_aer.noise import NoiseModel, depolarizing_error
import sys
import qft_benchmark
import ae_benchmark
import pe_benchmark
import mc_benchmark
import hamiltonian_simulation_benchmark
import vqe_benchmark
import tket_optimiser as tket_optimiser  
import execute as ex
import metrics as metrics


sys.path.insert(1, "_common/transformers/tket_optimiser")
sys.path.insert(1, "vqe/qiskit")
sys.path.insert(1, "hamiltonian-simulation/qiskit")
sys.path.insert(1, "quantum-fourier-transform/qiskit")
sys.path.insert(1, "amplitude-estimation/qiskit")
sys.path.insert(1, "phase-estimation/qiskit")
sys.path.insert(1, "monte-carlo/qiskit")


apps = [
    "Quantum Fourier Transform (1)",
    "Phase Estimation",
    "Amplitude Estimation",
    "Monte Carlo Sampling (2)",
    "VQE Simulation (1)",
    "Hamiltonian Simulation",
]

run_options_dict = {
    "min_qubits": 4,
    "max_qubits": 4,
    "max_circuits": 3,
    "num_shots": "1000",
    "backend_id": "qasm_simulator",
    "hub": "ibm-q",
    "group": "open",
    "project": "main",
    "provider_backend": None,
}

# these are all of the benchmark in the Charlie Baldwin graph
benchmarks = [
    qft_benchmark,
    ae_benchmark,
    pe_benchmark,
    mc_benchmark,
    hamiltonian_simulation_benchmark,
    vqe_benchmark,
]

def create_noise_model(fidelity):
        p = 15 / 4 * (1 - fidelity)
        noise_model = NoiseModel()
        depolarizing_err = depolarizing_error(p, 2)  # 2-qubit depolarizing error
        noise_model.add_all_qubit_quantum_error(
            depolarizing_err, ["cx"]
        )  # Apply to CNOT gates
        return noise_model

if __name__ == "__main__":

    # This script is copy/pasted from the hamiltonian_simulation_benchmark.py file
    # Go through 2Q fidelities .95 and .995
    # and go through using pytket (or compiling) in the loop below.

    fidelities = [0.95, 0.995]

    for benchmark in benchmarks: 
        for method in [2]:
            for f in fidelities:
                for use_pytket in [False]:

                    noise = create_noise_model(f)
                    ex.set_noise_model(noise)

                    print(
                        f"Starting to run benchmarks with {method}, use_pytket: {use_pytket}, 2Q error rate set to {f}"
                    )

                    # not really sure how tket optimiser works, so just use default settings
                    high_optimisation = tket_optimiser.tket_transformer_generator(
                        cx_fidelity=f
                    )
                    if use_pytket:
                        exec_options = {
                            "optimization_level": 0,
                            "layout_method": "sabre",
                            "routing_method": "sabre",
                            "transformer": high_optimisation,
                        }
                    else:
                        exec_options = None

                    benchmarks.run(**run_options_dict, exec_options=exec_options)

        metrics.plot_all_app_metrics(run_options_dict["backend_id"], do_all_plots=False, include_apps=apps)
