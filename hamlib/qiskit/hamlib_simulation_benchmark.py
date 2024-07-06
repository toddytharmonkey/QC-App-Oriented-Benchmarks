'''
Hamiltonian Simulation Benchmark Program - Qiskit
(C) Quantum Economic Development Consortium (QED-C) 2024.
'''

'''
This program benchmarks Hamiltonian simulation using Qiskit. 
The central function is the `run()` method, which orchestrates the entire benchmarking process.

HamiltonianSimulation forms the trotterized circuit used in the benchmark.

HamiltonianSimulationExact runs a classical calculation that perfectly simulates hamiltonian evolution, although it does not scale well. 
'''

import json
import os
import sys
import time
import numpy as np

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
sys.path[1:1] = ["../_common"]

import execute as ex
import metrics as metrics

import hamlib_simulation_kernel
from hamlib_simulation_kernel import HamiltonianSimulation, kernel_draw, get_valid_qubits
from hamlib_utils import create_full_filenames, construct_dataset_name
from hamiltonian_simulation_exact import HamiltonianSimulationExact, HamiltonianSimulationExact_Noiseless


# Benchmark Name
benchmark_name = "Hamiltonian Simulation"

np.random.seed(0)

verbose = False


# Import precalculated data to compare against
filename = os.path.join(os.path.dirname(__file__), os.path.pardir, "_common", "precalculated_data.json")
with open(filename, 'r') as file:
    data = file.read()
precalculated_data = json.loads(data)

# Creates a key for distribution of initial state for method = 3.
def key_from_initial_state(num_qubits, num_shots, init_state, random_pauli_flag):
    """
    Generates a dictionary representing the correct distribution of quantum states based on the initial state configuration.

    This function supports generating specific patterns or distributions for different initial quantum states like
    'checkerboard' and 'ghz'. Depending on the initial state configuration, it may also factor in the effect of random
    Pauli operations applied across the qubits.

    Args:
        num_qubits (int): The number of qubits in the quantum system.
        num_shots (int): The number of measurements or shots to simulate.
        init_state (str): The type of initial state to configure. Supported values are 'checkerboard' and 'ghz'.
        random_pauli_flag (bool): Flag to indicate if random Pauli operations are considered.

    Returns:
        dict: A dictionary where keys are bit strings representing quantum states, and values are the counts
              (or probabilities) of these states occurring.
    """
    def generate_pattern(starting_bit):
        # Generate a bit pattern that alternates, starting from the 'starting_bit'
        pattern = ''.join([str((i + starting_bit) % 2) for i in range(num_qubits)])
        return pattern

    correct_dist = {}

    if init_state == "checkerboard":
        if random_pauli_flag:
            starting_bit = 0 if num_qubits % 2 != 0 else 1
        else:
            starting_bit = 1 if num_qubits % 2 != 0 else 0

        correct_dist[generate_pattern(starting_bit)] = num_shots
    elif init_state == "ghz":
        correct_dist = {
            '0' * num_qubits: num_shots/2,
            '1' * num_qubits: num_shots/2
        }

    return correct_dist


############### Result Data Analysis

#def analyze_and_print_result(qc: QuantumCircuit, result, num_qubits: int,
def analyze_and_print_result(qc, result, num_qubits: int,
            type: str, num_shots: int, hamiltonian: str, method: int, random_pauli_flag: bool, init_state: str) -> tuple:
    """
    Analyze and print the measured results. Compute the quality of the result based on operator expectation for each state.

    Args:
        qc (QuantumCircuit): The quantum circuit.
        result: The result from the execution.
        num_qubits (int): Number of qubits.
        type (str): Type of the simulation.
        num_shots (int): Number of shots.
        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM" or "hamlib". 
        method (int): Method for fidelity checking (1 for noiseless trotterized quantum, 2 for exact classical), 3 for mirror circuit.

    Returns:
        tuple: Counts and fidelity.
    """
    counts = result.get_counts(qc)

    if verbose:
        #print(f"For type {type} measured: {counts}")
        print_top_measurements(f"For type {type} measured counts = ", counts, 100)

    hamiltonian = hamiltonian.strip().lower()

    # calculate correct distribution on the fly
    if method == 1:
        if verbose:
            print(f"... begin noiseless simulation for expected distribution ...")
        ts = time.time()
        correct_dist = HamiltonianSimulationExact_Noiseless(n_spins=num_qubits,init_state=init_state)
        if verbose:
            print(f"... noiseless simulation for expected distribution time = {round((time.time() - ts), 3)} sec") 
            
    elif method == 2:
        if verbose:
            print(f"... begin exact computation ...")
        ts = time.time()
        correct_dist = HamiltonianSimulationExact(n_spins=num_qubits,init_state=init_state)
        if verbose:
            print(f"... exact computation time = {round((time.time() - ts), 3)} sec") 
            
    elif method == 3:
        correct_dist = key_from_initial_state(num_qubits, num_shots, init_state, random_pauli_flag)
        
    else:
        raise ValueError("Method is not 1 or 2 or 3, or hamiltonian is not valid.")

    if verbose:
        #print(f"Correct dist: {correct_dist}")
        print_top_measurements(f"Correct dist = ", correct_dist, 100)

    # Use polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, correct_dist)
    
    return counts, fidelity
    
def print_top_measurements(label, counts, top_n):
    """
    Prints the top N measurements from a Qiskit measurement counts dictionary
    in a dictionary-like format. If there are more measurements not printed,
    indicates the count.
    
    Args:
        counts (dict): The measurement counts dictionary from Qiskit.
        top_n (int): The number of top measurements to print.
    """
    
    if label is not None: print(label)
    
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    
    total_measurements = len(sorted_counts)
    
    if top_n >= total_measurements:
        top_counts = sorted_counts
        more_counts = []
    else:
        top_counts = sorted_counts[:top_n]
        more_counts = sorted_counts[top_n:]
    
    print("{", end=" ")
    for i, (measurement, count) in enumerate(top_counts):
        print(f"'{measurement}': {round(count,6)}", end="")
        if i < len(top_counts) - 1:
            print(",", end=" ")
    
    if more_counts:
        num_more = len(more_counts)
        print(f", ... and {num_more} more.")
    else:
        print(" }")


############### Benchmark Loop

def run(min_qubits: int = 2, max_qubits: int = 8, max_circuits: int = 3,
        skip_qubits: int = 1, num_shots: int = 100,
        hamiltonian: str = "TFIM", method: int = 1,
        use_XX_YY_ZZ_gates: bool = False, random_pauli_flag: bool = False, init_state: str = None,
        backend_id: str = None, provider_backend = None,
        hub: str = "ibm-q", group: str = "open", project: str = "main", exec_options = None,
        context = None, api = None):
    """
    Execute program with default parameters.

    Args:
        min_qubits (int): Minimum number of qubits (smallest circuit is 2 qubits).
        max_qubits (int): Maximum number of qubits.
        max_circuits (int): Maximum number of circuits to execute per group.
        skip_qubits (int): Increment of number of qubits.
        num_shots (int): Number of shots for each circuit execution.
        use_XX_YY_ZZ_gates (bool): Flag to use unoptimized XX, YY, ZZ gates.
        backend_id (str): Backend identifier for execution.
        provider_backend: Provider backend instance.
        hub (str): IBM Quantum hub.
        group (str): IBM Quantum group.
        project (str): IBM Quantum project.
        exec_options: Execution options.

        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM". 
        method (int): Method for fidelity checking (1 for noiseless trotterized quantum, 2 for exact classical), 3 for mirror circuit.
        context: Execution context.

    Returns:
        None
    """
    
    print(f"{benchmark_name} Benchmark Program - Qiskit")
    
    # Create context identifier
    if context is None: context = f"{benchmark_name} Benchmark"
    
    # Validate parameters (smallest circuit is 2 qubits)
    max_qubits = max(2, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)
    if min_qubits % 2 == 1: min_qubits += 1  # min_qubits must be even (DEVNOTE: is this True?)
    skip_qubits = max(1, skip_qubits)

    # get key infomation about the selected Hamiltonian
    # DEVNOTE: Error handling here can be improved by simply returning False or raising exception
    try:
        hamlib_simulation_kernel.filename = create_full_filenames(hamiltonian)
        hamlib_simulation_kernel.dataset_name_template = construct_dataset_name(hamlib_simulation_kernel.filename)
    except ValueError:
        print(f"ERROR: cannot load HamLib data for Hamiltonian: {hamiltonian}")
        return
    
    if hamlib_simulation_kernel.dataset_name_template == "File key not found in data":
        print(f"ERROR: cannot load HamLib data for Hamiltonian: {hamiltonian}")
        return
        
    # assume default init_state if not given
    if init_state == None:
        init_state = "checkerboard"
        
    # Set the flag to use an XX YY ZZ shim if given
    if use_XX_YY_ZZ_gates:
        print("... using unoptimized XX YY ZZ gates")
    
    ################################
    
    # Initialize metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, num_qubits, type, num_shots):
        # Determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, expectation_a = analyze_and_print_result(qc, result, num_qubits, type, num_shots, hamiltonian, method, random_pauli_flag, init_state)
        metrics.store_metric(num_qubits, type, 'fidelity', expectation_a)

    # Initialize execution module using the execution result handler above and specified backend_id
    ex.init_execution(execution_handler)
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options,
            context=context)

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete

    # comment out the normal way to doing this
    # for num_qubits in range(min_qubits, max_qubits + 1, skip_qubits):
    
    # for HamLib, determine available widths and loop over those 
    valid_qubits = get_valid_qubits(min_qubits, max_qubits, skip_qubits)
    for num_qubits in valid_qubits:
    
        # Reset random seed
        np.random.seed(0)

        # Determine number of circuits to execute for this group
        num_circuits = max(1, max_circuits)
        
        print(f"************\nExecuting [{num_circuits}] circuits with num_qubits = {num_qubits}")

        # Parameters of simulation
        #### CANNOT BE MODIFIED W/O ALSO MODIFYING PRECALCULATED DATA #########
        # DEVNOTE: HamLib is calculating expectation on the fly, so these could be set rather than pulled
        
        # A large Trotter order approximates the Hamiltonian evolution better.
        # But a large Trotter order also means the circuit is deeper.
        # For ideal or noise-less quantum circuits, k >> 1 gives perfect Hamiltonian simulation.
        w = precalculated_data['w']  # Strength of disorder
        k = precalculated_data['k']   # Trotter error.
        t = precalculated_data['t']  # Time of simulation
        
        # Precalculated random numbers between [-1, 1]
        hx = precalculated_data['hx'][:num_qubits]
        hz = precalculated_data['hz'][:num_qubits]
        #######################################################################

        # Loop over only 1 circuit
        for circuit_id in range(num_circuits):
            ts = time.time()
            
            # create the HamLibSimulation kernel
            qc = HamiltonianSimulation(num_qubits, K=k, t=t,
                    hamiltonian=hamiltonian, init_state=init_state,
                    w=w, hx = hx, hz = hz, 
                    use_XX_YY_ZZ_gates = use_XX_YY_ZZ_gates,
                    method = method)
                    
            metrics.store_metric(num_qubits, circuit_id, 'create_time', time.time() - ts)

            # Submit circuit for execution on target (simulator, cloud simulator, or hardware)
            ex.submit_circuit(qc, num_qubits, circuit_id, num_shots)
        
        # Wait for some active circuits to complete; report metrics when groups complete
        ex.throttle_execution(metrics.finalize_group)
    
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)

    ##########
    
    # draw a sample circuit
    kernel_draw(hamiltonian, use_XX_YY_ZZ_gates, method)
       
    # Plot metrics for all circuit sizes
    options = {"ham": hamiltonian, "method":method, "shots": num_shots, "reps": max_circuits}
    if use_XX_YY_ZZ_gates: options.update({ "xyz": use_XX_YY_ZZ_gates })
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} - Qiskit", options=options)


#######################
# MAIN

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Bernstei-Vazirani Benchmark")
    #parser.add_argument("--api", "-a", default=None, help="Programming API", type=str)
    #parser.add_argument("--target", "-t", default=None, help="Target Backend", type=str)
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--num_shots", "-s", default=100, help="Number of shots", type=int)
    parser.add_argument("--num_qubits", "-n", default=0, help="Number of qubits (min = max = N)", type=int)
    parser.add_argument("--min_qubits", "-min", default=3, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)     
    parser.add_argument("--hamiltonian", "-ham", default="TFIM", help="Name of Hamiltonian", type=str)
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--use_XX_YY_ZZ_gates", action="store_true", help="Use explicit XX, YY, ZZ gates")
    #parser.add_argument("--theta", default=0.0, help="Input Theta Value", type=float)
    parser.add_argument("--nonoise", "-non", action="store_true", help="Use Noiseless Simulator")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--random_pauli_flag", "-ranp", action="store_true", help="random pauli flag")
    parser.add_argument("--init_state", "-init", default=None, help="initial state")
    return parser.parse_args()
 
# if main, execute method
if __name__ == '__main__':   
    args = get_args()
    
    # configure the QED-C Benchmark package for use with the given API
    # (done here so we can set verbose for now)
    #PhaseEstimation, kernel_draw = qedc_benchmarks_init(args.api)
    
    # special argument handling
    ex.verbose = args.verbose
    verbose = args.verbose
    hamlib_simulation_kernel.verbose = args.verbose
    
    if args.num_qubits > 0: args.min_qubits = args.max_qubits = args.num_qubits
    
    # execute benchmark program
    run(min_qubits=args.min_qubits, max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits, max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        hamiltonian=args.hamiltonian,
        method=args.method,
        random_pauli_flag=args.random_pauli_flag,
        use_XX_YY_ZZ_gates = args.use_XX_YY_ZZ_gates,
        init_state = args.init_state,
        #theta=args.theta,
        backend_id=args.backend_id,
        exec_options = {"noise_model" : None} if args.nonoise else {},
        #api=args.api
        )

