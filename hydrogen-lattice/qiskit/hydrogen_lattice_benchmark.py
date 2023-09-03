
"""
Hydogen Lattice Benchmark Program - Qiskit
"""

import datetime
import json
import logging
import math
import os
import re
import sys
import time
from collections import namedtuple
from typing import Any, List, Optional
from pathlib import Path
from qiskit.opflow.primitive_ops import PauliSumOp
import matplotlib.pyplot as plt
import glob

import numpy as np
from scipy.optimize import minimize

from qiskit import (Aer, ClassicalRegister,  # for computing expectation tables
                    QuantumCircuit, QuantumRegister, execute, transpile)
from qiskit.circuit import ParameterVector
from typing import Dict, List, Optional
from qiskit import Aer, execute
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.opflow import ComposedOp, PauliExpectation, StateFn, SummedOp
from qiskit.quantum_info import Statevector,Pauli
from qiskit.result import sampled_expectation_value

sys.path[1:1] = [ "_common", "_common/qiskit", "hydrogen-lattice/_common" ]
sys.path[1:1] = [ "../../_common", "../../_common/qiskit", "../../hydrogen-lattice/_common/" ]

import common
import execute as ex
import metrics as metrics
from matplotlib import cm

# import h-lattice_metrics from _common folder
import h_lattice_metrics as h_metrics

# DEVNOTE: this logging feature should be moved to common level
logger = logging.getLogger(__name__)
fname, _, ext = os.path.basename(__file__).partition(".")
log_to_file = False

try:
    if log_to_file:
        logging.basicConfig(
            # filename=f"{fname}_{datetime.datetime.now().strftime('%Y_%m_%d_%S')}.log",
            filename=f"{fname}.log",
            filemode='w',
            encoding='utf-8',
            level=logging.INFO,
            format='%(asctime)s %(name)s - %(levelname)s:%(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s %(name)s - %(levelname)s:%(message)s')
        
except Exception as e:
    print(f'Exception {e} occured while configuring logger: bypassing logger config to prevent data loss')
    pass

np.random.seed(0)

# Hydrogen Lattice inputs  ( Here input is Hamiltonian matrix --- Need to change)
hl_inputs = dict() #inputs to the run method
verbose = False
print_sample_circuit = True

# Indicates whether to perform the (expensive) pre compute of expectations
do_compute_expectation = True

# saved circuits for display
QC_ = None
Uf_ = None

# #theta parameters
vqe_parameter = namedtuple('vqe_parameter','theta')

# DEBUG prints
# give argument to the python script as "debug" or "true" or "1" to enable debug prints
if len(sys.argv) > 1:
    DEBUG = (sys.argv[1].lower() in ['debug', 'true', '1'])
else:
    DEBUG = False

# Add custom metric names to metrics module
def add_custom_metric_names():
    
    metrics.known_x_labels.update({
        'iteration_count' : 'Iterations'
    })
    metrics.known_score_labels.update({
        'solution_quality' : 'Solution Quality',
        'accuracy_volume' : 'Accuracy Volume',
        'accuracy_ratio' : 'Accuracy Ratio',
        'energy' : 'Energy (Hartree)'  
    })
    metrics.score_label_save_str.update({
        'solution_quality' : 'solution_quality', 
        'accuracy_volume' : 'accuracy_volume', 
        'accuracy_ratio' : 'accuracy_ratio',
        'energy' : 'energy'
    })

###################################
# HYDROGEN LATTICE CIRCUIT

# parameter mode to control length of initial thetas_array (global during dev phase)
# 1 - length 1
# 2 - length N, where N is number of excitation pairs
saved_parameter_mode = 1  

def get_initial_parameters(num_qubits: int, thetas_array):
    '''
    Generate an initial set of parameters given the number of qubits and user-provided thetas_array.
    If thetas_array is None, generate random array of parameters based on the parameter_mode
    If thetas_array is of length 1, repeat to the required length based on parameter_mode
    Otherwise, use the provided thetas_array.
    
    Parameters
    ----------
    num_qubits : int
        number of qubits in circuit 
    thetas_array : array of floats
        user-supplied array of initial values

    Returns
    -------
    initial_parameters : array
        array of parameter values required
    '''
    
    # compute required size of array based on number of occupation pairs
    size = 1
    if saved_parameter_mode > 1:
        num_occ_pairs = (num_qubits // 2)
        size = num_occ_pairs**2
    
    # if None passed in, create array of random values
    if thetas_array == None:
        initial_parameters = np.random.random(size=size)
    
    # if single value passed in, extend to required size
    elif size > 1 and len(thetas_array) == 1:
        initial_parameters = np.repeat(thetas_array, size)
    
    # otherwise, use what user provided
    else:
        if len(thetas_array) != size:
            print(f"WARNING: length of thetas_array {len(thetas_array)} does not equal required length {size}")
            print("         Generating random values instead.")
            initial_parameters = get_initial_parameters(num_qubits, None)
        else:  
            initial_parameters = np.array(thetas_array)
        
    if verbose:
        print(f"... get_initial_parameters(num_qubits={num_qubits}, mode={saved_parameter_mode})")
        print(f"    --> initial_parameter[{size}]={initial_parameters}")
        
    return initial_parameters
                
# Create the  ansatz quantum circuit for the VQE algorithm.
def VQE_ansatz(num_qubits: int,
            thetas_array,
            parameterized,
            num_occ_pairs: Optional[int] = None,
            *args, **kwargs) -> QuantumCircuit:
    
    if verbose:    
        print(f"  ... VQE_ansatz(num_qubits={num_qubits}, thetas_array={thetas_array}")  
    
    # Generate the ansatz circuit for the VQE algorithm.
    if num_occ_pairs is None:
        num_occ_pairs = (num_qubits // 2)  # e.g., half-filling, which is a reasonable chemical case

    # do all possible excitations if not passed a list of excitations directly
    excitation_pairs = []
    for i in range(num_occ_pairs):
        for a in range(num_occ_pairs, num_qubits):
            excitation_pairs.append([i, a])
    
    # create circuit of num_qubits
    circuit = QuantumCircuit(num_qubits)
    
    # Hartree Fock initial state
    for occ in range(num_occ_pairs):
        circuit.x(occ)
    
    # if parameterized flag set, create a ParameterVector
    parameter_vector = None
    if parameterized:
        parameter_vector = ParameterVector("t", length=len(excitation_pairs))
    
    # for parameter mode 1, make all thetas the same as the first
    if saved_parameter_mode == 1:
        thetas_array = np.repeat(thetas_array, len(excitation_pairs))
    
    # create a Hartree Fock initial state
    for idx, pair in enumerate(excitation_pairs):
        
        # if parameterized, use ParamterVector, otherwise raw theta value
        theta = parameter_vector[idx] if parameterized else thetas_array[idx]
        
        # apply excitation
        i, a = pair[0], pair[1]

        # implement the magic gate
        circuit.s(i)
        circuit.s(a)
        circuit.h(a)
        circuit.cx(a, i)

        # Ry rotation
        circuit.ry(theta, i)
        circuit.ry(theta, a)

        # implement M^-1
        circuit.cx(a, i)
        circuit.h(a)
        circuit.sdg(a)
        circuit.sdg(i)

    ''' TMI ...
    if verbose:
        print(f"      --> thetas_array={thetas_array}")
        print(f"      --> parameter_vector={str(parameter_vector)}") 
    '''
    return circuit, parameter_vector, thetas_array
    
# Create the benchmark program circuit array, for the given operator
def HydrogenLattice (num_qubits, operator, secret_int = 000000,
            thetas_array = None, parameterized = None):
    
    if verbose:    
        print(f"... HydrogenLattice(num_qubits={num_qubits}, thetas_array={thetas_array}") 
        
    # if no thetas_array passed in, create defaults 
    if thetas_array is None:
        thetas_array = [1.0]
        
    # create parameters in the form expected by the ansatz generator
    # this is an array of betas followed by array of gammas, each of length = rounds
    global _qc
    global theta
    
    # create the circuit the first time, add measurements
    if ex.do_transpile_for_execute:
        logger.info(f'*** Constructing parameterized circuit for {num_qubits = } {secret_int}')
 
        _qc, parameter_vector, thetas_array = VQE_ansatz(num_qubits=num_qubits,
                thetas_array=thetas_array, parameterized=parameterized,
                num_occ_pairs = None )
    
    _measurable_expression = StateFn(operator, is_measurement=True)
    _observables = PauliExpectation().convert(_measurable_expression)
    _qc_array, _formatted_observables = prepare_circuits(_qc, observables=_observables)

    # create a binding of Parameter values
    params = {parameter_vector: thetas_array} if parameterized else None
    
    if verbose:
        print(f"    --> params={params}")
        
    logger.info(f"Create binding parameters for {thetas_array} {params}")
    
    # save the first circuit in the array returned from prepare_circuits (with appendage)
    # to be used an example for display purposes
    qc = _qc_array[0]
    #print(qc)
    
    # save small circuit example for display
    global QC_
    if QC_ == None or num_qubits <= 4:
        if num_qubits <= 7: QC_ = qc

    # return a handle on the circuit, the observables, and parameters
    return _qc_array, _formatted_observables, params
   
############### Prepare Circuits from Observables

def prepare_circuits(base_circuit, observables):
    """
    Prepare the qubit-wise commuting circuits for a given operator.
    """
    circuits = list()

    if isinstance(observables, ComposedOp):
        observables = SummedOp([observables])
        
    for obs in observables:
        circuit = base_circuit.copy()
        circuit.append(obs[1], qargs=list(range(base_circuit.num_qubits)))
        circuit.measure_all()
        circuits.append(circuit)
        
    return circuits, observables

def compute_energy(result_array, formatted_observables, num_qubits): 
    """
    Compute the expectation value of the circuit with respect to the Hamiltonian for optimization
    """
    
    _probabilities = list()

    for _res in result_array:
        _counts = _res.get_counts()
        _probs = normalize_counts(_counts, num_qubits=num_qubits)
        _probabilities.append(_probs)

    _expectation_values = calculate_expectation_values(_probabilities, formatted_observables)

    energy = sum(_expectation_values)

    return energy    

def calculate_expectation_values(probabilities, observables):
    """
    Return the expectation values for an operator given the probabilities.
    """
    expectation_values = list()
    for idx, op in enumerate(observables):
        expectation_value = sampled_expectation_value(probabilities[idx], op[0].primitive)
        expectation_values.append(expectation_value)

    return expectation_values

def normalize_counts(counts, num_qubits=None):
    """
    Normalize the counts to get probabilities and convert to bitstrings.
    """
    
    normalizer = sum(counts.values())

    try:
        dict({str(int(key, 2)): value for key, value in counts.items()})
        if num_qubits is None:
            num_qubits = max(len(key) for key in counts)
        bitstrings = {key.zfill(num_qubits): value for key, value in counts.items()}
    except ValueError:
        bitstrings = counts

    probabilities = dict({key: value / normalizer for key, value in bitstrings.items()})
    assert abs(sum(probabilities.values()) - 1) < 1e-9
    return probabilities

#################################################
# EXPECTED RESULT TABLES (METHOD 1)
  
############### Expectation Tables Created using State Vector Simulator

# DEVNOTE: We are building these tables on-demand for now, but for larger circuits
# this will need to be pre-computed ahead of time and stored in a data file to avoid run-time delays.

# dictionary used to store pre-computed expectations, keyed by num_qubits and secret_string
# these are created at the time the circuit is created, then deleted when results are processed
expectations = {}

# Compute array of expectation values in range 0.0 to 1.0
# Use statevector_simulator to obtain exact expectation
def compute_expectation(qc, num_qubits, secret_int, backend_id='statevector_simulator', params=None):
    
    #ts = time.time()
    
    # to execute on Aer state vector simulator, need to remove measurements
    qc = qc.remove_final_measurements(inplace=False)
    
    if params != None:
        qc = qc.bind_parameters(params)
    
    #execute statevector simulation
    sv_backend = Aer.get_backend(backend_id)
    sv_result = execute(qc, sv_backend, params=params).result()

    # get the probability distribution
    counts = sv_result.get_counts()

    #print(f"... statevector expectation = {counts}")
    
    # store in table until circuit execution is complete
    id = f"_{num_qubits}_{secret_int}"
    expectations[id] = counts
    
    #print(f"  ... time to execute statevector simulator: {time.time() - ts}")
    
# Return expected measurement array scaled to number of shots executed
def get_expectation(num_qubits, secret_int, num_shots):

    # find expectation counts for the given circuit 
    id = f"_{num_qubits}_{secret_int}"
    
    if id in expectations:
        counts = expectations[id]
        
        # scale probabilities to number of shots to obtain counts
        for k, v in counts.items():
            counts[k] = round(v * num_shots)
        
        # delete from the dictionary
        del expectations[id]
        
        return counts
        
    else:
        return None
      

#################################################
# RESULT DATA ANALYSIS (METHOD 1)

expected_dist = {}

# Compare the measurement results obtained with the expected measurements to determine fidelity
def analyze_and_print_result (qc, result, num_qubits, secret_int, num_shots):
    global expected_dist
    
    # obtain counts from the result object
    counts = result.get_counts(qc)
    
    # retrieve pre-computed expectation values for the circuit that just completed
    expected_dist = get_expectation(num_qubits, secret_int, num_shots)
    
    # if the expectation is not being calculated (only need if we want to compute fidelity)
    # assume that the expectation is the same as measured counts, yielding fidelity = 1
    if expected_dist == None:
        expected_dist = counts
    
    if verbose: print(f"For width {num_qubits}   measured: {counts}\n  expected: {expected_dist}")
    # if verbose: print(f"For width {num_qubits} problem {secret_int}\n  measured: {counts}\n  expected: {expected_dist}")

    # use our polarization fidelity rescaling
    fidelity = metrics.polarization_fidelity(counts, expected_dist)
    
    # if verbose: print(f"For secret int {secret_int} fidelity: {fidelity}")    
    
    return counts, fidelity

##### METHOD 2 function to compute application-specific figures of merit

def calculate_quality_metric(energy=None,
            fci_energy=0,
            random_energy=0, 
            precision = 4,
            num_electrons = 2):
    """
    Returns the quality metrics, namely solution quality, accuracy volume, and accuracy ratio. 
    Solution quality is a value between zero and one. The other two metrics can take any value.

    Parameters
    ----------
    energy : list
        list of energies calculated for each iteration.

    fci_energy : float
        FCI energy for the problem.

    random_energy : float
        Random energy for the problem, precomputed and stored and read from json file   

    precision : float
        precision factor used in solution quality calculation
        changes the behavior of the monotonic arctan function

    num_electrons : int
        number of electrons in the problem
    """
    _relative_energy = np.absolute(np.divide(np.subtract( np.array(energy), fci_energy), fci_energy))
    
    #scale the solution quality to 0 to 1 using arctan 
    _solution_quality = np.subtract(1, np.divide(np.arctan(np.multiply(precision,_relative_energy)), np.pi/2))

    # define accuracy volume as the absolute energy difference between the FCI energy and the energy of the solution normalized per electron
    _accuracy_volume = np.divide(np.absolute(np.subtract( np.array(energy), fci_energy)), num_electrons)

    # define accuracy ratio as the difference between calculated energy and random energy divided by difference in random energy and FCI energy
    _accuracy_ratio = np.divide(np.subtract( np.array(energy), random_energy), np.subtract(fci_energy, random_energy))

    return _solution_quality, _accuracy_volume, _accuracy_ratio


#################################################
# DATA SAVE FUNCTIONS

# Create a folder where the results will be saved. 
# For every circuit width, metrics will be stored the moment the results are obtained
# In addition to the metrics, the parameter values obtained by the optimizer, as well as the counts
# measured for the final circuit will be stored.
def create_data_folder(save_res_to_file, detailed_save_names, backend_id):
    global parent_folder_save
    
    # if detailed filenames requested, use directory name with timestamp
    if detailed_save_names:
        start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        parent_folder_save = os.path.join('__data', f'{backend_id}', f'run_start_{start_time_str}')
        
    # otherwise, just put all json files under __data/backend_id
    else:
        parent_folder_save = os.path.join('__data', f'{backend_id}')
    
    # create the folder if it doesn't exist already
    if save_res_to_file and not os.path.exists(parent_folder_save):
        os.makedirs(os.path.join(parent_folder_save))
        
# Function to save final iteration data to file
def store_final_iter_to_metrics_json(backend_id,
                            num_qubits,
                            radius,
                            instance_num,
                            num_shots,
                            converged_thetas_list,
                            energy,
                            detailed_save_names,
                            dict_of_inputs,
                            save_final_counts,
                            save_res_to_file,
                            _instances=None):
    """
    For a given problem (specified by num_qubits and instance),
    1. For a given restart, store properties of the final minimizer iteration to metrics.circuit_metrics_final_iter, and
    2. Store various properties for all minimizer iterations for each restart to a json file.
    """
    # In order to compare with uniform random sampling, get some samples

    # Store properties of the final iteration, the converged theta values, as well as the known optimal value for the current problem, in metrics.circuit_metrics_final_iter. 
    metrics.store_props_final_iter(num_qubits, instance_num, 'energy', energy)
    metrics.store_props_final_iter(num_qubits, instance_num, 'converged_thetas_list', converged_thetas_list)
    
    # Save final iteration data to metrics.circuit_metrics_final_iter
    # This data includes final counts, cuts, etc.
    if save_res_to_file:
    
        # Save data to a json file
        dump_to_json(parent_folder_save,
                    num_qubits, radius, instance_num,  
                    dict_of_inputs, 
                    converged_thetas_list, 
                    energy,
                    save_final_counts=save_final_counts)

def dump_to_json(parent_folder_save, num_qubits, radius, instance_num, 
                dict_of_inputs, 
                converged_thetas_list,
                energy,
                save_final_counts=False):
    """
    For a given problem (specified by number of qubits and instance_number), 
    save the evolution of various properties in a json file.
    Items stored in the json file: Data from all iterations (iterations), inputs to run program ('general properties'), converged theta values ('converged_thetas_list'), computes results.
    if save_final_counts is True, then also store the distribution counts 
    """
    
    #print(f"... saving data for width={num_qubits} radius={radius} instance={instance_num}")
    if not os.path.exists(parent_folder_save): os.makedirs(parent_folder_save)
    store_loc = os.path.join(parent_folder_save,'width_{}_instance_{}.json'.format(num_qubits, instance_num))
    
    # Obtain dictionary with iterations data corresponding to given instance_num 
    all_restart_ids = list(metrics.circuit_metrics[str(num_qubits)].keys())
    ids_this_restart = [r_id for r_id in all_restart_ids if int(r_id) // 1000 == instance_num]
    iterations_dict_this_restart =  {r_id : metrics.circuit_metrics[str(num_qubits)][r_id] for r_id in ids_this_restart}
    
    # Values to be stored in json file
    dict_to_store = {'iterations' : iterations_dict_this_restart}
    dict_to_store['general_properties'] = dict_of_inputs
    dict_to_store['converged_thetas_list'] = converged_thetas_list
    dict_to_store['energy'] = energy
    #dict_to_store['unif_dict'] = unif_dict
    
    # Also store the value of counts obtained for the final counts
    '''
    if save_final_counts:
        dict_to_store['final_counts'] = iter_dist
        #iter_dist.get_counts()
    ''' 
    
    # Now write the data fo;e
    with open(store_loc, 'w') as outfile:
        json.dump(dict_to_store, outfile)

#################################################
# DATA LOAD FUNCTIONS
       
# %% Loading saved data (from json files)

def load_data_and_plot(folder, backend_id=None, **kwargs):
    """
    The highest level function for loading stored data from a previous run
    and plotting optgaps and area metrics

    Parameters
    ----------
    folder : string
        Directory where json files are saved.
    """
    _gen_prop = load_all_metrics(folder, backend_id=backend_id)
    if _gen_prop != None:
        gen_prop = {**_gen_prop, **kwargs}
        plot_results_from_data(**gen_prop)


def load_all_metrics(folder, backend_id=None):
    """
    Load all data that was saved in a folder.
    The saved data will be in json files in this folder

    Parameters
    ----------
    folder : string
        Directory where json files are saved.

    Returns
    -------
    gen_prop : dict
        of inputs that were used in maxcut_benchmark.run method
    """
    # Note: folder here should be the folder where only the width=... files are stored, and not a folder higher up in the directory
    assert os.path.isdir(folder), f"Specified folder ({folder}) does not exist."
    
    metrics.init_metrics()
    
    list_of_files = os.listdir(folder)
    #print(list_of_files)
    
    # list with elements that are tuples->(width,restartInd,filename)
    width_restart_file_tuples = [(*get_width_restart_tuple_from_filename(fileName), fileName)
                                 for (ind, fileName) in enumerate(list_of_files) if fileName.startswith("width")]  
    
    #sort first by width, and then by restartInd
    width_restart_file_tuples = sorted(width_restart_file_tuples, key=lambda x:(x[0], x[1])) 
    distinct_widths = list(set(it[0] for it in width_restart_file_tuples)) 
    list_of_files = [
        [tup[2] for tup in width_restart_file_tuples if tup[0] == width] for width in distinct_widths
        ]
    
    # connot continue without at least one dataset
    if len(list_of_files) < 1:
        print("ERROR: No result files found")
        return None
        
    for width_files in list_of_files:
        # For each width, first load all the restart files
        for fileName in width_files:
            gen_prop = load_from_width_restart_file(folder, fileName)
      
        # next, do processing for the width
        method = gen_prop['method']
        if method == 2:
            num_qubits, _ = get_width_restart_tuple_from_filename(width_files[0])
            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(str(num_qubits))
          
    # override device name with the backend_id if supplied by caller
    if backend_id != None:
        metrics.set_plot_subtitle(f"Device = {backend_id}")
          
    return gen_prop


# # load data from a specific file

def load_from_width_restart_file(folder, fileName):
    """
    Given a folder name and a file in it, load all the stored data and store the values in metrics.circuit_metrics.
    Also return the converged values of thetas, the final counts and general properties.

    Parameters
    ----------
    folder : string
        folder where the json file is located
    fileName : string
        name of the json file

    Returns
    -------
    gen_prop : dict
        of inputs that were used in maxcut_benchmark.run method
    """
    
    # Extract num_qubits and s from file name
    num_qubits, restart_ind = get_width_restart_tuple_from_filename(fileName)
    print(f"Loading from {fileName}, corresponding to {num_qubits} qubits and restart index {restart_ind}")
    with open(os.path.join(folder, fileName), 'r') as json_file:
        data = json.load(json_file)
        gen_prop = data['general_properties']
        converged_thetas_list = data['converged_thetas_list']
        energy = data['energy']
        if gen_prop['save_final_counts']:
            # Distribution of measured cuts
            final_counts = data['final_counts']
        
        backend_id = gen_prop.get('backend_id')
        metrics.set_plot_subtitle(f"Device = {backend_id}")
        
        # Update circuit metrics
        for circuit_id in data['iterations']:
            # circuit_id = restart_ind * 1000 + minimizer_loop_ind
            for metric, value in data['iterations'][circuit_id].items():
                metrics.store_metric(num_qubits, circuit_id, metric, value)
                
        method = gen_prop['method']
        if method == 2:
            metrics.store_props_final_iter(num_qubits, restart_ind, 'energy', energy)
            metrics.store_props_final_iter(num_qubits, restart_ind, 'converged_thetas_list', converged_thetas_list)
            if gen_prop['save_final_counts']:
                metrics.store_props_final_iter(num_qubits, restart_ind, None, final_counts)

    return gen_prop
    
def get_width_restart_tuple_from_filename(fileName):
    """
    Given a filename, extract the corresponding width and degree it corresponds to
    For example the file "width=4_degree=3.json" corresponds to 4 qubits and degree 3

    Parameters
    ----------
    fileName : TYPE
        DESCRIPTION.

    Returns
    -------
    num_qubits : int
        circuit width
    degree : int
        graph degree.

    """
    pattern = 'width_([0-9]+)_instance_([0-9]+).json'
    match = re.search(pattern, fileName)
    #print(match)

    # assert match is not None, f"File {fileName} found inside folder. All files inside specified folder must be named in the format 'width_int_restartInd_int.json"
    num_qubits = int(match.groups()[0])
    degree = int(match.groups()[1])
    return (num_qubits,degree)


################################################
# PLOT METHODS

def plot_results_from_data(num_shots=100, radius = 0.75, max_iter=30, max_circuits = 1,
            method=2,
            score_metric='solution_quality', x_metric='cumulative_exec_time', y_metric='num_qubits', fixed_metrics={},
            num_x_bins=15, y_size=None, x_size=None, x_min=None, x_max=None,
            detailed_save_names=False, individual=False, **kwargs):
    """
    Plot results from the data contained in metrics tables.
    """

    # Add custom metric names to metrics module (in case this is run outside of run())
    add_custom_metric_names()
    
    # handle single string form of score metrics
    if type(score_metric) == str:
            score_metric = [score_metric]
    
    # for hydrogen lattice, objective function is always 'Energy'
    obj_str = "Energy"
 
    suffix = ''

    # If detailed names are desired for saving plots, put date of creation, etc.
    if detailed_save_names:
        cur_time=datetime.datetime.now()
        dt = cur_time.strftime("%Y-%m-%d_%H-%M-%S")
        suffix = f's{num_shots}_r{radius}_mi{max_iter}_{dt}'
    
    suptitle = f"Benchmark Results - Hydrogen Lattice ({method}) - Qiskit"
    backend_id = metrics.get_backend_id()
    options = {'shots' : num_shots, 'radius' : radius, 'restarts' : max_circuits}
    
    # plot all line metrics, including solution quality and accuracy ratio 
    # vs iteration count and cumulative execution time
    h_metrics.plot_all_line_metrics(suptitle,
                score_metrics=["energy", "solution_quality", "accuracy_ratio"],
                x_vals=["iteration_count", "cumulative_exec_time"],
                individual=individual,
                backend_id=backend_id,
                options=options)
    
    # plot all cumulative metrics, including average_iteration_time and accuracy ratio 
    # over number of qubits
    h_metrics.plot_all_cumulative_metrics(suptitle,
                score_metrics=["energy", "solution_quality", "accuracy_ratio"],
                x_vals=["iteration_count", "cumulative_exec_time"],
                individual=individual,
                backend_id=backend_id,
                options=options)
                
    # plot all area metrics
    metrics.plot_all_area_metrics(suptitle,
                score_metric=score_metric, x_metric=x_metric, y_metric=y_metric,
                fixed_metrics=fixed_metrics, num_x_bins=num_x_bins,
                x_size=x_size, y_size=y_size, x_min=x_min, x_max=x_max,
                options=options, suffix=suffix, which_metric='solution_quality')
    

################################################
################################################
# RUN METHOD

MAX_QUBITS = 16

def run (min_qubits=2, max_qubits=4, skip_qubits=2, max_circuits=3, num_shots=100,
        method=2, radius=None,
        thetas_array=None, parameterized=False, parameter_mode=1, do_fidelities=True,
        minimizer_function=None, minimizer_tolerance=1e-3, max_iter=30, comfort=False,
        score_metric=['solution_quality', 'accuracy_ratio'],
        x_metric=['cumulative_exec_time','cumulative_elapsed_time'], y_metric='num_qubits',
        fixed_metrics={}, num_x_bins=15, y_size=None, x_size=None, plot_results = True,
        save_res_to_file = False, save_final_counts = False, detailed_save_names = False, 
        backend_id='qasm_simulator', provider_backend=None,
        hub="ibm-q", group="open", project="main", exec_options=None, _instances=None) :
    """
    Parameters
    ----------
    min_qubits : int, optional
        The smallest circuit width for which benchmarking will be done The default is 3.
    max_qubits : int, optional
        The largest circuit width for which benchmarking will be done. The default is 6.
    max_circuits : int, optional
        Number of restarts. The default is None.
    num_shots : int, optional
        Number of times the circut will be measured, for each iteration. The default is 100.
    method : int, optional
        If 1, then do standard metrics, if 2, implement iterative algo metrics. The default is 1.
    thetas_array : list, optional
        list or ndarray of theta values. The default is None.
    N : int, optional
        For the max % counts metric, choose the highest N% counts. The default is 10.
    alpha : float, optional
        Value between 0 and 1. The default is 0.1.
    parameterized : bool, optional
        Whether to use parameter objects in circuits or not. The default is False.
    parameter_mode : bool, optional 
        If 1, use thetas_array of length 1, otherwise (num_qubits//2)**2, to match excitation pairs
    do_fidelities : bool, optional
        Compute circuit fidelity. The default is True.
    minimizer_function : function
        custom function used for minimizer
    minimizer_tolerance : float
        tolerance for minimizer, default is 1e-3,
    max_iter : int, optional
        Number of iterations for the minimizer routine. The default is 30.
    score_metric : list or string, optional
        Which metrics are to be plotted in area metrics plots. The default is 'fidelity'.
    x_metric : list or string, optional
        Horizontal axis for area plots. The default is 'cumulative_exec_time'.
    y_metric : list or string, optional
        Vertical axis for area plots. The default is 'num_qubits'.
    fixed_metrics : TYPE, optional
        DESCRIPTION. The default is {}.
    num_x_bins : int, optional
        DESCRIPTION. The default is 15.
    y_size : TYPint, optional
        DESCRIPTION. The default is None.
    x_size : string, optional
        DESCRIPTION. The default is None.
    backend_id : string, optional
        DESCRIPTION. The default is 'qasm_simulator'.
    provider_backend : string, optional
        DESCRIPTION. The default is None.
    hub : string, optional
        DESCRIPTION. The default is "ibm-q".
    group : string, optional
        DESCRIPTION. The default is "open".
    project : string, optional
        DESCRIPTION. The default is "main".
    exec_options : string, optional
        DESCRIPTION. The default is None.
    plot_results : bool, optional
        Plot results only if True. The default is True.
    save_res_to_file : bool, optional
        Save results to json files. The default is True.
    save_final_counts : bool, optional
        If True, also save the counts from the final iteration for each problem in the json files. The default is True.
    detailed_save_names : bool, optional
        If true, the data and plots will be saved with more detailed names. Default is False
    confort : bool, optional    
        If true, print comfort dots during execution
    """

    # Store all the input parameters into a dictionary.
    # This dictionary will later be stored in a json file
    # It will also be used for sending parameters to the plotting function
    dict_of_inputs = locals()
    
    thetas = []   #Need to change
    
    # Update the dictionary of inputs
    dict_of_inputs = {**dict_of_inputs, **{'thetas_array': thetas, 'max_circuits' : max_circuits}}
    
    # Delete some entries from the dictionary; they may contain secrets or function pointers
    for key in ["hub", "group", "project", "provider_backend", "exec_options", "minimizer_function"]:
        dict_of_inputs.pop(key)
    
    global hydrogen_lattice_inputs
    hydrogen_lattice_inputs = dict_of_inputs
    
    global QC_
    global circuits_done
    global minimizer_loop_index
    global opt_ts
    
    print("Hydrogen Lattice Benchmark Program - Qiskit")

    QC_ = None
    
    # validate parameters
    max_qubits = max(2, max_qubits)
    max_qubits = min(MAX_QUBITS, max_qubits)
    min_qubits = min(max(2, min_qubits), max_qubits)

    try:
        print("Validating user inputs...")
        # raise an exception if either min_qubits or max_qubits is not even
        if min_qubits % 2 != 0 or max_qubits % 2 != 0:
            raise ValueError("min_qubits and max_qubits must be even. min_qubits = {}, max_qubits = {}".format(min_qubits, max_qubits))
    except ValueError as err:
        # display error message and stop execution if min_qubits or max_qubits is not even
        logger.error(err)
        if min_qubits % 2 != 0:
            min_qubits += 1
        if max_qubits % 2 != 0:
            max_qubits -= 1
            max_qubits = min(max_qubits, MAX_QUBITS)
        print(err.args[0] + "\n Running for for values min_qubits = {}, max_qubits = {}".format(min_qubits, max_qubits))
    
    # don't compute exectation unless fidelity is is needed
    global do_compute_expectation
    do_compute_expectation = do_fidelities
    
    # save the desired parameter mode globally (for now, during dev)
    global saved_parameter_mode
    saved_parameter_mode = parameter_mode
    
    # given that this benchmark does every other width, set y_size default to 1.5
    if y_size == None:
        y_size = 1.5
        
    # Initialize metrics module with empty metrics arrays
    metrics.init_metrics()

    # Add custom metric names to metrics module
    add_custom_metric_names()
    
    # Define custom result handler
    def execution_handler (qc, result, num_qubits, s_int, num_shots):  
     
        # determine fidelity of result set
        num_qubits = int(num_qubits)
        counts, fidelity = analyze_and_print_result(qc, result, num_qubits, int(s_int), num_shots)
        metrics.store_metric(num_qubits, s_int, 'solution_quality', fidelity)

    def execution_handler2 (qc, result, num_qubits, s_int, num_shots):
    
        # Stores the results to the global saved_result variable
        global saved_result
        saved_result = result
        
    # Initialize execution module using the execution result handler above and specified backend_id
    # for method=2 we need to set max_jobs_active to 1, so each circuit completes before continuing
    if method == 2:
        ex.max_jobs_active = 1
        ex.init_execution(execution_handler2)
    else:
        ex.init_execution(execution_handler)
    
    ex.set_execution_target(backend_id, provider_backend=provider_backend,
            hub=hub, group=group, project=project, exec_options=exec_options)

    # create a data folder for the results
    create_data_folder(save_res_to_file, detailed_save_names, backend_id)
    
# -------------classical Pauli sum operator from list of Pauli operators and coefficients----
    # Below function is to reduce some dependency on qiskit ( String data type issue)-------------
    # def pauli_sum_op(ops, coefs):
    #     if len(ops) != len(coefs):
    #         raise ValueError("The number of Pauli operators and coefficients must be equal.")
    #     pauli_sum_op_list = [(op, coef) for op, coef in zip(ops, coefs)]
    #     return pauli_sum_op_list
#-------------classical Pauli sum operator from list of Pauli operators and coefficients-----------------

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    # DEVNOTE: increment by 2 for paired electron circuits
    global instance_filepath 

    # read precomputed energies file from the random energies path

    try:
        random_energies_dict = common.get_random_energies_dict()
    except ValueError as err:
        logger.error(err)
        print("Error reading precomputed random energies json file. Please create the file by running the script 'compute_random_energies.py' in the _common/random_sampler directory")
        raise

    for num_qubits in range(min_qubits, max_qubits + 1, 2):
        
        if method == 1:
            print(f"************\nExecuting [{max_circuits}] circuits for num_qubits = {num_qubits}")
        else:
            print(f"************\nExecuting [{max_circuits}] restarts for num_qubits = {num_qubits}")
        
        # # If radius is negative, 
        # if radius < 0 :
        #     radius = max(3, (num_qubits + radius))
            
        # looping all instance files according to max_circuits given        
        for instance_num in range(1, max_circuits + 1):
            # Index should start from 1
            
            # if radius is given we should do same radius for max_circuits times
            if radius is not None:
                try:
                    instance_filepath = common.get_instance_filepaths(num_qubits, radius)
                except ValueError as err:
                    logger.error(err)

                    print(f"  ... problem not found. Running default radius")
                    instance_filepath = common.get_instance_filepaths(num_qubits)[0]

            # if radius is not given we should do all the radius for max_circuits times
            else:
                instance_filepath_list = common.get_instance_filepaths(num_qubits)
                try:
                    if len(instance_filepath_list) >= instance_num :
                        instance_filepath = instance_filepath_list[instance_num-1]
                    else:
                        raise ValueError("problem not found, please check the instances folder and generate the required problem and solution instances using provided scripts.")
                except ValueError as err:
                    logger.error(err)
                    raise
            
            # get the filename from the instance_filepath and get the random energy from the dictionary
            filename = os.path.basename(instance_filepath)
            filename = filename.split('.')[0]
            random_energy = random_energies_dict[filename]

            # operator is paired hamiltonian  for each instance in the loop  
            ops,coefs = common.read_paired_instance(instance_filepath)
            operator = PauliSumOp.from_list(list(zip(ops, coefs)))
            
            # solution has list of classical solutions for each instance in the loop
            sol_file_name = instance_filepath[:-5] + ".sol"
            method_names, values = common.read_puccd_solution(sol_file_name)
            solution = list(zip(method_names, values))   
            
            # if the file does not exist, we are done with this number of qubits
            if operator == None:
                print(f"  ... problem not found.")
                break
            
            # create an intial thetas_array, given the circuit width and user input
            thetas_array_0 = get_initial_parameters(num_qubits, thetas_array)
            
            ###############
            if method == 1:
            
                # create the circuit(s) for given qubit size and secret string, store time metric
                ts = time.time()
                
                # create the circuits to be tested
                qc_array, frmt_obs, params = HydrogenLattice(num_qubits=num_qubits,
                            secret_int=instance_num,
                            operator=operator,
                            thetas_array=thetas_array_0,
                            parameterized=parameterized)
                '''
                # loop over the array of circuits generated
                for qc in qc_array:
                '''
                # Instead, we only do one of the circuits, the last one. which is in the 
                # z-basis.  This is the one that is most illustrative of a device's fidelity
                qc = qc_array[2]
                
                ''' TMI ...
                # for testing and debugging ...
                #if using parameter objects, bind before printing
                if verbose:
                    print(qc.bind_parameters(params) if parameterized else qc)
                ''' 
                
                metrics.store_metric(num_qubits, instance_num, 'create_time', time.time()-ts)
                
                # pre-compute and save an array of expected measurements
                # for comparison in the analysis method
                # (do not count as part of creation time, as this is a benchmark function)
                if do_compute_expectation:
                    logger.info('Computing expectation')
                    
                    # pass parameters as they are used during execution
                    compute_expectation(qc, num_qubits, instance_num, params=params)
            
                # submit circuit for execution on target, with parameters
                ex.submit_circuit(qc, num_qubits, instance_num, shots=num_shots, params=params)
                
                ''' (see comment above about looping)
                    # Break out of this loop, so we only execute the first of the ansatz circuits
                    # DEVNOTE: maybe we should do all three, and aggregate, just as in method2
                    #break
                ''' 

            ###############
            if method == 2:
            
                logger.info(f'===============  Begin method 2 loop, enabling transpile')
                
                # a unique circuit index used inside the inner minimizer loop as identifier
                # Value of 0 corresponds to the 0th iteration of the minimizer
                minimizer_loop_index = 0 
                
                # Always start by enabling transpile ...
                ex.set_tranpilation_flags(do_transpile_metrics=True, do_transpile_for_execute=True)
                    
                # define the radius and energy variables
                doci_energy = float(next(value for key, value in solution if key == 'doci_energy'))
                fci_energy = float(next(value for key, value in solution if key == 'fci_energy'))
                
                current_radius = float(os.path.basename(instance_filepath).split('_')[2])
                current_radius += float(os.path.basename(instance_filepath).split('_')[3][:2])*.01

                cumlative_iter_time = [0]
                start_iters_t = time.time()
                
                #########################################
                #### Objective function to compute energy
                
                def objective_function(thetas_array):
                    '''
                    Objective function that calculates the expected energy for the given parameterized circuit

                    Parameters
                    ----------
                    thetas_array : list
                        list of theta values.
                    '''
                    
                    # Every circuit needs a unique id; add unique_circuit_index instead of s_int
                    global minimizer_loop_index
                    unique_id = instance_num * 1000 + minimizer_loop_index
                    
                    # variables used to aggregate metrics for all terms
                    result_array = []
                    quantum_execution_time = 0.0
                    quantum_elapsed_time = 0.0

                    # create ansatz from the operator, in multiple circuits, one for each measured basis
                    # call the HydrogenLattice ansatz to generate a parameterized hamiltonian
                    ts = time.time()
                    qc_array, frmt_obs, params = HydrogenLattice(
                            num_qubits=num_qubits,
                            secret_int=unique_id,
                            thetas_array=thetas_array,
                            parameterized=parameterized,
                            operator=operator)
                    
                    # store the time it took to create the circuit
                    metrics.store_metric(num_qubits, unique_id, 'create_time', time.time()-ts)
                    
                    #####################
                    # loop over each of the circuits that are generated with basis measurements and execute
                    
                    if verbose:
                        print(f"... ** compute energy for num_qubits={num_qubits}, circuit={unique_id}, parameters={params}, thetas_array={thetas_array}")
                    
                    # loop over each circuit and execute
                    for qc in qc_array:      

                        # bind parameters to circuit before execution
                        if parameterized:
                            qc.bind_parameters(params)

                        # submit circuit for execution on target with the current parameters
                        ex.submit_circuit(qc, num_qubits, unique_id, shots=num_shots, params=params)
    
                        # wait for circuit to complete by calling finalize  ...
                        # finalize execution of group (each circuit in loop accumulates metrics)
                        ex.finalize_execution(None, report_end=False)
                        
                        # after first execution and thereafter, no need for transpilation if parameterized
                        if parameterized:
                        
                            # DEVNOTE: since Hydro uses 3 circuits inside this loop, and execute.py can only
                            # cache 1 at a time, we cannot yet implement caching.  Transpile every time for now.
                            cached_circuits = False
                            if cached_circuits:
                                ex.set_tranpilation_flags(do_transpile_metrics=False,
                                                    do_transpile_for_execute=False)
                                logger.info(f'**** First execution complete, disabling transpile')
                             
                        # result array stores the multiple results we measure along different Pauli basis.
                        global saved_result
                        result_array.append(saved_result)

                        # Aggregate execution and elapsed time for running all three circuits 
                        # corresponding to different measurements along the different Pauli bases
                        quantum_execution_time = quantum_execution_time + \
                                metrics.circuit_metrics[str(num_qubits)][str(unique_id)]['exec_time']
                        quantum_elapsed_time = quantum_elapsed_time + \
                                metrics.circuit_metrics[str(num_qubits)][str(unique_id)]['elapsed_time']

                    #####################
                    # classical processing of results 
                        
                    global opt_ts
                    global cumulative_iter_time

                    cumlative_iter_time.append(cumlative_iter_time[-1] + quantum_execution_time)

                    # store the new exec time and elapsed time back to metrics
                    metrics.store_metric(str(num_qubits), str(unique_id), 'exec_time', quantum_execution_time)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'elapsed_time', quantum_elapsed_time)

                    # increment the minimizer loop index, the index is increased by one
                    # for the group of three circuits created ( three measurement basis circuits)
                    minimizer_loop_index += 1
                    
                    # print "comfort dots" (newline before the first iteration)
                    if comfort:
                        if minimizer_loop_index == 1: print("")
                        print(".", end ="")
                        if verbose: print("")

                    # Start counting classical optimizer time here again
                    tc1 = time.time()
                    
                    # compute energy for this combination of observables and measurements
                    global energy
                    energy = compute_energy(result_array = result_array,
                            formatted_observables = frmt_obs, num_qubits=num_qubits)

                    # append the most recent energy value to the list
                    lowest_energy_values.append(energy)
    
                    # calculate the solution quality, accuracy volume and accuracy ratio
                    global solution_quality, accuracy_volume, accuracy_ratio
                    solution_quality, accuracy_volume, accuracy_ratio = calculate_quality_metric(
                            energy=energy,
                            fci_energy=fci_energy,
                            random_energy=random_energy, 
                            precision=0.5,
                            num_electrons=num_qubits)

                    # store the metrics for the current iteration
                    metrics.store_metric(str(num_qubits), str(unique_id), 'energy', energy)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'fci_energy', fci_energy)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'solution_quality', solution_quality)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'accuracy_volume', accuracy_volume)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'accuracy_ratio', accuracy_ratio)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'fci_energy', fci_energy)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'doci_energy', doci_energy)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'radius', current_radius)
                    metrics.store_metric(str(num_qubits), str(unique_id), 'iteration_count', minimizer_loop_index)
                    
                    return energy
                
                # callback for each iteration (currently unused)
                def callback_thetas_array(thetas_array):
                    pass
                     
                ###########################
                # End of Objective Function
                
                # if in verbose mode, comfort dots need a newline before optimizer gets going
                #if comfort and verbose:
                    #print("")

                # Initialize an empty list to store the energy values from each iteration
                lowest_energy_values = []

                # execute COPYLA classical optimizer to minimize the objective function
                # objective function is called repeatedly with varying parameters
                # until the lowest energy found
                if minimizer_function == None:
                    ret = minimize(objective_function,
                            x0=thetas_array_0.ravel(),  # note: revel not really needed for this ansatz
                            method='COBYLA',
                            tol=minimizer_tolerance,
                            options={'maxiter': max_iter, 'disp': False},
                            callback=callback_thetas_array) 
                            
                # or, execute a custom minimizer
                else:
                    ret = minimizer_function(objective_function=objective_function,
                            initial_parameters=thetas_array_0.ravel(),  # note: revel not really needed for this ansatz
                            callback=callback_thetas_array
                            )
                            
                #if verbose:
                #print(f"\nEnergies for problem file {os.path.basename(instance_filepath)} for {num_qubits} qubits and radius {current_radius} of paired hamiltionians")
                #print(f"  PUCCD calculated energy : {ideal_energy}")
            
                if comfort: print("")
                print(f"Classically Computed Energies from solution file {os.path.basename(sol_file_name)} for {num_qubits} qubits and radius {current_radius}")
                print(f"  DOCI calculated energy : {doci_energy}")
                print(f"  FCI calculated energy : {fci_energy}")
                print(f"  Random Solution calculated energy : {random_energy}")
                
                print(f"Computed Energies for {num_qubits} qubits and radius {current_radius}")
                print(f"  Lowest Energy : {lowest_energy_values[-1]}")
                print(f"  Solution Quality : {solution_quality}, Accuracy Ratio : {accuracy_ratio}")
                
                # pLotting each instance of qubit count given 
                cumlative_iter_time = cumlative_iter_time[1:]                

                # save the data for this qubit width, and instance number
                store_final_iter_to_metrics_json(backend_id=backend_id,
                                num_qubits=num_qubits, 
                                radius = radius,
                                instance_num=instance_num,
                                num_shots=num_shots, 
                                converged_thetas_list=ret.x.tolist(),
                                energy = lowest_energy_values[-1],
                                # iter_size_dist=iter_size_dist, iter_dist=iter_dist,
                                detailed_save_names=detailed_save_names,
                                dict_of_inputs=dict_of_inputs, save_final_counts=save_final_counts,
                                save_res_to_file=save_res_to_file, _instances=_instances)

            ###### End of instance processing

        # for method 2, need to aggregate the detail metrics appropriately for each group
        # Note that this assumes that all iterations of the circuit have completed by this point
        if method == 2:                  
            metrics.process_circuit_metrics_2_level(num_qubits)
            metrics.finalize_group(str(num_qubits))
            
    # Wait for some active circuits to complete; report metrics when groups complete
    ex.throttle_execution(metrics.finalize_group)
        
    # Wait for all active circuits to complete; report metrics when groups complete
    ex.finalize_execution(metrics.finalize_group)       
    
    # print a sample circuit
    if print_sample_circuit:
        if method == 1: 
            print("Sample Circuit:"); print(QC_ if QC_ != None else "  ... too large!")

    # Plot metrics for all circuit sizes
    if method == 1:
        metrics.plot_metrics(f"Benchmark Results - Hydrogen Lattice ({method}) - Qiskit",
                options=dict(shots=num_shots))
    elif method == 2:
        if plot_results:
            plot_results_from_data(**dict_of_inputs)

#################################
# MAIN

# # if main, execute method
if __name__ == '__main__': run()

# # %%

# run()
