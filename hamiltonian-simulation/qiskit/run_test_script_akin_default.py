import hamiltonian_simulation_benchmark as ham 
import json
import sys
import os
import logging
import numpy as np
from matplotlib.pyplot import figure
from qiskit import transpile
from qiskit_aer import Aer

sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit", "../../_common/transformers"]
import execute as ex
import metrics as metrics
import tket_optimiser as tket_optimiser

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import TimeEvolutionProblem, SciPyRealEvolver
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer.noise import NoiseModel, depolarizing_error

# Below code is copy/pasted from Jupyter Notebook
# The functions in this cell are used to exactly simulate the hamiltonian evolution using a classical matrix evolution. 

def initial_state(n_spins: int, initial_state: str = "checker") -> QuantumCircuit:
    """
    Initialize the quantum state.

    Dev note: This function is copy/pasted from HamiltonianSimulation.
    
    Args:
        n_spins (int): Number of spins (qubits).
        initial_state (str): The chosen initial state. By default applies the checkerboard state, but can also be set to "ghz", the GHZ state.

    Returns:
        QuantumCircuit: The initialized quantum circuit.
    """
    qc = QuantumCircuit(n_spins)

    if initial_state.strip().lower() == "checkerboard" or initial_state.strip().lower() == "neele":
        # Checkerboard state, or "Neele" state
        for k in range(0, n_spins, 2):
            qc.x([k])
    elif initial_state.strip().lower() == "ghz":
        # GHZ state: 1/sqrt(2) (|00...> + |11...>)
        qc.h(0)
        for k in range(1, n_spins):
            qc.cx(k-1, k)

    return qc

def construct_TFIM_hamiltonian(n_spins: int) -> SparsePauliOp:
    """
    Construct the Transverse Field Ising Model (TFIM) Hamiltonian.

    Args:
        n_spins (int): Number of spins (qubits).

    Returns:
        SparsePauliOp: The Hamiltonian represented as a sparse Pauli operator.
    """
    pauli_strings = []
    coefficients = []
    g = 1  # Strength of the transverse field

    # Pauli spin vector product terms
    for i in range(n_spins):
        x_term = 'I' * i + 'X' + 'I' * (n_spins - i - 1)
        pauli_strings.append(x_term)
        coefficients.append(g)

    identity_string = ['I'] * n_spins

    # ZZ operation on each pair of qubits in a linear chain
    for j in range(2):
        for i in range(j % 2, n_spins - 1, 2):
            zz_term = identity_string.copy()
            zz_term[i] = 'Z'
            zz_term[(i + 1) % n_spins] = 'Z'
            zz_term = ''.join(zz_term)
            pauli_strings.append(zz_term)
            coefficients.append(1.0)

    return SparsePauliOp.from_list(zip(pauli_strings, coefficients))

def construct_heisenberg_hamiltonian(n_spins: int, w: int, hx: list[float], hz: list[float]) -> SparsePauliOp:
    """
    Construct the Heisenberg Hamiltonian with disorder.

    Args:
        n_spins (int): Number of spins (qubits).
        w (float): Strength of two-qubit interactions for heisenberg hamiltonian. 
        hx (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 
        hz (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 

    Returns:
        SparsePauliOp: The Hamiltonian represented as a sparse Pauli operator.
    """

    pauli_strings = []
    coefficients = []

    # Disorder terms
    for i in range(n_spins):
        x_term = 'I' * i + 'X' + 'I' * (n_spins - i - 1)
        z_term = 'I' * i + 'Z' + 'I' * (n_spins - i - 1)
        pauli_strings.append(x_term)
        coefficients.append(w * hx[i])
        pauli_strings.append(z_term)
        coefficients.append(w * hz[i])

    identity_string = ['I'] * n_spins

    # Interaction terms
    for j in range(2):
        for i in range(j % 2, n_spins - 1, 2):
            xx_term = identity_string.copy()
            yy_term = identity_string.copy()
            zz_term = identity_string.copy()

            xx_term[i] = 'X'
            xx_term[(i + 1) % n_spins] = 'X'

            yy_term[i] = 'Y'
            yy_term[(i + 1) % n_spins] = 'Y'

            zz_term[i] = 'Z'
            zz_term[(i + 1) % n_spins] = 'Z'

            pauli_strings.append(''.join(xx_term))
            coefficients.append(1.0)
            pauli_strings.append(''.join(yy_term))
            coefficients.append(1.0)
            pauli_strings.append(''.join(zz_term))
            coefficients.append(1.0)

    return SparsePauliOp.from_list(zip(pauli_strings, coefficients))

def construct_hamiltonian(n_spins: int, hamiltonian: str, w: float, hx : list[float], hz: list[float]) -> SparsePauliOp:
    """
    Construct the Hamiltonian based on the specified method.

    Args:
        n_spins (int): Number of spins (qubits).
        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM". 
        w (float): Strength of two-qubit interactions for heisenberg hamiltonian. 
        hx (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 
        hz (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 

    Returns:
        SparsePauliOp: The constructed Hamiltonian.
    """

    hamiltonian = hamiltonian.strip().lower()

    if hamiltonian == "heisenberg":
        return construct_heisenberg_hamiltonian(n_spins, w, hx, hz)
    elif hamiltonian == "tfim":
        return construct_TFIM_hamiltonian(n_spins)
    else:
        raise ValueError("Invalid Hamiltonian specification.")

def HamiltonianSimulationExact(n_spins: int, t: float, init_state: str, hamiltonian: str, w: float, hx: list[float], hz: list[float]) -> dict:
    """
    Perform exact Hamiltonian simulation using classical matrix evolution.

    Args:
        n_spins (int): Number of spins (qubits).
        t (float): Duration of simulation.
        init_state (str): The chosen initial state. By default applies the checkerboard state, but can also be set to "ghz", the GHZ state.
        hamiltonian (str): Which hamiltonian to run. "heisenberg" by default but can also choose "TFIM". 
        w (float): Strength of two-qubit interactions for heisenberg hamiltonian. 
        hx (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 
        hz (list[float]): Strength of internal disorder parameter for heisenberg hamiltonian. 

    Returns:
        dict: The distribution of the evolved state.
    """
    hamiltonian_sparse = construct_hamiltonian(n_spins, hamiltonian, w, hx, hz)
    time_problem = TimeEvolutionProblem(hamiltonian_sparse, t, initial_state=initial_state(n_spins, init_state))
    result = SciPyRealEvolver(num_timesteps=1).evolve(time_problem)
    return result.evolved_state.probabilities_dict()

def create_noise_model(fidelity):

    if fidelity == 1:
        return None
    else: 
        p = 15/4 * (1 - fidelity)
        noise_model = NoiseModel()
        depolarizing_err = depolarizing_error(p, 2)  # 2-qubit depolarizing error
        noise_model.add_all_qubit_quantum_error(depolarizing_err, ['cx'])  # Apply to CNOT gates
        return noise_model 

from PIL import Image, ImageDraw, ImageFont

def save_and_combine_images(images, output_filename, method, fidelities):
    """
    Given images that correspond to multiple configurations of using the pytket optimizer or not,
    this function combines them into a grid with appropriate titles and labels.
    """

    num_fidelities = len(fidelities)
    
    if len(images) != 2 * num_fidelities:
        raise ValueError(f"Input images list isn't of length {2 * num_fidelities}, it is instead length {len(images)}")

    # Determine the maximum width and height of the input images
    max_width = max(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)

    # Padding for labels
    label_padding = 50
    title_padding = 100

    # Create a new image to accommodate the grid with extra space for text
    combined_image = Image.new(
        "RGB", 
        (2 * max_width + label_padding, num_fidelities * max_height + title_padding + label_padding), 
        "white"
    )  # Adjusted space for labels and title

    # Create drawing object
    draw = ImageDraw.Draw(combined_image)

    # Use a larger font size; download a .ttf file or use available system fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", size=36)  # Larger font for title
        label_font = ImageFont.truetype("arial.ttf", size=28)  # Larger font for labels
    except IOError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    # Set text for title and labels
    title = f"Hamiltonian Simulation Method {method}"
    x_labels = ["No compilation", "Pytket gate compilation"]

    # Calculate center for the title and add it
    title_width = draw.textlength(title, font=title_font)
    title_height = title_font.size
    draw.text(((2 * max_width + label_padding - title_width) / 2, 10), title, fill="black", font=title_font)

    # Paste the images and add axis labels
    for index, image in enumerate(images):
        x = (index % 2) * max_width + label_padding
        y = (index // 2) * max_height + title_padding  # Adjust for title and label space

        # Center the image in the cell
        img_x = x + (max_width - image.size[0]) // 2
        img_y = y + (max_height - image.size[1]) // 2
        combined_image.paste(image, (img_x, img_y))

    # Center x-axis labels below each column of images
    for i in range(2):
        x_label = x_labels[i]
        label_width = draw.textlength(x_label, font=label_font)
        label_height = label_font.size
        draw.text(
            (i * max_width + (max_width - label_width) / 2 + label_padding, 
             num_fidelities * max_height + title_padding + 10), 
            x_label, fill="black", font=label_font
        )

    # Y-axis fidelity labels
    for i, fidelity in enumerate(fidelities):
        y_label = f"Fidelity {fidelity}"
        y_label_width = draw.textlength(y_label, font=label_font)
        y_label_height = label_font.size
        y_label_x = 10
        y_label_y = i * max_height + title_padding + (max_height - y_label_height) / 2  # Centered in each row of images
        draw.text((y_label_x, y_label_y), y_label, fill="black", font=label_font)

    # Save the new image
    combined_image.save(output_filename)



def set_precalculated_data(w, k, t, min_qubits, max_qubits):

    """
    This code is copy/pasted from the Jupyter Notebook precalculated_Data.ipnyb.
    """

    backend = Aer.get_backend("qasm_simulator")

    precalculated_data = {}

# store parameters in precalculated data
    precalculated_data["w"] = w
    precalculated_data["k"] = k
    precalculated_data["t"] = t

# add parameter random values to precalculated data to ensure consistency
    np.random.seed(26)
    precalculated_data['hx'] = list(2 * np.random.random(20) - 1) # random numbers between [-1, 1]
    np.random.seed(75)
    precalculated_data['hz'] = list(2 * np.random.random(20) - 1) # random numbers between [-1, 1]

    num_shots = 100000

    for n_spins in range(min_qubits, max_qubits+1):

            print(f"Now running n_spins {n_spins}")

            hx = precalculated_data['hx'][:n_spins]
            hz = precalculated_data['hz'][:n_spins]

            qc = ham.HamiltonianSimulation(n_spins, k, t, w=w, hx = hx, hz = hz)


            transpiled_qc = transpile(qc, backend, optimization_level=0)
            job = backend.run(transpiled_qc, shots=num_shots)
            result = job.result()
            counts = result.get_counts(qc)

            dist = {}
            for key in counts.keys():
                prob = counts[key] / num_shots
                dist[key] = prob

            # add dist values to precalculated data for use in fidelity calculation
            precalculated_data[f"Qubits{n_spins}"] = dist  

    ham.precalculated_data = precalculated_data

if __name__ == "__main__":

    """
    The purpose of this script is to go through several ranges of k or t, which would usually require you to edit the precalculated data jupyter notebook. 

    Choose max_qubits that are relatively small, since an expensive calculate to recalculate the "precalculated" distribution will be done for all the different settings in k_range and time_range. 
    """

    min_qubits=2
    max_qubits=12
    skip_qubits=1
    max_circuits=10
    num_shots=1000


    backend_id="qasm_simulator"

    hub="ibm-q"; group="open"; project="main"
    provider_backend = None
    exec_options = {}
    #set to true when the executor fails to launch
    ex.verbose = False
    # metrics.show_plot_images = False

    # selected trotter steps to go through, can be any length 
    k_range = [5]

    # selected times to go through, can be of any length 
    # a special note: do not choose t=1 when k=3.. that happens to produce gates with 0 rotation that are compiled out!
    time_range = [.2]

    # methods to go through, can be list of length 1 or 2 

    w=1

    # 2Q fidelity with depolarization model, should be length 2 for script to work. default, from Charlie Baldwin's graph, is .95 and .995. 
    f_range = [1]

    for k in k_range:
        for t in time_range:

            # overwrite the precalculated_data inside of hamiltonian_simulation_benchmark for the settings desired 
            # does this in a global manner by editing the hamiltonian_simulation_benchmark module
            set_precalculated_data(w=w, k=k, t=t, min_qubits = min_qubits, max_qubits = max_qubits)


            for f in f_range: 

                exec_options ={}
                ham.run(min_qubits=min_qubits, max_qubits=max_qubits, skip_qubits=skip_qubits,
        max_circuits=max_circuits, num_shots=num_shots,
        backend_id=backend_id, provider_backend=provider_backend,
        hub=hub, group=group, project=project, exec_options=exec_options)


    #uncomment if you wish to graph with the other benchmarks
    # sys.path.insert(1, "../../bernstein-vazirani/qiskit")
    # import bv_benchmark
    # bv_benchmark.run(min_qubits=min_qubits, max_qubits=max_qubits, skip_qubits=skip_qubits,
    #             max_circuits=max_circuits, num_shots=num_shots,
    #             method=1,
    #             backend_id=backend_id, provider_backend=provider_backend,
    #             hub=hub, group=group, project=project, exec_options=exec_options)
    #
    # import sys
    # sys.path.insert(1, "../../deutsch-jozsa/qiskit")
    # import dj_benchmark
    # dj_benchmark.run(min_qubits=min_qubits, max_qubits=max_qubits, skip_qubits=skip_qubits,
    #                 max_circuits=max_circuits, num_shots=num_shots,
    #                 backend_id=backend_id, provider_backend=provider_backend,
    #                 hub=hub, group=group, project=project, exec_options=exec_options)
    # import sys
    # sys.path.insert(1, "../../hidden-shift/qiskit")
    # import hs_benchmark
    # hs_benchmark.run(min_qubits=min_qubits, max_qubits=max_qubits,
    #             max_circuits=max_circuits, num_shots=num_shots,
    #             backend_id=backend_id, provider_backend=provider_backend,
    #             hub=hub, group=group, project=project, exec_options=exec_options)
    #
    #
    # import sys
    # sys.path.insert(1, "../../quantum-fourier-transform/qiskit")
    # import qft_benchmark
    # qft_benchmark.run(min_qubits=min_qubits, max_qubits=max_qubits, skip_qubits=skip_qubits,
    #             max_circuits=max_circuits, num_shots=num_shots,
    #             method=1,
    #             backend_id=backend_id, provider_backend=provider_backend,
    #             hub=hub, group=group, project=project, exec_options=exec_options)
    #
    # import sys
    # sys.path.insert(1, "../../quantum-fourier-transform/qiskit")
    # import qft_benchmark
    # qft_benchmark.run(min_qubits=min_qubits, max_qubits=max_qubits, skip_qubits=skip_qubits,
    #             max_circuits=max_circuits, num_shots=num_shots,
    #             method=2,
    #             backend_id=backend_id, provider_backend=provider_backend,
    #             hub=hub, group=group, project=project, exec_options=exec_options)
    #
    #
    # import sys
    # sys.path.insert(1, "../../phase-estimation/qiskit")
    # import pe_benchmark
    # pe_benchmark.run(min_qubits=min_qubits, max_qubits=max_qubits, skip_qubits=skip_qubits,
    #                 max_circuits=max_circuits, num_shots=num_shots,
    #                 backend_id=backend_id, provider_backend=provider_backend,
    #                 hub=hub, group=group, project=project, exec_options=exec_options)
    #
    # import sys
    # sys.path.insert(1, "../../grovers/qiskit")
    # import grovers_benchmark
    # grovers_benchmark.run(min_qubits=min_qubits, max_qubits=max_qubits, skip_qubits=skip_qubits,
    #             max_circuits=max_circuits, num_shots=num_shots,
    #             backend_id=backend_id, provider_backend=provider_backend,
    #             hub=hub, group=group, project=project, exec_options=exec_options)
    #
    # metrics.show_plot_images = True
    #
    # metrics.plot_all_app_metrics("qasm_simulator", do_all_plots=False, include_apps=None)


