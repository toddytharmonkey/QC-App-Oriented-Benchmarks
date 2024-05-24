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



from qiskit_aer.noise import NoiseModel, depolarizing_error

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

    # Create a new image to accommodate the grid with extra space for text
    combined_image = Image.new(
        "RGB", (2 * max_width, num_fidelities * max_height + 150), "white"
    )  # Adjusted space for labels and title

    # Create drawing object
    draw = ImageDraw.Draw(combined_image)

    # Use a larger font size; download a .ttf file or use available system fonts
    try:
        font = ImageFont.truetype("arial.ttf", size=24)  # Specify path to a TTF font file and size
    except IOError:
        font = ImageFont.load_default()

    # Set text for title and labels
    title = f"Hamiltonian Simulation Method {method}"
    y_label = "2Q gate fidelity"
    x_labels = ["No compilation", "Pytket gate compilation"]

    # Calculate center for the title and add it
    title_width = draw.textlength(title, font=font)
    title_height = font.size
    draw.text(((2 * max_width - title_width) / 2, 10), title, fill="black", font=font)

    # Paste the images and add axis labels
    for index, image in enumerate(images):
        x = (index % 2) * max_width
        y = (index // 2) * max_height + 50  # Adjust for title space

        # Center the image in the cell
        img_x = x + (max_width - image.size[0]) // 2
        img_y = y + (max_height - image.size[1]) // 2
        combined_image.paste(image, (img_x, img_y))

    # Center x-axis labels below each column of images
    for i in range(2):
        x_label = x_labels[i]
        label_width = draw.textlength(x_label, font=font)
        label_height = font.size
        draw.text((i * max_width + (max_width - label_width) / 2, num_fidelities * max_height + 60), x_label, fill="black", font=font)

    # Y-axis fidelity labels
    for i, fidelity in enumerate(fidelities):
        y_label = f"Fidelity {fidelity}"
        y_label_width = draw.textlength(y_label, font=font)
        y_label_height = font.size
        y_label_x = 10
        y_label_y = i * max_height + 50 + (max_height - y_label_height) / 2  # Centered in each row of images
        draw.text((y_label_x, y_label_y), y_label, fill="black", font=font)

    # Save the new image
    combined_image.save(output_filename)

from qiskit.visualization import plot_distribution
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
    precalculated_data["h_x"] = list(
        np.ones(20)
    )  # random numbers between [-1, 1]
    np.random.seed(75)
    precalculated_data["h_z"] = list(
        np.ones(20)
    )  # random numbers between [-1, 1]

    num_shots = 100000

    for n_spins in range(min_qubits, max_qubits+1):

        print(f"Now running n_spins {n_spins}")

        qc = ham.HamiltonianSimulation(n_spins, k, t, method=1)

        dist2 = ham.Hamiltonian_Simulation_Exact(n_spins, t, method=1)

        qc3 = ham.HamiltonianSimulation(n_spins, k, t, method=2)

        transpiled_qc = transpile(qc, backend, optimization_level=0)
        job = backend.run(transpiled_qc, shots=num_shots)
        result = job.result()

        counts = result.get_counts(qc)

        transpiled_qc3 = transpile(qc3, backend, optimization_level=0)
        job3 = backend.run(transpiled_qc3, shots=num_shots)
        result3 = job3.result()
        counts3 = result3.get_counts()

        dist = {}
        for key in counts.keys():
            prob = counts[key] / num_shots
            dist[key] = prob

            #     dist2 = {}
            #     for key in counts2.keys():
            #         prob = counts2[key] / num_shots
            #         dist2[key] = prob

            dist3 = {}
            for key in counts3.keys():
                prob = counts3[key] / num_shots
                dist3[key] = prob

                # add dist values to precalculated data for use in fidelity calculation
                precalculated_data[f"Qubits - {n_spins}"] = dist

                precalculated_data[f"Qubits2 - {n_spins}"] = dist2

                precalculated_data[f"Qubits3 - {n_spins}"] = dist3

        ham.precalculated_data = precalculated_data

import matplotlib.pyplot as plt 

if __name__ == "__main__":

    for t in np.linspace(0,1,100):
        
        print(t)

        set_precalculated_data(1,1,1,6,6)

        exact = ham.Hamiltonian_Simulation_Exact(6, t, method=1)

        plot_distribution(exact, sort='asc', filename= f"exact/{t}".replace(".",""))
      
