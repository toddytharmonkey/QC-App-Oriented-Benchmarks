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

def save_and_combine_images(images, output_filename, method):
    """
    Given four images that correspond to the four possible configurations of using the pytket optimiser or not,
    this function combines them into a 2x2 grid with appropriate titles and labels.
    """

    if len(images) != 4:
        raise ValueError(f"Input images list isn't of length 4, it is instead length {len(images)}")

    # Determine the maximum width and height of the input images
    max_width = max(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)

    # Create a new image to accommodate 2x2 grid with extra space for text
    combined_image = Image.new("RGB", (2 * max_width, 2 * max_height + 150), "white")  # Adjusted space for labels and title

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

    # Paste the images and add axis labels
    for index, image in enumerate(images):
        x = (index % 2) * max_width
        y = (index // 2) * max_height + 50  # Adjust for title space

        # Center the image in the cell
        img_x = x + (max_width - image.size[0]) // 2
        img_y = y + (max_height - image.size[1]) // 2
        combined_image.paste(image, (img_x, img_y))



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
    precalculated_data["h_x"] = list(
        2 * np.random.random(20) - 1
    )  # random numbers between [-1, 1]
    np.random.seed(75)
    precalculated_data["h_z"] = list(
        2 * np.random.random(20) - 1
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


if __name__ == "__main__":

    """
    The purpose of this script is to go through several ranges of k or t, which would usually require you to edit the precalculated data jupyter notebook. 

    Choose max_qubits that are relatively small, since an expensive calculate to recalculate the "precalculated" distribution will be done for all the different settings in k_range and time_range. 
    """

    # min and max qubits. on my laptop, 12 is the maximum amount 
    min_qubits = 2
    max_qubits = 4

    # selected trotter steps to go through, can be any length 
    k_range = [5]

    # selected times to go through, can be of any length 
    # a special note: do not choose t=1 when k=3.. that happens to produce gates with 0 rotation that are compiled out!
    # time_range = np.random.uniform(1,2,1)
    time_range = [.1]

    # methods to go through, can be list of length 1 or 2 
    methods = [1]

    # 2Q fidelity with depolarization model, should be length 2 for script to work. default, from Charlie Baldwin's graph, is .95 and .995. 
    f_range = [.95, 1]

    for k in k_range:
        for t in time_range:

            # overwrite the precalculated_data inside of hamiltonian_simulation_benchmark for the settings desired 
            # does this in a global manner by editing the hamiltonian_simulation_benchmark module
            set_precalculated_data(w=1, k=k, t=t, min_qubits = min_qubits, max_qubits = max_qubits)

            for method in methods:

                for f in f_range: 

                    for use_pytket in [False, True]:

                        noise = create_noise_model(f)

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
                                "noise_model": noise,
                            }
                        else:
                            exec_options = {"noise_model": noise}

                        suffix = f"{k}_{t}_{method}_{f}_{use_pytket}"

                        # this produces images in ./__images/qasm_simulator, and we use those to make combined images
                        ham.run(
                            min_qubits=min_qubits,
                            max_qubits=max_qubits,
                            method=method,
                            exec_options=exec_options,
                            suffix=suffix,
                        )

                    image_suffix = f"{k}_{t}_{method}_{f}".replace(".","")
                    # construct and save example transpiled (pre-compiled) circuit 
                    qc = ham.HamiltonianSimulation((min_qubits + max_qubits)//2, K=k, t=t, method=method, measure_x= False) 
                    transpile(qc,ex.backend, optimization_level=0).draw("mpl", filename="qc_" + image_suffix + "_False")

                    # construct and save example compiled (pytket) circuit 
                    compiled_qc = high_optimisation(qc, backend=ex.backend)
                    compiled_qc.draw("mpl", filename="qc_" + image_suffix + "_True")

                        # the code in this for loop will generate a bunch of images for all the different specified methods, fidelities, and use of pytket. 

                benchmark_result_images = []

                for f in f_range:
                    for use_pytket in [False, True]: 

                        suffix = f"{k}_{t}_{method}_{f}_{use_pytket}"

                        file_name = "__images/qasm_simulator/Hamiltonian-Simulation-vplot" + suffix + ".jpg" 

                        benchmark_result_images.append(Image.open(file_name))

                save_and_combine_images(benchmark_result_images, f"combined_vplots_method_{method}_{k}_{t}" + ".jpg", method)

                circuit_images = []

                for f in f_range:
                    for use_pytket in [False, True]: 

                        suffix = f"{k}_{t}_{method}_{f}_{use_pytket}".replace(".","")

                        file_name = "qc_" + suffix + ".png" 

                        circuit_images.append(Image.open(file_name))

                save_and_combine_images(circuit_images, f"combined_circuit_plot_{method}_{k}_{t}" + ".jpg", method)
