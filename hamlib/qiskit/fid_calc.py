import json
import numpy as np
import matplotlib.pyplot as plt
import itertools


def load_results(filename="results.json"):
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def analyze_results(data):
    # want this to be like num_qubits_list, method_1_half_fideltieis: [2.22..], blbablablabla.
    fidelities = {"num_qubits_list": []}

    for num_qubits, results_by_qubit in data.items():
        fidelities["num_qubits_list"].append(num_qubits)

        for experiment_name, experimental_data in results_by_qubit[
            "experiments"
        ].items():
            if experiment_name not in fidelities:
                fidelities[experiment_name] = []

            if "K = 10" in experiment_name or "Method 3" in experiment_name:
                fidelities[experiment_name].append(
                    np.sqrt(experimental_data["fidelity"]["hf_fidelity"])
                )

            elif "K = 5" in experiment_name:
                fidelities[experiment_name].append(
                    experimental_data["fidelity"]["hf_fidelity"]
                )

            else:
                raise Exception

    return fidelities


def plot_fidelities(fidelities, plot_title):
    plt.figure(figsize=(10, 6))

    num_qubits = fidelities.pop("num_qubits_list")

    marker = itertools.cycle(("^", ">", "<", "v"))

    for index, (experiment_name, fidelity_list) in enumerate(fidelities.items()):
        if "K = 10" in experiment_name or "Method 3" in experiment_name:
            label = " " + experiment_name + " sqrt fidelity"
        else:
            label = " " + experiment_name + " fidelity"

        plt.plot(
            num_qubits,
            fidelity_list,
            marker=next(marker),
            linestyle="-",
            label=label,
            markersize=10,
        )

    # manually add fresh new method = 3 data
    # below are commented out, only have 1000 shots
    # method_3_average_random_paulis = np.sqrt([0.917, 0.86, 0.766, 0.693, 0.663, 0.618, 0.567, 0.506, 0.474, 0.419])

    # same thing as above but with 1,000,000 shots
    #
    # TFIM
    # method_3_average_random_paulis = np.sqrt(
    #   [
    #     0.926,
    #     0.852,
    #     0.787,
    #     0.731,
    #     0.666,
    #     0.631,
    #     0.578,
    #     0.536,
    #     0.495,
    #     0.416
    #   ],
    # )
    #Heisenberg 
    method_3_average_random_paulis = np.sqrt([
        0.782,
        0.573,
        0.421,
        0.309,
        0.229,
        0.174,
        0.126,
        0.093,
        0.066,
        0.038
      ])
    plt.plot(
        num_qubits,
        method_3_average_random_paulis,
        marker=next(marker),
        linestyle="-",
        label=' Method 3 "Truly Random" Paulis K = 5 sqrt fidelity',
        markersize=10,
    )
    plt.xlabel("Number of Qubits")
    plt.ylabel("Hellinger Fidelity")
    plt.title("Hellinger Fidelities Vs Number of Qubits for " + plot_title)
    plt.legend()
    plt.grid(True)
    plt.savefig("fidelity_comparison" + plot_title[:-5] + ".png")
    plt.show()


if __name__ == "__main__":
    filename = "hamlib_heis.json"
    data = load_results(filename)
    fidelities = analyze_results(data)
    plot_fidelities(fidelities, filename)
