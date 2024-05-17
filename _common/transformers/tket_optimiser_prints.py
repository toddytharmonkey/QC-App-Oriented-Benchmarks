from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (  # type: ignore
    auto_rebase_pass,
    RemoveRedundancies,
    SynthesiseTket,
    CXMappingPass,
    DecomposeBoxes,
    FullPeepholeOptimise,
    CliffordSimp,
    SimplifyInitial,
    KAKDecomposition,
    PauliSimp,
    RemoveBarriers,
    #RemoveImplicitQubitPermutation,
    RebaseTket,
)
from pytket.architecture import Architecture
from pytket.placement import NoiseAwarePlacement
from pytket.extensions.qiskit.qiskit_convert import (
    process_characterisation,
    get_avg_characterisation,
)
from pytket import OpType
from qiskit.circuit.quantumcircuit import QuantumCircuit

from matplotlib.pyplot import figure
# step_index defined for testing purposes
step_index = 0 
def print_tket_circuit(tket_circuit, title):
    """
    Turn a tket circuit into a qiskit one, then print it and wait for user input
    """

    global step_index

    qc = tk_to_qiskit(tket_circuit)
    qc.draw("mpl", filename= f"test/step_{step_index}" + title)
    step_index = step_index + 1 


def rebase_pass():
    return auto_rebase_pass({OpType.CX, OpType.X, OpType.SX, OpType.Rz})


def tket_transformer_generator(
    cx_fidelity=1.0,
    pauli_simp=False,
    remove_barriers=True,
    resynthesise=True,
):
    """Generator for transformer using TKET passes

    :param quick: Perform quick optimisation, defaults to False
    :type quick: bool, optional
    :param cx_fidelity: Estimated CX gate fidelity, defaults to 1.0
    :type cx_fidelity: float, optional
    :param pauli_simp: True if the circuit contains a large number of
        Pauli gadgets (exponentials of pauli strings).
    :type pauli_simp: bool, optional
    """

    def transformation_method(
        circuit: QuantumCircuit, backend
    ) -> list([QuantumCircuit]):
        """Transformer using TKET optimisation passes.

        :param circuit: Circuit to be optimised
        :type circuit: QuantumCircuit
        :param backend: Backed on which circuit is run.
        :return: List of transformed circuits
        :rtype: list[QuantumCircuit]
        """

        global step_index

        circuit.draw("mpl", filename= "initial_qiskit_circuit")

        tk_circuit = qiskit_to_tk(circuit)
        print_tket_circuit(tk_circuit, "start")
        if remove_barriers:
            RemoveBarriers().apply(tk_circuit)
        DecomposeBoxes().apply(tk_circuit)

        print_tket_circuit(tk_circuit, "decomposeboxes")

        # Obtain device data for noise aware placement.
        characterisation = process_characterisation(backend)
        averaged_errors = get_avg_characterisation(characterisation)

        # If the circuit contains a large number of pauli gadgets,
        # run PauliSimp
        if pauli_simp:
            if len(tk_circuit.commands_of_type(OpType.Reset)) > 0:
                raise Exception(
                    "PauliSimp does not support reset operations."
                )
            auto_rebase_pass(
                {OpType.CX, OpType.Rz, OpType.Rx}
            ).apply(tk_circuit)
            PauliSimp().apply(tk_circuit)

        # Initialise pass list and perform thorough optimisation.
        if resynthesise:
            FullPeepholeOptimise().apply(tk_circuit)
            print_tket_circuit(tk_circuit, "fullpeepholeoptimise")


        # Add noise aware placement and routing to pass list.
        coupling_map = backend.configuration().coupling_map
        if coupling_map:
            arch = Architecture(coupling_map)
            CXMappingPass(
                arch,
                NoiseAwarePlacement(
                    arch,
                    averaged_errors["node_errors"],
                    averaged_errors["edge_errors"],
                    averaged_errors["readout_errors"],
                ),
                directed_cx=False,
                delay_measures=False,
            ).apply(tk_circuit)

        if coupling_map:
            print_tket_circuit(tk_circuit, "coupling map step")

        if resynthesise:
            KAKDecomposition(cx_fidelity=cx_fidelity).apply(tk_circuit)
            print_tket_circuit(tk_circuit, "KAKDecomposition")
        
        CliffordSimp().apply(tk_circuit)
        print_tket_circuit(tk_circuit, "CliffordSimp")
        SynthesiseTket().apply(tk_circuit)
        print_tket_circuit(tk_circuit, "SynthesiseTket")

        # Rebase to backend gate set and perform basic optimisation
        rebase_pass().apply(tk_circuit)
        print_tket_circuit(tk_circuit, "rebase_pass")

        RemoveRedundancies().apply(tk_circuit)
        print_tket_circuit(tk_circuit, "RemoveRedundancies")
        SimplifyInitial(
            allow_classical=False,
            create_all_qubits=True,
        ).apply(tk_circuit)
        print_tket_circuit(tk_circuit, "SimplifyInitial_Finalstep")

        circuit = tk_to_qiskit(tk_circuit)

        return circuit

    return transformation_method
