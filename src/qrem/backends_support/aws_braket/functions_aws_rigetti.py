"""
@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""
from braket.circuits import Circuit, Gate, Instruction, circuit, Observable
import numpy as np

def _apply_elementary_gate_native(quantum_circuit: Circuit,
                                  qubit_index: int,
                                  gate_label: str)-> Circuit:
    """
    Decomposes quantum gate into elementary rotations in pyquil and adds it to a circuit.

    :param quantum_circuit: Object to which a gate should be added
    :type quantum_circuit: braket.circuits.Circuit

    :param qubit_index: Qubit on which gate should act.
    :type qubit_index: int

    :param gate_label: Symbolic label of the gate. If not supported, returns error.
    Supported gates = ["I", "X", "Y", "Z", "H", "S", "S*", "T", "T*"]
    :type gate_label: str

    :return: "quantum circuit" with a target gate appended.
    :rtype: braket.circuits.Circuit
    """
    if gate_label.upper() == 'I':
        quantum_circuit += Instruction(Gate.I(), qubit_index)
    elif gate_label.upper() == 'X':
        # X
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi)
        quantum_circuit = quantum_circuit.rx(qubit_index, -np.pi / 2)

    elif gate_label.upper() == 'Y':
        # Y
        quantum_circuit = quantum_circuit.rz(qubit_index, -2.446121240088584)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi)
        quantum_circuit = quantum_circuit.rx(qubit_index, -np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, -2.446121240088584)

    elif gate_label.upper() == 'Z':
        # Z
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi)


    elif gate_label.upper() == 'H':
        # H
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rx(qubit_index, -np.pi / 2)


    elif gate_label.upper() == 'S':
        # S
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)


    elif gate_label.upper() == 'S*':
        # Sdag
        quantum_circuit = quantum_circuit.rz(qubit_index, -np.pi / 2)

    elif gate_label.upper() == 'T':
        # T

        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 4)

    elif gate_label.upper() == 'T*':
        # Tdag
        quantum_circuit = quantum_circuit.rz(qubit_index, -np.pi / 4)

    else:
        raise ValueError(f"Gate label: '{gate_label}' not recognized.")

    return quantum_circuit


def _apply_pauli_eigenstate(eigenstate_index: int,
                            quantum_circuit: Circuit,
                            qubit_index: int):
    """

    :param eigenstate_index:
    :type eigenstate_index:
    :param quantum_circuit:
    :type quantum_circuit:
    :param qubit_index:
    :type qubit_index:
    :return:
    :rtype:
    """
    # Z+
    if eigenstate_index == 0:
        quantum_circuit = _apply_elementary_gate_native(quantum_circuit=quantum_circuit,
                                                        qubit_index=qubit_index,
                                                        gate_label='I')

    # Z-
    elif eigenstate_index == 1:
        quantum_circuit = _apply_elementary_gate_native(quantum_circuit=quantum_circuit,
                                                        qubit_index=qubit_index,
                                                        gate_label='X')

    # X+
    elif eigenstate_index == 2:
        quantum_circuit = _apply_elementary_gate_native(quantum_circuit=quantum_circuit,
                                                        qubit_index=qubit_index,
                                                        gate_label='H')
    # Y+
    elif eigenstate_index == 3:
        quantum_circuit = _apply_elementary_gate_native(quantum_circuit=quantum_circuit,
                                                        qubit_index=qubit_index,
                                                        gate_label='H')
        quantum_circuit = _apply_elementary_gate_native(quantum_circuit=quantum_circuit,
                                                        qubit_index=qubit_index,
                                                        gate_label='S')

    # X-
    elif eigenstate_index == 4:
        quantum_circuit = _apply_elementary_gate_native(quantum_circuit=quantum_circuit,
                                                        qubit_index=qubit_index,
                                                        gate_label='X')
        quantum_circuit = _apply_elementary_gate_native(quantum_circuit=quantum_circuit,
                                                        qubit_index=qubit_index,
                                                        gate_label='H')
    # Y-
    elif eigenstate_index == 5:
        quantum_circuit = _apply_elementary_gate_native(quantum_circuit=quantum_circuit,
                                                        qubit_index=qubit_index,
                                                        gate_label='X')
        quantum_circuit = _apply_elementary_gate_native(quantum_circuit=quantum_circuit,
                                                        qubit_index=qubit_index,
                                                        gate_label='H')
        quantum_circuit = _apply_elementary_gate_native(quantum_circuit=quantum_circuit,
                                                        qubit_index=qubit_index,
                                                        gate_label='S')

    else:
        raise ValueError(f"Incorrect eigenstate index: '{eigenstate_index}'!")

    return quantum_circuit


