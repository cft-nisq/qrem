"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""

import copy
from typing import Optional, List
from collections import Counter

import numpy as np
from tqdm import tqdm

from qrem.functions_qrem import functions_data_analysis as fda

from pyquil.gates import RX, RZ, MEASURE
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference

from qrem.common.printer import qprint


#MOVE_TO qrem.common.math ? 
# simpler circuits
__dictionary_with_angles_pauli_eigenstates__ = \
    {"Z+": {"RZ": (0, 0, 0),
            "RX": (0, 0)},

     "Z-": {"RZ": (-0.9199372448290238, np.pi, 2.2216554087607694),
            "RX": (np.pi / 2, -np.pi / 2)},

     "X+": {"RZ": (np.pi / 2, np.pi / 2, 0),
            "RX": (np.pi / 2, 0)},

     "X-": {"RZ": (np.pi / 2, -np.pi / 2, 0),
            "RX": (np.pi / 2, 0)},

     "Y+": {"RZ": (np.pi / 2, np.pi, 0),
            "RX": (np.pi / 2, 0)},

     "Y-": {"RZ": (np.pi / 2, 0, 0),
            "RX": (np.pi / 2, 0)},
     }

enumerated_labels_pauli_eigenstates = dict(enumerate(['Z+', 'Z-',
                                                      "X+", 'X-',
                                                      "Y+", "Y-"]))


def _apply_elementary_gate_native(quantum_program,
                                  qubit_index: int,
                                  gate_label: str):
    # This funcion decomposes elementary quantum gates (I, X, Y, Z, H, S, T) into gates native to rigetti devices
    if gate_label.upper() == 'I':
        # quantum_circuit+=Instruction(Gate.I(), qubit_index)
        quantum_program = quantum_program.rz(qubit_index, 0)

    elif gate_label.upper() == 'X':
        # X
        quantum_program = quantum_program.rz(qubit_index, np.pi)
        quantum_program = quantum_program.rx(qubit_index, np.pi / 2)
        quantum_program = quantum_program.rz(qubit_index, np.pi)
        quantum_program = quantum_program.rx(qubit_index, -np.pi / 2)

    elif gate_label.upper() == 'Y':
        # Y
        quantum_program = quantum_program.rz(qubit_index, -2.446121240088584)
        quantum_program = quantum_program.rx(qubit_index, np.pi / 2)
        quantum_program = quantum_program.rz(qubit_index, np.pi)
        quantum_program = quantum_program.rx(qubit_index, -np.pi / 2)
        quantum_program = quantum_program.rz(qubit_index, -2.446121240088584)

    elif gate_label.upper() == 'Z':
        # Z
        quantum_program = quantum_program.rz(qubit_index, np.pi)


    elif gate_label.upper() == 'H':
        # H
        quantum_program = quantum_program.rz(qubit_index, np.pi)
        quantum_program = quantum_program.rx(qubit_index, np.pi / 2)
        quantum_program = quantum_program.rz(qubit_index, np.pi / 2)
        quantum_program = quantum_program.rx(qubit_index, -np.pi / 2)


    elif gate_label.upper() == 'S':
        # S
        quantum_program = quantum_program.rz(qubit_index, np.pi / 2)


    elif gate_label.upper() == 'S*':
        # Sdag
        quantum_program = quantum_program.rz(qubit_index, -np.pi / 2)

    elif gate_label.upper() == 'T':
        # T
        quantum_program = quantum_program.rz(qubit_index, np.pi / 4)

    elif gate_label.upper() == 'T*':
        # Tdag
        quantum_program = quantum_program.rz(qubit_index, -np.pi / 4)

    else:
        raise ValueError(f"Gate label: '{gate_label}' not recognized.")

    return quantum_program


def _declare_generic_DOT_gate_parametric(quantum_program,
                                         qubit_index: int,
                                         number_of_rz_rotations=3,
                                         number_of_rx_rotations=2,
                                         name_suffix='',
                                         name_prefix=''):
    quantum_program.declare(f"{name_prefix}RZ_q-{qubit_index}{name_suffix}", "REAL",
                            number_of_rz_rotations)
    quantum_program.declare(f"{name_prefix}RX_q-{qubit_index}{name_suffix}", "REAL",
                            number_of_rx_rotations)

    return quantum_program


def _apply_generic_DOT_gate_parametric(quantum_program,
                                       qubit_index_physical: int):
    """

    :param quantum_program:
    :type quantum_program:
    :param qubit_index_physical:
    :type qubit_index_physical:

    :return:
    :rtype:
    """

    quantum_program += RZ(MemoryReference(f"RZ_q-{qubit_index_physical}", 0), qubit_index_physical)
    quantum_program += RX(MemoryReference(f"RX_q-{qubit_index_physical}", 0), qubit_index_physical)
    quantum_program += RZ(MemoryReference(f"RZ_q-{qubit_index_physical}", 1), qubit_index_physical)
    quantum_program += RX(MemoryReference(f"RX_q-{qubit_index_physical}", 1), qubit_index_physical)
    quantum_program += RZ(MemoryReference(f"RZ_q-{qubit_index_physical}", 2), qubit_index_physical)

    return quantum_program


def create_memory_map_DOT(list_of_keys: List[str],
                          qubit_indices: List[int]):
    if isinstance(list_of_keys, dict):
        list_of_keys = list(list_of_keys.keys())
    elif isinstance(list_of_keys, str):
        list_of_keys = [list_of_keys]

    number_of_qubits = len(qubit_indices)

    memory_map = {}
    for qubit_index_enumerated in range(number_of_qubits):
        qubit_index_physical = qubit_indices[qubit_index_enumerated]
        memory_map[f"RZ_q-{qubit_index_physical}"] = []
        memory_map[f"RX_q-{qubit_index_physical}"] = []

    for circuit_key in list_of_keys:
        for qubit_index_enumerated in range(number_of_qubits):
            gate_index = int(circuit_key[qubit_index_enumerated])
            angles_now = __dictionary_with_angles_pauli_eigenstates__[
                enumerated_labels_pauli_eigenstates[gate_index]]
            angles_RZ_now = list(angles_now['RZ'])
            angles_RX_now = list(angles_now['RX'])

            qubit_index_physical = qubit_indices[qubit_index_enumerated]
            memory_map[f"RZ_q-{qubit_index_physical}"].append(angles_RZ_now)
            memory_map[f"RX_q-{qubit_index_physical}"].append(angles_RX_now)

    return memory_map


def get_generic_base_program_DOT(qubit_indices: List[int],
                                 compilation_name: Optional[str] = 'parametric-native'):
    # if sdk_name.upper() in ['PYQUIL-FOR-AZURE-QUANTUM']:

    classical_register_size = len(qubit_indices)

    quantum_program = Program()
    quantum_program.declare("ro",
                            "BIT",
                            classical_register_size)

    if compilation_name.upper() in ['PARAMETRIC-NATIVE', 'NATIVE-PARAMETRIC']:
        for qubit_index in qubit_indices:
            quantum_program = _declare_generic_DOT_gate_parametric(quantum_program=quantum_program,
                                                                   qubit_index=qubit_index)

        for qubit_index in qubit_indices:
            quantum_program = _apply_generic_DOT_gate_parametric(quantum_program=quantum_program,
                                                                 qubit_index_physical=qubit_index)
    else:
        raise ValueError(f"Wrong compilation method: '{compilation_name}'")

    for qubit_index_enumerated in range(len(qubit_indices)):
        quantum_program += MEASURE(qubit_indices[qubit_index_enumerated],
                                   MemoryReference(f"ro",
                                                   qubit_index_enumerated))

    return quantum_program


def get_backend_wrapper(backend_name,
                        sdk_name):
    if sdk_name.upper() in ['PYQUIL-FOR-AZURE-QUANTUM']:

        from pyquil_for_azure_quantum import get_qpu, get_qvm, get_qc

        if backend_name.upper() in ['ASPEN-M-2', 'ASPEN-11']:
            backend_instance = get_qpu(backend_name)
        else:

            if backend_name.upper() in ['QVM']:
                backend_instance = get_qvm()

            else:
                if backend_name.upper() in ['ASPEN-M-2-QVM', 'ASPEN-11-QVM']:
                    # test = backend_name
                    # print('test', backend_name[0:-4])
                    backend_instance = get_qc(backend_name[0:-4],
                                              as_qvm=True,
                                              compiler_timeout=60 * 30,
                                              execution_timeout=60 * 60 * 24)
                else:
                    backend_instance = get_qc(backend_name,
                                              as_qvm=True,
                                              compiler_timeout=60 * 30,
                                              execution_timeout=60 * 60 * 24
                                              )

    elif sdk_name.upper() in ['PYQUIL']:
        from pyquil import get_qc
        backend_instance = get_qc(backend_name)
    else:
        raise ValueError(f"Wrong backend name: '{backend_name}'.")

    return backend_instance


def run_batches_parametric(
        backend_name: str,
        sdk_name: str,
        number_of_shots: int,
        qubit_indices: List[int],
        base_program,
        memory_map,
        compilation_method: Optional[str] = 'parametric-native',
):
    # TODO FBM: add batching jobs

    if base_program is None:
        base_program = get_generic_base_program_DOT(qubit_indices=qubit_indices)

    base_program.wrap_in_numshots_loop(shots=number_of_shots)

    backend_instance = get_backend_wrapper(backend_name=backend_name,
                                           sdk_name=sdk_name)

    target_number_of_circuits = len(list(memory_map.values())[0])

    qprint('\nSending jobs to execution on: ', backend_name + '.')
    qprint('Number of shots: ', str(number_of_shots) + ' .')
    qprint('Target number of circuits: ', str(target_number_of_circuits) + ' .')
    qprint("Compilation method:", compilation_method)

    # TODO FBM: add handling errors, waiting etc

    if compilation_method.upper() in ['NATIVE-PARAMETRIC', 'PARAMETRIC-NATIVE']:

        # get native quil commands
        native_quil = backend_instance.compiler.quil_to_native_quil(base_program,
                                                                    # compilatio=60*60
                                                                    )
        if backend_name.upper() in ['ASPEN-M-2', 'ASPEN-11']:

            # create executable
            executable = backend_instance.compile(native_quil,
                                                  to_native_gates=False)
            results = backend_instance.run_batch(executable,
                                                 memory_map)
        else:
            results = []
            for circuit_index in tqdm(range(target_number_of_circuits)):
                native_quil_now = copy.deepcopy(native_quil)
                for angle_label, angles_values_list in memory_map.items():
                    angles_values_now = angles_values_list[circuit_index]
                    for index_angle in range(len(angles_values_now)):
                        native_quil_now.write_memory(region_name=angle_label,
                                                     value=angles_values_now[index_angle],
                                                     offset=index_angle)

                executable = backend_instance.compile(program=native_quil_now,
                                                      to_native_gates=True)
                results_now = backend_instance.run(executable)
                results.append(results_now)
            # executable = backend_instance.compile(native_quil,
            #                                       to_native_gates=True)
            # results = backend_instance.run_experiment(base_program,
            #                                      memory_map=memory_map)



    else:
        raise ValueError(f"Wrong compilation method: '{compilation_method}'.")

    return results


def convert_results_to_counts(single_result):
    bitstrings_list = single_result.readout_data['ro']
    to_strings = [''.join([str(s) for s in x]) for x in bitstrings_list]

    return dict(Counter(to_strings))


def _convert_results_to_counts_tuples(single_result):
    bitstrings_list = single_result.readout_data['ro']
    to_strings = [tuple(x) for x in bitstrings_list]

    return dict(Counter(to_strings))


def convert_results_to_counts_dictionaries_DOT(list_of_circuits_labels,
                                               results_list):
    results_dictionary = {}

    for circuit_index in tqdm(range(len(list_of_circuits_labels))):
        label_now = ''.join([str(s) for s in list_of_circuits_labels[circuit_index]])
        result_now = results_list[circuit_index]
        counts_now = _convert_results_to_counts_tuples(result_now)

        if label_now not in results_dictionary.keys():
            new_dict = counts_now
        else:
            existing_counts = results_dictionary[label_now]
            new_dict = fda.merge_multiple_counts_dictionaries([existing_counts, counts_now])

            # x, y = counts_now, existing_counts
            # new_dict = {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}
            #

        results_dictionary[label_now] = new_dict

    formatted_results_dictionary = {
        label_circuit: {''.join([str(x) for x in bitstring_tuple]): ticks for bitstring_tuple, ticks in
                        counts_dictionary.items()} for label_circuit, counts_dictionary in
        tqdm(results_dictionary.items())}

    return formatted_results_dictionary
#
# def run_batches(batches,
#                 backend_name,
#                 number_of_shots,
#                 sdk_name='pyquil',
#                 saving_IDs_dictionary={'saving': False,
#                                        'directory': None,
#                                        'file_name': None,
#                                        'dictionary_to_save': {}},
#                 provider_data: Optional[Dict[str, str]] = None):
#     qprint('\nSending jobs to execution on: ', backend_name + '.')
#     qprint('Number of shots: ', str(number_of_shots) + ' .')
#     qprint('Target number of jobs: ', str(len(batches)) + ' .')
#
#     batches_wrapped_with_shots = \
#         [[program.wrap_in_numshots_loop(number_of_shots) for program in batch] for batch in batches]
#
#     if backend_name in ['qvm']:
#         if sdk_name.lower() in ['pyquil-for-azure-quantum']:
#             pass
#
#
#
#         elif sdk_name.lower() in ['pyquil']:
#             # TODO FBM: finish this
#             raise ValueError("pyquil not added yet")
#
#
#     else:
#         if sdk_name.lower() in ['pyquil-for-azure-quantum']:
#             pass
#
#
#
#         elif sdk_name.lower() in ['pyquil']:
#             # TODO FBM: finish this
#             raise ValueError("pyquil not added yet")
