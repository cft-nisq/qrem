"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""
from typing import List
import numpy as np

from qrem.functions_qrem import ancillary_functions as anf

from qrem.common.printer import qprint
from qrem.common import convert

def get_parallel_tomography_on_non_overlapping_subsets(number_of_circuits: int,
                                                       number_of_symbols: int,
                                                       non_overlapping_subsets: List[List[int]],
                                                       ) -> List[List[int]]:
    """
    :param number_of_circuits: should be power of number_of_symbols
    :param non_overlapping_subsets:
    :param number_of_symbols:
    :return:
    :rtype:
    """


    lengths_list = [len(x) for x in non_overlapping_subsets]
    number_of_qubits = sum(lengths_list)
    unique_lengths = np.unique(lengths_list)
    maximal_length = int(np.max(unique_lengths))


    if number_of_circuits % number_of_symbols != 0:
        number_of_circuits2 = -int(number_of_circuits // -maximal_length) * maximal_length
        qprint(
            f"Number of circuits should be power of {number_of_symbols}.\nChanging number of circuits from {number_of_circuits} to {number_of_circuits2}")
        number_of_circuits = number_of_circuits2

    if number_of_circuits<number_of_symbols**maximal_length:
        number_of_circuits2 = int(number_of_symbols**maximal_length)
        qprint(f"Number of circuits should be at least {number_of_symbols}^{maximal_length}={number_of_symbols**maximal_length} for subsets of locality at most {maximal_length}.\nChanging number of circuits from {number_of_circuits} to {number_of_circuits2}")
        number_of_circuits = number_of_circuits2

    local_registers = {length: convert.get_ditstrings_register(base=number_of_symbols,
                                                           number_of_dits=length)
                       for length in unique_lengths}

    enumerated_subsets = dict(enumerate(non_overlapping_subsets))

    circuits_subsets = []
    rng = np.random.default_rng()
    for subset in non_overlapping_subsets:
        local_register = local_registers[len(subset)]
        number_of_repetitions = number_of_circuits // len(local_register)
        all_states_local = local_register * number_of_repetitions
        rng.shuffle(all_states_local)
        circuits_subsets.append(all_states_local)

    circuits_tuples = list(zip(*circuits_subsets))
    circuits_list_integers = []
    for circ_tuple in circuits_tuples:
        circuit_integers = np.zeros((number_of_qubits),dtype=int)
        for subset_index in range(len(circ_tuple)):
            subset_now = enumerated_subsets[subset_index]
            local_circ_now = circ_tuple[subset_index]

            for qubit_index in range(len(subset_now)):
                circuit_integers[subset_now[qubit_index]] = local_circ_now[qubit_index]

        circuit_integers = list(circuit_integers)
        circuits_list_integers.append(circuit_integers)

    return circuits_list_integers

# number_of_qubits = 150
# number_of_symbols=5
# number_of_circuits = 100*number_of_symbols**2
#
# subsets = [[2*i,2*i+1] for i in range(number_of_qubits//2)]
# # print(subsets)
#
# circuits_list = get_parallel_tomography_on_non_overlapping_subsets(number_of_circuits=number_of_circuits,
#                                          number_of_symbols=number_of_symbols,
#                                          non_overlapping_subsets=subsets)
# # print(circuits_list)




