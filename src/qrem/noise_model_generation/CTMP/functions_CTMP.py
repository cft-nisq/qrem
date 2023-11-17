"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""

from collections import defaultdict
from typing import Dict, Tuple, List, Union, Optional

from tqdm import tqdm
import numpy as np
from qrem.common import convert, utils

from qrem.functions_qrem import quantum_ancillary_functions as quanf, ancillary_functions as anf
from qrem.functions_qrem import ground_state_approximations as gsa


#(PP) dobuled in other files
#(PP) part of a model of Bravi noise
#(PP) should go to qrem.noise_modelling
def _return_1local_gen(operator_type):
    if operator_type == '0':
        gen = [[-1, 0],
               [1, 0]]
    elif operator_type == '1':
        gen = [[0, 1],
               [0, -1]]
    else:
        raise ValueError('Wrong generator type!')
    return np.array(gen)


#(PP) dobuled in other files
#(PP) part of a model of Bravi noise
#(PP) should go to qrem.noise_modelling
def _return_2local_gen(operator_type):
    if operator_type == '00':
        gen = [[-1, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 0, 0, 0]]
    elif operator_type == '01':
        gen = [[0, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0]]
    elif operator_type == '10':
        gen = [[0, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 0]]
    elif operator_type == '11':
        gen = [[0, 0, 0, 1],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, -1]]
    else:
        raise ValueError('Wrong generator type!')
    return np.array(gen)


def __convert_rates_dictionary_into_hamiltonian(ctmp_rates_dictionary,
                                                number_of_qubits):
    constant = 0
    hamiltonian_weights = defaultdict(float)
    for qubits_subset, rates_dic in ctmp_rates_dictionary.items():
        if len(qubits_subset) == 1:
            r0, r1 = rates_dic['0'], rates_dic['1']
            constant += (r0 + r1) / 2
            hamiltonian_weights[qubits_subset] += (r0 - r1) / 2


        elif len(qubits_subset) == 2:
            r00, r01, r10, r11 = rates_dic['00'], rates_dic['01'], rates_dic['10'], rates_dic['11']

            constant += (r00 + r01 + r10 + r11) / 4

            hamiltonian_weights[(qubits_subset[0],)] += (r00 + r01 - r10 - r11) / 4
            hamiltonian_weights[(qubits_subset[1],)] += (r00 - r01 + r10 - r11) / 4
            hamiltonian_weights[qubits_subset] += (r00 - r01 - r10 + r11) / 4

    return hamiltonian_weights, constant


def __get_G_energy_for_fixed_state(input_state: Union[str, List[Union[str, int]]],
                                   ctmp_rates_dictionary: Dict[Tuple[int], Dict[str, float]],
                                   number_of_qubits: int,
                                   qubits_mapping: Dict[int, int] = None):
    if qubits_mapping is None:
        qubits_mapping = {qi: qi for qi in range(number_of_qubits)}

    part_1q, part_2q = 0, 0
    for subset, local_dict in ctmp_rates_dictionary.items():

        if len(subset) == 1:
            qi = qubits_mapping[subset[0]]
            part_1q += ctmp_rates_dictionary[subset][input_state[qi]]
        elif len(subset) == 2:
            qi, qj = qubits_mapping[subset[0]], qubits_mapping[subset[1]]
            part_2q += ctmp_rates_dictionary[subset][input_state[qi] + input_state[qj]]
        else:
            raise ValueError(f"Wrong rates dictionary with key: 'subset'.")

    return part_1q + part_2q


def return_generator_matrix(operator_type):
    if len(operator_type) == 1:
        return _return_1local_gen(operator_type=operator_type)
    elif len(operator_type) == 2:
        return _return_2local_gen(operator_type=operator_type)
    else:
        raise ValueError('Wrong generator dimension in r_dict!')


def get_generators_dictionary_from_rates(rates_dictionary):
    generators_dictionary = {}
    for qubits_tuple in rates_dictionary.keys():
        operator_now = 0
        for operator_type in rates_dictionary[qubits_tuple].keys():
            operator_now += return_generator_matrix(operator_type=operator_type) * \
                            rates_dictionary[qubits_tuple][operator_type]

        generators_dictionary[qubits_tuple] = operator_now

    return generators_dictionary


def get_global_generator_from_rates(rates_dictionary: Dict[Tuple[int], Dict[str, float]],
                                    number_of_qubits: int):
    global_noise_matrix = np.zeros((2 ** number_of_qubits, 2 ** number_of_qubits), dtype=float)

    for qubits_tuple in rates_dictionary.keys():
        subset_size = len(qubits_tuple)
        local_dimension = int(2 ** subset_size)

        local_generator_now = np.zeros((local_dimension, local_dimension), dtype=float)
        for operator_type in rates_dictionary[qubits_tuple]:
            error_rate = rates_dictionary[qubits_tuple][operator_type]

            local_generator_now += error_rate * return_generator_matrix(operator_type=operator_type)

        # raise KeyError
        generator_now_embeded = quanf.embed_operator_in_bigger_hilbert_space(
            number_of_qubits=number_of_qubits,
            local_operator=local_generator_now,
            global_indices=qubits_tuple).real

        global_noise_matrix += generator_now_embeded

    return global_noise_matrix


def construct_model_from_rates(rates_dictionary):
    generators_dictionary = {}
    for qubits_tuple in rates_dictionary.keys():
        subset_size = len(qubits_tuple)
        local_dimension = int(2 ** subset_size)

        local_generator_now = np.zeros((local_dimension, local_dimension), dtype=float)
        for operator_type in rates_dictionary[qubits_tuple]:
            error_rate = rates_dictionary[qubits_tuple][operator_type]

            local_generator_now += error_rate * return_generator_matrix(operator_type=operator_type)

        generators_dictionary[qubits_tuple] = local_generator_now
    return generators_dictionary


def _find_optimal_noise_strength_CTMP_bruteforce(
        ctmp_rates_dictionary: Dict[Tuple[int], Dict[str, float]],
        number_of_qubits: int,
        qubits_mapping: Optional[Dict[int, int]] = None):
    if qubits_mapping is None:
        qubits_mapping = {qi: qi for qi in range(number_of_qubits)}

    gamma = 10 ** 6
    gamma_max = 0

    integers_range = range(2 ** number_of_qubits)

    if number_of_qubits > 10:
        integers_range = tqdm(integers_range)

    for integer in integers_range:
        bitstring = convert.integer_to_bitstring(integer=integer, number_of_bits=number_of_qubits)

        potential_gamma = __get_G_energy_for_fixed_state(input_state=bitstring,
                                                         ctmp_rates_dictionary=ctmp_rates_dictionary,
                                                         number_of_qubits=number_of_qubits,
                                                         qubits_mapping=qubits_mapping
                                                         )

        if potential_gamma < gamma:
            gamma = potential_gamma

        if potential_gamma > gamma_max:
            gamma_max = potential_gamma
    # print(gamma, gamma_max)
    return gamma, gamma_max


def _find_optimal_noise_strength_CTMP_SDP_relaxation(
        ctmp_rates_dictionary: Dict[Tuple[int], Dict[str, float]],
        number_of_qubits: int,
        hierarchy_level: Optional[int] = 1) -> Tuple[float]:
    hamiltonian_dictionary, constant = __convert_rates_dictionary_into_hamiltonian(
        ctmp_rates_dictionary=ctmp_rates_dictionary,
        number_of_qubits=number_of_qubits)

    lower_bound, upper_bound = gsa.find_ground_state_with_SDP_relaxations(
        weights_dictionary=hamiltonian_dictionary,
        number_of_qubits=number_of_qubits,
        hierarchy_level=hierarchy_level)

    return lower_bound + constant, upper_bound + constant


def get_CTMP_error_rates_sum(rates_dictionary: Dict[Tuple[int], Dict[str, float]]):
    noise_strength_CTMP = 0
    for subset in rates_dictionary.keys():
        for operator_type in rates_dictionary[subset].keys():
            noise_strength_CTMP += rates_dictionary[subset][operator_type]

    return noise_strength_CTMP


def find_optimal_noise_strength_CTMP(ctmp_rates_dictionary: Dict[Tuple[int], Dict[str, float]],
                                     number_of_qubits: int,
                                     method_name='SDP_relaxation',
                                     method_kwargs=None):
    if method_name.upper() == 'BRUTEFORCE':
        if method_kwargs is None:
            method_kwargs = {'qubits_mapping': None}
        _, noise_strength_CTMP = _find_optimal_noise_strength_CTMP_bruteforce(
            ctmp_rates_dictionary=ctmp_rates_dictionary,
            number_of_qubits=number_of_qubits,
            **method_kwargs)
    elif method_name.upper() == 'SDP_RELAXATION':
        if method_kwargs is None:
            method_kwargs = {'hierarchy_level': 1}

        _, noise_strength_CTMP = _find_optimal_noise_strength_CTMP_SDP_relaxation(
            ctmp_rates_dictionary=ctmp_rates_dictionary,
            number_of_qubits=number_of_qubits,
            **method_kwargs)

    elif method_name.upper() == 'SUM_OF_RATES':
        noise_strength_CTMP = get_CTMP_error_rates_sum(rates_dictionary=ctmp_rates_dictionary)
    else:
        raise ValueError(f"Method name '{method_name}' not recognized.\n "
                         f"Please choose one of the following:"
                         f" - 'SDP_relaxation' "
                         f" - 'bruteforce' "
                         f" - 'sum_of_rates' ")
    return noise_strength_CTMP


def get_pairs_complement_dictionary_from_clusters_list(clusters_list: List[Tuple[int]]) -> Dict[
    Tuple[int], List[int]]:
    """

    :param clusters_list: List of clusters in the noise model
    :type clusters_list:
    :return:
    :rtype:
    """

    pairs_complements_dictionary = {}
    number_of_qubits = int(np.max([np.max(cl) for cl in clusters_list]))
    qubit_to_cluster_map = {}
    for cluster in clusters_list:
        for qubit_index in cluster:
            qubit_to_cluster_map[(qubit_index,)] = cluster

    all_qubits = list(range(number_of_qubits))
    for qi in range(number_of_qubits):
        cluster = qubit_to_cluster_map[(qi,)]
        not_cluster = list(set(all_qubits).difference(set(cluster)))

        for qj in not_cluster:
            pairs_complements_dictionary[tuple(sorted((qi, qj)))] = []

        for qj in cluster:
            if qj > qi:
                pairs_complements_dictionary[(qi, qj)] = list(set(cluster).difference(set((qi, qj))))
    return pairs_complements_dictionary


def get_theoretical_overhead_CTMP(noise_strength_CTMP: float, ):
    return np.exp(4 * noise_strength_CTMP)


def get_theoretical_number_of_samples_CTMP(noise_strength_CTMP: float,
                                           target_estimation_accuracy: float):
    return np.ceil(4 * np.exp(4 * noise_strength_CTMP) / target_estimation_accuracy ** 2)


def _convert_to_local_noise_rates_dictionary(CTMP_rates_dictionary,
                                             clusters_list: List[Tuple[int]]):
    local_rates_dictionaries = {cluster: {} for cluster in clusters_list}
    for cluster in clusters_list:
        pairs_in_cluster = [(qi, qj) for qi in cluster for qj in cluster if qj > qi]
        cluster_mapping = utils.map_index_to_order(cluster)

        for (qi, qj) in pairs_in_cluster:
            local_rates_dictionaries[cluster][(cluster_mapping[qi],
                                               cluster_mapping[qj])] = CTMP_rates_dictionary[
                (qi, qj)]

        for q0 in cluster:
            local_rates_dictionaries[cluster][(cluster_mapping[q0],)] = \
                CTMP_rates_dictionary[(q0,)]

    return local_rates_dictionaries


def calculate_noise_strengths_local(
        local_rates_dictionaries,
        method='bruteforce'):


    local_noise_strengths = {}
    for cluster, local_rates_dictionary in local_rates_dictionaries.items():
        noise_strength_CTMP = find_optimal_noise_strength_CTMP(
            ctmp_rates_dictionary=local_rates_dictionary,
            number_of_qubits=len(cluster),
            method_name=method)

        local_noise_strengths[cluster] = noise_strength_CTMP

    return local_noise_strengths


def get_local_noise_matrices(
        local_rates_dictionaries,
        local_noise_strengths
):
    local_generators_dictionary_nested = {}

    clusters_list = list(local_rates_dictionaries.keys())

    for cluster, local_rates_dictionary in local_rates_dictionaries.items():
        local_generators_dictionary_nested[cluster] = get_generators_dictionary_from_rates(
            rates_dictionary=local_rates_dictionary)

    local_noise_matrices = {}
    for cluster in clusters_list:
        # enumerated_indices = utils.map_index_to_order(cluster)
        number_of_qubits_local = len(cluster)
        iden = np.eye(2 ** len(cluster), dtype=float)

        local_generator = 0
        local_generators_now = local_generators_dictionary_nested[cluster]
        for key, value in local_generators_now.items():
            local_generator += quanf.embed_operator_in_bigger_hilbert_space(
                number_of_qubits=number_of_qubits_local,
                local_operator=value,
                global_indices=key).real

        local_noise_matrices[cluster] = abs(
            iden + 1 / local_noise_strengths[cluster] * local_generator)

    return local_noise_matrices
