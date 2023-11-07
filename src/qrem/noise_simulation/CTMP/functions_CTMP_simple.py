
from collections import defaultdict
import copy
from typing import Tuple, List
from tqdm import tqdm

import numpy as np

from qrem.functions_qrem import ancillary_functions as anf, quantum_ancillary_functions as quanf
from qrem.functions_qrem import functions_data_analysis as fda
from qrem.functions_qrem import functions_probabilities as fun_prob
from qrem.noise_model_generation.CTMP import functions_CTMP as ctmp_fun
# from qrem.noise_simulation.CTMP.functions_CTMP_conditional import *

from qrem.common import convert, math
# from qrem.common.printer import qprint_array




#MOcomm - poorely documented, needs to be understood







def get_local_stochastic_matrix_from_generator(local_generator):
    return local_generator + np.eye(local_generator.shape[0])


def get_local_stochastic_matrix(operator_type):
    return get_local_stochastic_matrix_from_generator(
        ctmp_fun.return_generator_matrix(operator_type=operator_type))


def get_local_stochastic_matrices_from_rates(rates_dictionary):
    noise_matrices_dictionary = {qubits_tuple: {} for qubits_tuple in rates_dictionary.keys()}

    for qubits_tuple in rates_dictionary.keys():
        for operator_type in rates_dictionary[qubits_tuple].keys():
            noise_matrices_dictionary[qubits_tuple][operator_type] = get_local_stochastic_matrix(
                operator_type=operator_type)

    return noise_matrices_dictionary


def get_probability_distribution_over_error_rates(rates_dictionary,
                                                  normalization=None):
    if normalization is None:
        normalization = ctmp_fun.get_CTMP_error_rates_sum(rates_dictionary=rates_dictionary)

    probability_distribution = {qubits_tuple: {} for qubits_tuple in rates_dictionary.keys()}

    for qubits_tuple in rates_dictionary.keys():
        for operator_type in rates_dictionary[qubits_tuple].keys():
            probability_distribution[qubits_tuple][operator_type] = rates_dictionary[qubits_tuple][
                                                                        operator_type] / normalization

    return probability_distribution


def __sample_from_local_matrix_special(local_input_state: str,
                                       local_matrix: np.ndarray,
                                       ):
    probability_distribution = local_matrix[:, int(local_input_state, 2)]

    return convert.integer_to_bitstring(math.get_sample_from_multinomial_distribution(
        probability_distribution=probability_distribution), len(local_input_state))


def __flatten_probability_over_matrices(probability_over_matrices):
    probability_over_matrices_flattened = []

    for subset in probability_over_matrices.keys():
        for matrix_type in probability_over_matrices[subset].keys():
            probability_over_matrices_flattened.append(probability_over_matrices[subset][matrix_type])

    return probability_over_matrices_flattened


def __flatten_matrices_dictionary(local_noise_matrices_dictionary):
    local_noise_matrices_list = []

    for subset in local_noise_matrices_dictionary.keys():
        for matrix_type in local_noise_matrices_dictionary[subset].keys():
            local_noise_matrices_list.append(
                (subset, local_noise_matrices_dictionary[subset][matrix_type]))

    return local_noise_matrices_list


def _simple_sampling_one_step(global_input_state,
                              probability_over_matrices_list,
                              local_noise_matrices_list: List[Tuple[Tuple[int], np.ndarray]]):
    matrix_index = math.get_sample_from_multinomial_distribution(
        probability_distribution=probability_over_matrices_list)
    chosen_stuff = local_noise_matrices_list[matrix_index]

    chosen_subset, local_noise_matrix = chosen_stuff[0], chosen_stuff[1]
    local_input_state = ''.join([global_input_state[qi] for qi in chosen_subset])

    local_output_state = __sample_from_local_matrix_special(local_matrix=local_noise_matrix,
                                                            local_input_state=local_input_state)

    # print(local_noise_matrix)
    global_output_state = list(copy.deepcopy(global_input_state))

    for qi in range(len(chosen_subset)):
        global_output_state[chosen_subset[qi]] = local_output_state[qi]

    return ''.join(global_output_state)


def sample_from_power_of_B_matrix_simple(global_input_state,
                                         matrix_power,
                                         probability_over_matrices_list,
                                         local_noise_matrices_list
                                         ):
    output_state_now = copy.deepcopy(global_input_state)

    for power_index in range(matrix_power):
        output_state_now = _simple_sampling_one_step(global_input_state=output_state_now,
                                                     probability_over_matrices_list=probability_over_matrices_list,
                                                     local_noise_matrices_list=local_noise_matrices_list)

    return output_state_now

def _get_inputs_to_sampling_algorithm(rates_dictionary):
    noise_strength_CTMP = ctmp_fun.get_CTMP_error_rates_sum(rates_dictionary=rates_dictionary)
    local_noise_matrices = get_local_stochastic_matrices_from_rates(rates_dictionary=rates_dictionary)
    probability_over_matrices = get_probability_distribution_over_error_rates(
        rates_dictionary=rates_dictionary,
        normalization=noise_strength_CTMP)

    local_noise_matrices_flat = __flatten_matrices_dictionary(
        local_noise_matrices_dictionary=local_noise_matrices)
    probability_over_matrices_flat = __flatten_probability_over_matrices(
        probability_over_matrices=probability_over_matrices)

    return local_noise_matrices_flat, probability_over_matrices_flat


def sample_from_CTMP_noise_model_simple_method(global_input_state,
                                               rates_dictionary,
                                               number_of_samples):
    """

    Please note that this method requires setting value of CTMP normalization (gamma parameter from Ref.[#TODO FBM: add ref])
    to sum of error rates, which might be prohibitively big.

    :param global_input_state:
    :type global_input_state:
    :param rates_dictionary:
    :type rates_dictionary:
    :param number_of_samples:
    :type number_of_samples:
    :return:
    :rtype:

    """
    noise_strength_CTMP = ctmp_fun.get_CTMP_error_rates_sum(rates_dictionary=rates_dictionary)
    local_noise_matrices_flat, probability_over_matrices_flat = _get_inputs_to_sampling_algorithm(rates_dictionary)

    counts_dictionary = defaultdict(float)

    for _ in range(number_of_samples):
        random_power = np.random.poisson(noise_strength_CTMP,
                                         size=1)[0]

        output_state_now = sample_from_power_of_B_matrix_simple(global_input_state=global_input_state,
                                                                matrix_power=random_power,
                                                                probability_over_matrices_list=probability_over_matrices_flat,
                                                                local_noise_matrices_list=local_noise_matrices_flat)
        counts_dictionary[output_state_now] += 1

    return counts_dictionary



def estimate_corrected_energy_CTMP_simple(input_state,
                                          rates_dictionary,
                                          number_of_samples_ctmp,
                                          weights_dictionary_tuples):
    noisy_counts_dictionary_sampled_ctmp = sample_from_CTMP_noise_model_simple_method(
        global_input_state=input_state,
        number_of_samples=number_of_samples_ctmp,
        rates_dictionary=rates_dictionary
    )

    energy_modeled_ctmp_sampled = 0
    for output_state, number_of_times in noisy_counts_dictionary_sampled_ctmp.items():
        energy_modeled_ctmp_sampled += number_of_times * quanf.get_energy_from_bitstring_diagonal(
            bitstring=output_state,
            weights_dict=weights_dictionary_tuples)
    energy_modeled_ctmp_sampled /= number_of_samples_ctmp

    return energy_modeled_ctmp_sampled




# np.random.poisson(lambda_par)

#
# number_of_qubits = 8
#
# all_pairs = [(i, j) for i in range(number_of_qubits) for j in
#              range(i + 1, number_of_qubits)]
# single_qubits = [(qi,) for qi in range(number_of_qubits)]
# # all_pairs = all_pairs+[(i,) for i in range(number_of_qubits)]
# rates_2q = {subset: {x: np.random.uniform(0.01, 0.05) for x in ['00', '01', '10', '11']} for subset in
#             all_pairs}
# rates_1q = {subset: {x: np.random.uniform(0.01, 0.05) for x in ['0', '1']} for subset in single_qubits}
#
# rates_dictionary = {**rates_1q, **rates_2q}
# # print(rates_dictionary)
#
# # normalization_rates = ctmp_fun.get_CTMP_error_rates_sum(rates_dictionary=rates_dictionary)
#
# global_generator_matrix = ctmp_fun.get_global_generator_from_rates(rates_dictionary=rates_dictionary,
#                                                           number_of_qubits=number_of_qubits)
#
# # qprint_array(global_generator_matrix)
# global_noise_matrix = sc.linalg.expm(global_generator_matrix)
#
# classical_register = anf.get_classical_register_bitstrings(range(number_of_qubits))
#
# input_state = classical_register[-1]
#
# noisy_distro_exact = global_noise_matrix[:, int(input_state, 2)]
#
# number_of_samples = 10 ** 5
#
# # number_of_samples = int(np.ceil(number_of_samples_init*np.exp(normalization_rates*2)))
#
# # generators_dictionary = ctmp_fun.construct_model_from_rates(rates_dictionary=rates_dictionary)
#
# noisy_counts_dictionary_sampled = sample_from_CTMP_noise_model_simple_method(global_input_state=input_state,
#                                                                             rates_dictionary=rates_dictionary,
#                                                                             number_of_samples=number_of_samples)
# noisy_distro_sampled = fda.convert_counts_dictionary_to_probability_distribution(noisy_counts_dictionary_sampled)
#
# # from povms_qi import povmtools
# # povmtools.prob
#
# # print(noisy_distro_exact)
# # print(noisy_distro_sampled)
# from qrem.functions_qrem import povmtools
# print(povmtools.calculate_total_variation_distance(noisy_distro_exact,noisy_distro_sampled))
# # print(noisy_distro_sampled)
