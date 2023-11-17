import copy
import time
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union

import scipy as sc
import numpy as np
from tqdm import tqdm

from qrem.functions_qrem import povmtools
from qrem.functions_qrem import ancillary_functions as anf
from qrem.noise_simulation.CN import functions_sampling as fus

import re

from qrem.common import convert
from qrem.common import utils


__parities__dict = {1: {('0',): 1,
                        ('1',): -1},

                    2: {('0', '0'): 1,
                        ('1', '1'): 1,
                        ('0', '1'): -1,
                        ('1', '0'): -1,

                        }

                    }

_parities__dict2 = {'0': 1,
                    '1': -1,
                    '00': 1,
                    '11': 1,
                    '01': -1,
                    '10': -1,

                    }

#different normalisation
_parities__dict3 = {'0': 0,
                    '1': 1,
                    '00': 0,
                    '11': 0,
                    '01': 1,
                    '10': 1,

                    }



#ORGANIZE - this function computes energy of some local hamiltonians form marginals, this is used in benchmark functions probably (MO)
def estimate_energy_from_marginals(weights_dictionary: Dict[Tuple[int], float],
                                   marginals_dictionary: Dict[Tuple[int], np.ndarray],
                                   additional_multipliers=None):
    """
    Compute energy of Hamiltonian from dictionary of marginal distributions.

    :param weights_dictionary:
    :param marginals_dictionary:
    :return:
    """

    energy = 0
    for key_local_term in weights_dictionary.keys():
        weight = weights_dictionary[key_local_term]
        marginal = marginals_dictionary[key_local_term]

        # print(marginal,len(marginal))
        qubits_number = int(np.log2(len(marginal)))

        local_term_energy = 0
        for result_index in range(len(marginal)):
            bitstring = list(convert.integer_to_bitstring(result_index, qubits_number))
            bit_true = [int(x) for x in bitstring]
            parity = (-1) ** (np.count_nonzero(bit_true))
            #
            # bitstring_getitem = convert.integer_to_bitstring(result_index, qubits_number).__getitem__
            # parity = __get_part_bitstring_parity_special(bitstring_getitem=bitstring_getitem,
            #                                              qubit_indices=range(qubits_number))
            #
            # print(bit_true,parity, weight, marginal[result_index])

            local_term_energy += weight * marginal[result_index] * parity
            # print(local_term_energy)
        if additional_multipliers is not None:
            local_term_energy *= additional_multipliers[key_local_term]

        energy += local_term_energy

    if isinstance(energy, list) or isinstance(energy, np.ndarray):
        return energy[0]
    else:
        return energy

def __get_part_bitstring_parity_special(bitstring_getitem,
                                        qubit_indices):
    """Used only in get_energy_from_bitstring_diagonal_local()"""
    return (-1) ** list(map(bitstring_getitem, qubit_indices)).count('1')

"""
def __get_part_bitstring_parity_special2(bitstring_getitem,
                                         qubit_indices):
    return __parities__dict[len(qubit_indices)][tuple(map(bitstring_getitem,
                                                    qubit_indices))]

def __get_part_bitstring_parity_special3(bitstring,
                                         qubit_indices):
    return __parities__dict[tuple([bitstring[qi] for qi in qubit_indices])]
"""

def __get_part_bitstring_parity_special4(bitstring,
                                         qubit_indices):
    """Used only in estimate_energy_from_counts_dictionary_alternative()"""
    return _parities__dict2[''.join(([bitstring[qi] for qi in qubit_indices]))]


#(PP) Hamiltonian w stylu Isinga
def get_energy_from_bitstring_diagonal_local(bitstring: str,
                                             weights_dictionary: Dict[Tuple[int], float],
                                             additional_multipliers=None
                                             ):
    """Calculates the energy corresponding to a state (encoded in bistring) and a Hamiltonian
    (encoded in weights and additional_multipliers)"""
    if isinstance(bitstring,list):
        bitstring = ''.join(bitstring)

    bitstring_getitem = bitstring.__getitem__

    parities = {qubit_indices: __get_part_bitstring_parity_special(bitstring_getitem,
                                                                   qubit_indices) for qubit_indices in
                weights_dictionary.keys()}

    energy = 0
    for qubit_indices, hamiltonian_coefficient in weights_dictionary.items():
        # marginal_bitstring = [int(bitstring[q]) for q in qubit_indices]
        # parity = (-1) ** (np.count_nonzero(marginal_bitstring))
        local_energy = parities[qubit_indices] * hamiltonian_coefficient
        if additional_multipliers is not None:
            local_energy *= additional_multipliers[qubit_indices]
        energy += local_energy

    return energy

#(PP) Hamiltonian w stylu Isinga
"""
def get_energy_from_bitstring_diagonal_global(bitstring: str,
                                             weights_dictionary: Dict[Tuple[int], float],
                                             additional_multipliers=None
                                             ):

    energy = 0
    for qubit_indices, hamiltonian_coefficient in weights_dictionary.items():
        marginal_bitstring = [int(bitstring[q]) for q in qubit_indices]
        parity = (-1) ** (np.count_nonzero(marginal_bitstring))
        local_energy = parity * hamiltonian_coefficient
        if additional_multipliers is not None:
            local_energy *= additional_multipliers[qubit_indices]
        energy += local_energy

    return energy
"""

def get_noisy_energy_product_noise_model(input_state,
                                         noise_matrices_dictionary,
                                         weights_dictionary_tuples,
                                         needed_pairs=None):
    pairs_marginals_noisy = fus.calculate_pairs_marginals_from_tensor_model_fixed_input(
        input_state=input_state,
        noise_matrices_dictionary=noise_matrices_dictionary,
        needed_pairs=needed_pairs,
        get_also_1q_marginals=True)

    energy_modeled_exact = estimate_energy_from_marginals(
        weights_dictionary=weights_dictionary_tuples,
        marginals_dictionary=pairs_marginals_noisy)

    return energy_modeled_exact

def estimate_energy_from_counts_dictionary(counts_dictionary: Dict[str, int],
                                           weights_dictionary: Dict[Tuple[int], float],
                                           additional_multipliers=None,
                                           normalize=True):
    """
    Calculates the energy averaged over results, used by the class EnergyEstimatorBase

    :param counts_dictionary:
    :type counts_dictionary:
    :param weights_dictionary:
    :type weights_dictionary:
    :return:
    :rtype:
    """

    energy = 0
    for bitstring, shots in counts_dictionary.items():
        energy += get_energy_from_bitstring_diagonal_local(bitstring=bitstring,
                                                           weights_dictionary=weights_dictionary,
                                                           additional_multipliers=additional_multipliers) * shots

    if normalize:
        normalization = sum(list(counts_dictionary.values()))
        energy /= normalization

    return energy

def estimate_energy_from_counts_dictionary_alternative(counts_dictionary: Dict[str, int],
                                                       weights_dictionary: Dict[Tuple[int], float],
                                                       additional_multipliers=None,
                                                       normalize: Optional[Union[bool, float]] = True,
                                                       print_progress_bar=False):
    """
    Alternative to estimate_energy_from_counts_dictionary()
    used ONLY in noise_mitigation/expected_values/CRS/testing_mitigation_clusters_SCR_amd.py

    :param counts_dictionary:
    :type counts_dictionary:
    :param weights_dictionary:
    :type weights_dictionary:
    :return:
    :rtype:
    """
    t0 = time.time()
    tests = True
    if tests:
        if additional_multipliers is not None:
            energy = 0
            for qubit_indices in tqdm(weights_dictionary.keys()):
                local_energy=0
                for bitstring, shots in counts_dictionary.items():
                    local_energy+= _parities__dict2[''.join([bitstring[qi] for qi in qubit_indices])] * shots

                energy+=local_energy*additional_multipliers[qubit_indices]*weights_dictionary[qubit_indices]
        else:
            energy = 0
            for qubit_indices in tqdm(weights_dictionary.keys(),disable=True):
                local_energy=0
                for bitstring, shots in counts_dictionary.items():
                    local_energy+= _parities__dict2[''.join([bitstring[qi] for qi in qubit_indices])] * shots

                energy += local_energy * weights_dictionary[
                    qubit_indices]

    else:

        parities = {}
        for bitstring in counts_dictionary.keys():
            parities[bitstring] = {
            qubit_indices: __get_part_bitstring_parity_special4(
                bitstring=bitstring,
                qubit_indices=qubit_indices) for
            qubit_indices in weights_dictionary.keys()}

        t1=time.time()
            # for subset in

        energy = 0
        if additional_multipliers is not None:
            for qubit_indices, hamiltonian_coefficient in tqdm(weights_dictionary.items(),
                                                               disable=not print_progress_bar):
                local_energy = 0
                for bitstring, shots in counts_dictionary.items():
                    local_energy += shots * parities[bitstring][qubit_indices]

                # local_energy = parities[qubit_indices][1]-parities[qubit_indices][-1]



                local_energy *= additional_multipliers[qubit_indices]
                local_energy *= hamiltonian_coefficient

                energy += local_energy
        else:
            for qubit_indices, hamiltonian_coefficient in tqdm(weights_dictionary.items(),
                                                               disable=not print_progress_bar):
                local_energy = 0
                for bitstring, shots in counts_dictionary.items():
                    local_energy += shots * parities[bitstring][qubit_indices]
                local_energy *= hamiltonian_coefficient

                energy += local_energy

        t2 = time.time()


    t1=time.time()
    if isinstance(normalize, float):
        energy /= normalize
    elif isinstance(normalize, bool):
        if normalize:
            normalization = sum(list(counts_dictionary.values()))
            energy /= normalization

    t2=time.time()

    # print(t1-t0)
    # print('hej')
    return energy


def merge_multiple_counts_dictionaries(counts_dictionaries_list: List[Dict[str, int]]) -> Dict[
    str, int]:
    """
    Merge multiple counts dictionaries.
    This is useful when you have results of multiple implementations of the same experiment.

    :param counts_dictionaries_list: list of results of counts dictionaries of the form:
                                    {'bitstring":number of measurements}
    :return:
    """

    # first dictionary will be template to which we will add counts
    merged_counts = copy.deepcopy(counts_dictionaries_list[0])

    # we go through all other dictionaries
    for counts_dictionary in counts_dictionaries_list[1:]:
        for bitstring, ticks in counts_dictionary.items():
            if bitstring in merged_counts.keys():
                merged_counts[bitstring] += ticks
            else:
                merged_counts[bitstring] = ticks

    return merged_counts

def reverse_bitstrings_in_counts_dictionary(counts_dictionary):
    """Used only by convert_counts_overlapping_tomography()"""
    counts_reversed = {}

    for bitstring, ticks in counts_dictionary.items():
        counts_reversed[bitstring[::-1]] = ticks
    return counts_reversed

def convert_counts_overlapping_tomography(counts_dictionary: Dict[str, Dict[str, int]],
                                          experiment_name: str,
                                          reverse_bitstrings=True,
                                          old_send_procedure=True):

    # TODO FBM: add support for Quantum Detector Tomography
    # MOcomm - this looks
    """
    This function converts unprocessed dictionary of experiment results, where multiple keys can describe identical
    circuits (eg. "DDOT-010no0", "DDOT-010no1", see description of string_cutter below), to a dictionary where a key
    uniquely corresponds to a circuit and value is the combined counts from all such circuits in the unprocessed
    dictionary. In resulting dictionary outcome bit strings and circuit names are in big-endian order.

    param counts_dictionary: dictionary where the key is circuit name (e.g. "DDOT-010no3", described in inner
                              string_cutter function below) and value is dictionary of counts, where the key is a string
                              denoting classical outcome and the value the number of its occurrences in given experiment.
    param experiment_name: string denoting the type of experiment whose results counts_dictionary contains, e.g. 'QDOT';
                            the valid names are specified in SeparableCircuitsCreator.__valid_experiments_names__
    param reverse_bitstrings: bool; if False, the bitstring denoting classical outcome in counts_dictionary will be
                               interpreted as big-endian (qubit 0 on the left); if True it will be interpreted as
                               little-endian (qubit 0 on the right) and will be reversed to conform to QREM convention.

    :return: big_counts_dictionary: dictionary where key is string describing circuit (e.g. '010' means gates:
                                    iden, X, iden on qubits 0, 1, 2 respectively) and value is
                                    dictionary where key is bitstring describing outcome (e.g. '100') and value is
                                    number of occurences of that outcome in the experiment; these strings are big-endian.
    """
    def string_cutter(circuit_name: str):
        """
        This inner function cuts the name of the circuit to the format that will later be used by
        tomography data analyzers.


        param circuit_name:
        It assumes the following convention:

        circuit_name = "experiment name" + "-" + "circuit label"+
        "no"+ "integer identifier for multiple implementations of the same circuit"

        for example the circuit can have name:
        "DDOT-010no3"

        which means that this experiment is Diagonal Detector Overlapping Tomography (DDOT),
        the circuit implements state "010" (i.e., gates iden, X, iden on qubits 0,1,2 - big-endian order), and
        in the whole circuits sets this is the 4th (we start counting from 0) circuit that implements
        that particular state.

        :return: big_counts_dictionary
        """
        from qrem.noise_characterization.tomography_design.overlapping import SeparableCircuitsCreator

        if experiment_name.lower() not in SeparableCircuitsCreator.__valid_experiments_names__:
            raise ValueError(f"ONLY the following experiments are supported:\n{SeparableCircuitsCreator.__valid_experiments_names__}")

        experiment_string_len = len(list(experiment_name))
        full_name_now = circuit_name[experiment_string_len + 1:]
        new_string = ''
        for symbol_index in range(len(full_name_now)):
            if full_name_now[symbol_index] + full_name_now[symbol_index + 1] == 'no':
                break
            new_string += full_name_now[symbol_index]
        return new_string

    big_counts_dictionary = {}

    for circuit_name, counts_dict_now in counts_dictionary.items():
        
        #the line below was used with data obtained with the old sending routine  
        
        if old_send_procedure:
            proper_name_now = string_cutter(circuit_name)
        else: 
            m=re.search(r'\d+', circuit_name)
            proper_name_now=m.group()
        if reverse_bitstrings:
            counts_dict_now = reverse_bitstrings_in_counts_dictionary(counts_dict_now)

        if proper_name_now not in big_counts_dictionary.keys():
            big_counts_dictionary[proper_name_now] = counts_dict_now
        else:
            big_counts_dictionary[proper_name_now] = merge_multiple_counts_dictionaries(
                [big_counts_dictionary[proper_name_now], counts_dict_now])

    return big_counts_dictionary


def convert_subset_counts_dictionary_to_probability_distribution(counts_dictionary):
    """Makes a probability distribution from counts dictionary,
    IMPORTANT: fills in with zeros for results absent in counts_dictionary
    USED A LOT by functions from cn\mitigation.py and functions_qrem\new_mitigation_routines.py"""
    key = next(iter(counts_dictionary))
    subset_counts_dictionary = counts_dictionary[key]
    subset = next(iter(subset_counts_dictionary))
    number_of_qubits = len(subset) 
    probability_distribution = np.zeros((2 ** number_of_qubits), dtype=float)
    
    normalization_shots = sum(list(subset_counts_dictionary[subset]))
    probability_distribution= counts_dictionary[key][subset] / normalization_shots



    return  probability_distribution


def get_marginal_from_probability_distribution(
        global_probability_distribution: np.ndarray,
        bits_of_interest: List[int],
        register_names: Optional[List[str]] = None) -> np.ndarray:
    """Return marginal distribution from vector of global distribution
    :param global_probability_distribution: distribution on all bits
    :param bits_of_interest: bits we are interested in (so we average over other bits)
                            Assuming that qubits are labeled
                            from 0 to log2(len(global_probability_distribution))
    :param register_names: bitstrings register, default is
                           '00...00', '000...01', '000...10', ..., etc.

    :return: marginal_distribution : marginal probability distribution

    NOTE: we identify bits with qubits in the variables bitstring_names

    #TODO FBM: do some speed tests on some details of those solutions
    """

    if len(bits_of_interest) == 0:
        print('0 length bits list')
        return global_probability_distribution

    try:
        if isinstance(global_probability_distribution[0][0], complex) or isinstance(
                global_probability_distribution[0][0], np.complex128):
            global_probability_distribution = global_probability_distribution.real
    except(IndexError):
        if isinstance(global_probability_distribution[0], complex) or isinstance(
                global_probability_distribution[0], np.complex128):
            global_probability_distribution = global_probability_distribution.real

    global_dimension = len(global_probability_distribution)
    global_number_of_qubits = int(np.log2(global_dimension))
    all_qubits = list(range(global_number_of_qubits))
    bits_to_average_over = list(set(all_qubits).difference(set(bits_of_interest)))

    number_of_bits_in_marginal = global_number_of_qubits - len(bits_to_average_over)
    dimension_of_marginal = 2 ** number_of_bits_in_marginal

    if register_names is None:
        bitstring_names = anf.get_classical_register_bitstrings(list(range(global_number_of_qubits)),
                                                                global_number_of_qubits)
    else:
        bitstring_names = register_names

    marginal_distribution = np.zeros((dimension_of_marginal, 1), dtype=float)
    for j in range(global_dimension):
        # this is slightly faster than going through "for bitstring_global in bitstring_names
        # and then converting bitstring_global to integer
        # and also faster than creating the global bitstring in situ
        bitstring_global = bitstring_names[j]

        bitstring_local = ''.join(
            [bitstring_global[qubit_index] for qubit_index in bits_of_interest])

        marginal_distribution[int(bitstring_local, 2)] += global_probability_distribution[j]

    return marginal_distribution


def _sort_clusters_division(clusters_division: List[Tuple[int]]):
    sorted_inside = [tuple(sorted(x)) for x in clusters_division]

    return tuple(sorted(sorted_inside, key=lambda x: x[0]))


def get_noise_matrices_from_POVMs_dictionary(POVMs_dictionary):
    """Takes POVM dictionary and returns noise matrices,
    IMPORTANT: used by characterization\characterization_routine.py"""
    noise_matrices_dictionary = {}

    for qubits_subset, POVM in tqdm(POVMs_dictionary.items()):
        noise_matrices_dictionary[qubits_subset] = povmtools.get_noise_matrix_from_povm(povm=POVM)

    return noise_matrices_dictionary

def get_multiple_mitigation_strategies_clusters_for_pairs_of_qubits(pairs_of_qubits,
                                                           clusters_sets,
                                                     dictionary_results,
                                                     #marginals_dictionary,
                                                     noise_matrices_dictionary,
                                                    show_progress_bar = True
                                                     ):
    from qrem.noise_mitigation.probability_distributions.CorrectionDataGenerator import \
        CorrectionDataGenerator

    number_of_qubits = sum([len(x) for x in list(clusters_sets.keys())[0]])

    joint_correction_matrices_dictionary = {}
    correction_indices = {}
    mitigation_data ={}
    test_check={}
    for clusters_list, model_parameters in tqdm(clusters_sets.items(), disable = not show_progress_bar):
        neighborhoods = {cluster: [] for cluster in clusters_list}

        correction_data_generator = CorrectionDataGenerator(results_dictionary_ddt=dictionary_results,
                                                            #marginals_dictionary=marginals_dictionary,
                                                            number_of_qubits=number_of_qubits,
                                                            clusters_list=clusters_list,
                                                            neighborhoods=neighborhoods,
                                                            noise_matrices_dictionary=
                                                            noise_matrices_dictionary,
                                                            #TJT: his is probably not a good idea
                                                            #correction_matrices_dictionary=joint_correction_matrices_dictionary# )
                                                            correction_matrices_dictionary={})

        correction_data_generator.get_pairs_correction_data(pairs_list=pairs_of_qubits,
                                                            show_progress_bar=False)
        
        
        cluster_check={}
        for subset in  correction_data_generator._correction_matrices.keys():
            cluster_check[subset]=(correction_data_generator._noise_matrices_dictionary[subset]['averaged'].dot(correction_data_generator._correction_matrices[subset]))
        test_check[clusters_list] = cluster_check
            
                
         

        #JT: Here we use a private property _correction_matrices, _correction_indices of CorrectionDataGenerator class, thsi needs to be fixed  
        c_list = [correction_data_generator._correction_indices, correction_data_generator._correction_matrices]
        c_tuple=tuple(c_list)
        mitigation_data[clusters_list] = c_tuple
        
        
        joint_correction_matrices_dictionary = {**joint_correction_matrices_dictionary,
                                                **correction_data_generator._correction_matrices}

        correction_indices_now = correction_data_generator._correction_indices

        correction_indices[clusters_list] = correction_indices_now


    return joint_correction_matrices_dictionary, correction_indices, mitigation_data


# """
# def get_CTMP_rates_from_results(results_dictionary_ddot: Dict[str, Dict[str, int]],
#                                 number_of_qubits: int):
#     single_qubits = list(range(number_of_qubits))
#     pairs = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]

#     local_noise_matrices_CTMP = {pair: np.zeros((4, 4), dtype=float) for pair in pairs}

#     for pair in pairs:
#         pair_complement = list(set(single_qubits).difference(set(pair)))

#         for global_input_state, results_dictionary in results_dictionary_ddot.items():
#             marginal_input_state = ''.join([global_input_state[x] for x in pair])
#             input_state_complement = [global_input_state[x] for x in pair_complement]

#             for global_output_state, ticks in results_dictionary.items():
#                 marginal_output_state = ''.join([global_output_state[x] for x in pair])
#                 output_state_complement = [global_output_state[x] for x in pair_complement]

#                 if output_state_complement == input_state_complement:
#                     # check if this is their convention!
#                     local_noise_matrices_CTMP[pair][int(marginal_output_state, 2),
#                                                     int(marginal_input_state, 2)] += ticks

#     # normalize to stochastic matrices
#     for pair in pairs:
#         for k in range(4):
#             local_noise_matrices_CTMP[pair][:, k] /= sum(local_noise_matrices_CTMP[pair][:, k])

#     # Get G matrices
#     G_matrices = {pair: sc.linalg.logm(local_noise_matrices_CTMP[pair]) for pair in pairs}

#     # ancillary function
#     def _chop_negatives(M):
#         (m, n) = M.shape

#         chopped_M = copy.deepcopy(M)
#         for i in range(m):
#             for j in range(n):
#                 if i != j and M[i, j] < 0:
#                     chopped_M[i, j] = 0

#         return chopped_M

#     # Get G' matrices
#     G_prime_matrices = {pair: _chop_negatives(G_matrices[pair]) for pair in pairs}

#     rates_dictionary_1q = {f"q{q}": {'0': 0, '1': 0} for q in single_qubits}
#     for qj in single_qubits:
#         r0, r1 = 0, 0

#         for q_other in single_qubits:
#             if q_other != qj:
#                 G_prime_matrix_now = G_prime_matrices[tuple(sorted([qj, q_other]))]

#                 r0 += G_prime_matrix_now[2, 0] + G_prime_matrix_now[3, 1] \
#                     # +G_prime_matrix_now[1,0]+G_prime_matrix_now[3,2]

#                 r1 += G_prime_matrix_now[0, 2] + G_prime_matrix_now[1, 3] \
#                     # +G_prime_matrix_now[0, 1] + G_prime_matrix_now[2, 3]

#         r0 /= 2 * (number_of_qubits - 1)
#         r1 /= 2 * (number_of_qubits - 1)

#         rates_dictionary_1q[f"q{qj}"]['0'] = r0
#         rates_dictionary_1q[f"q{qj}"]['0'] = r1

#     rates_dictionary_2q = {f"q{pair[0]}q{pair[1]}": {'00': 0,
#                                                      '01': 0,
#                                                      '10': 0,
#                                                      '11': 0} for pair in pairs}

#     for pair in pairs:
#         G_prime_matrix_now = G_prime_matrices[pair]
#         rates_dictionary_2q[f"q{pair[0]}q{pair[1]}"]['00'] = G_prime_matrix_now[3, 0]
#         rates_dictionary_2q[f"q{pair[0]}q{pair[1]}"]['01'] = G_prime_matrix_now[2, 1]
#         rates_dictionary_2q[f"q{pair[0]}q{pair[1]}"]['10'] = G_prime_matrix_now[1, 2]
#         rates_dictionary_2q[f"q{pair[0]}q{pair[1]}"]['11'] = G_prime_matrix_now[0, 3]

#     rates_dictionary = {**rates_dictionary_1q,
#                         **rates_dictionary_2q}

#     return rates_dictionary


# def convert_counts_dictionary_to_probability_distribution(counts_dictionary):
#     number_of_qubits = len(list(list(counts_dictionary.keys())[0]))
#     probability_distribution = np.zeros((2 ** number_of_qubits), dtype=float)
#     normalization_shots = sum(list(counts_dictionary.values()))

#     for key, shots in counts_dictionary.items():
#         probability_distribution[int(key, 2)] += shots / normalization_shots
#     return probability_distribution

# """

# """
# def get_qubits_to_clusters_map(clusters_division):
#     qubits_indices = []

#     for cluster in clusters_division:
#         qubits_indices += list(cluster)

#     qubits_to_clusters_map = {}
#     for qubit in qubits_indices:
#         for cluster in clusters_division:
#             if qubit in cluster:
#                 qubits_to_clusters_map[qubit] = cluster
#     return qubits_to_clusters_map
#     # #MOVE_TO >> core.quantum or to characterisation
#     # #DUPLICATE in dunctions data analysis.py
#     # def get_qubits_to_clusters_map(clusters_list,
#     #                             number_of_qubits):
#     #     qubits_to_clusters_map = {}
#     #     for qubit in range(number_of_qubits):
#     #         for cluster in clusters_list:
#     #             if qubit in cluster:
#     #                 qubits_to_clusters_map[qubit] = cluster

#     #     return qubits_to_clusters_map



# def get_pairs_complements_clusters(clusters_list):
#     pairs_complements_dictionary = {}
#     for cluster in clusters_list:
#         pairs_in_cluster = [(qi, qj) for qi in cluster for qj in cluster if qj > qi]

#         for pair in pairs_in_cluster:
#             pairs_complements_dictionary[pair] = utils.lists_difference(cluster, pair)
#     return pairs_complements_dictionary



# def divide_counts_dictionary_into_batches(counts_dictionary:Dict[str,int],
#                                           number_of_batches:int):
#     if number_of_batches ==1 or number_of_batches is None or number_of_batches==0:
#         return [counts_dictionary]

#     list_of_counts = []
#     for output_bitstring, number_of_ticks in counts_dictionary.items():
#         list_of_counts = list_of_counts + [output_bitstring] * number_of_ticks

#     rng = np.random.default_rng()
#     rng.shuffle(list_of_counts)
#     rng.shuffle(list_of_counts)
#     rng.shuffle(list_of_counts)
#     rng.shuffle(list_of_counts)
#     rng.shuffle(list_of_counts)

#     batch_size = len(list_of_counts) // number_of_batches
#     batches = [list_of_counts[index_batch * batch_size:(index_batch + 1) * batch_size]
#                for index_batch in range(number_of_batches)]

#     batched_dictionaries = [dict(Counter(batch_now)) for batch_now in batches]

#     return batched_dictionaries

# def divide_dot_results_dictionary_into_batches(dot_results:Dict[str,Dict[str,int]],
#                                                number_of_batches:int):
#     batched_results = [{} for _ in range(number_of_batches)]

#     for input_state, counts_dictionary in tqdm(dot_results.items()):
#         batched_counts = divide_counts_dictionary_into_batches(counts_dictionary=counts_dictionary,
#                                                                    number_of_batches=number_of_batches)

#         for batch_index in range(len(batched_counts)):
#             batched_results[batch_index][input_state] = batched_counts[batch_index]

#     return batched_results

# def version_control_results_dot(dictionary_data_counts:Dict[str,dict]):

#     try:
#         # Old convention, shouldn't be used anymore
#         dictionary_results = dictionary_data_counts['results_dictionary_preprocessed']

#     except(KeyError):

#         try:
#             dictionary_results = dictionary_data_counts['results_dictionary_preprocessed']
#         except(KeyError):
#             try:
#                 dictionary_results_DDOT = dictionary_data_counts['DDOT_results']

#             except(KeyError):
#                 dictionary_results_DDOT = {}



#             try:
#                 dictionary_results_QDOT = dictionary_data_counts['QDOT_results']
#             except(KeyError):
#                 dictionary_results_QDOT = {}

#             dictionary_results = {**dictionary_results_QDOT, **dictionary_results_DDOT}

#     return dictionary_results
# """
# class KeyDependentDictForMarginals(defaultdict):
#     # TODO FBM: This is too slow for big systems, remove usage of this from repository
#     """
#     This is class used to store marginal probability distributions in dictionary.
#     It is "key dependent" in the sense that if user tries to refer to non-existing value for some
#     KEY, then this value is created as a marginal distribution which size depends on the KEY
#     NOTE: We assume that provided KEY is a string denoting  qubits subset
#     (see self.value_creating_function)


#     COPYRIGHT NOTE
#     The main idea of this code was taken from Reddit thread:
#     https://www.reddit.com/r/Python/comments/27crqg/making_defaultdict_create_defaults_that_are_a/

#     """

#     def __init__(self):
#         super().__init__(None)  # initialize as standard defaultdict

#         # This is the function which takes the string "key" that is assumed to label qubits subset
#         # in the form 'q2q3q11...' etc. It takes this key, calculates number of qubits N, and creates
#         # empty vector of the size d=2^N.
#         self.value_creating_function = lambda key: np.zeros(
#             (int(2 ** len(key)), 1),
#             dtype=float)

#     # called when key is missing
#     def __missing__(self, key):
#         # calculate the key-dependent value
#         ret = self.value_creating_function(key)
#         # put the value inside the dictionary
#         self[key] = ret
#         return ret


# class key_dependent_dict_for_marginals(defaultdict):
#     """
#     same as KeyDependentDictForMarginals but different name
#     TODO FBM: refactor this
#     """

#     def __init__(self):
#         super().__init__(None)  # initialize as standard defaultdict

#         # This is the function which takes the string "key" that is assumed to label qubits subset
#         # in the form 'q2q3q11...' etc. It takes this key, calculates number of qubits N, and creates
#         # empty vector of the size d=2^N.
#         self.value_creating_function = lambda key: np.zeros(
#             (int(2 ** len(convert.get_qubit_indices_from_keystring(key))), 1),
#             dtype=float)

#     # called when key is missing
#     def __missing__(self, key):
#         # calculate the key-dependent value
#         ret = self.value_creating_function(key)
#         # put the value inside the dictionary
#         self[key] = ret
#         return ret


# """
# def get_state_from_circuit_name(circuit_name):
#     state_name = ''
#     for string in circuit_name:
#         if string in ['1', 'X', 'x']:
#             state_name += '1'
#         elif string in ['0', 'I', 'i_index', 'id', 'Id']:
#             state_name += '0'

#     return state_name


# def get_mini_dict(number_of_qubits):
#     register = povmtools.register_names_qubits(range(number_of_qubits), number_of_qubits)
#     return {key: np.zeros((int(2 ** number_of_qubits), 1)) for key in register}
# """
