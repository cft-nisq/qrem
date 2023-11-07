
from typing import Tuple, Dict, List, Union, Optional

import numpy as np

from qrem.functions_qrem import ancillary_functions as anf, functions_data_analysis as fda
from qrem.functions_qrem import quantum_ancillary_functions as qanf 
from qrem.noise_characterization.base_classes.marginals_analyzer_base import MarginalsAnalyzerBase
from qrem.noise_model_generation.CN.GlobalNoiseMatrixCreator import GlobalNoiseMatrixCreator

from qrem.common import convert, utils

#TODO FBM: Clean up this file


# {cluster_tuple:{'averaged'}}









##############################################################
##################### Noise model sampling##########
##############################################################

def probabilistic_bitflip_choices(bit: str,
                                  bitflip_probability: float):
    # this is faster than using np.random.binomial
    import random
    return random.choices([convert.negate_bitstring(bit), bit],
                          weights=[bitflip_probability, 1 - bitflip_probability])[0]


def stochastic_relabeling(bitstring: str,
                          stochastic_matrices: Dict[str, np.ndarray],
                          reverse_bitstring=False):
    number_of_qubits = len(list(bitstring))

    flipped_bitstring = ''
    #
    if reverse_bitstring:
        bitstring = bitstring[::-1]

    for qubit_index in range(number_of_qubits):
        bit = bitstring[qubit_index]
        stochastic_map = stochastic_matrices['q%s' % qubit_index]
        bitflip_probability = stochastic_map[int(convert.negate_bitstring(bit), 2), int(bit, 2)]
        flipped_bit = probabilistic_bitflip_choices(bit, bitflip_probability)
        flipped_bitstring += flipped_bit

    return flipped_bitstring



def sample_from_noise_model(input_state: List[str],
                            number_of_qubits: int,
                            number_of_samples: int,
                            noise_model_description: Dict[Tuple[int], Dict[str, np.ndarray]],
                            neighborhoods_tuples: Dict[Tuple[int], List[int]]
                            ):
    noisy_bitstrings = [np.zeros((number_of_qubits), dtype=str) for _ in range(number_of_samples)]
    for cluster_qubits, noise_matrices_on_cluster in noise_model_description.items():
        # print(neighborhoods_tuples[cluster_qubits])
        # print(cluster_qubits)
        # print(cluster_qubits)
        input_on_cluster = ''.join([input_state[x] for x in cluster_qubits])

        neighbors = neighborhoods_tuples[cluster_qubits]

        if len(neighbors) == 0:
            noise_matrix = noise_matrices_on_cluster['averaged']
        else:
            input_on_neighbors = ''.join(
                [input_state[x] for x in neighbors])

            noise_matrix = noise_matrices_on_cluster[neighbors][input_on_neighbors]

        prob_distro_now = noise_matrix[:, int(input_on_cluster, 2)]
        cluster_qubits_enumerated = dict(enumerate(cluster_qubits))
        # print(noise_matrix)
        for sample_index in range(number_of_samples):
            resampling = np.random.multinomial(n=1, pvals=prob_distro_now)
            new_bitstring_marginal = list(convert.integer_to_bitstring(integer=np.argmax(resampling),
                                                                   number_of_bits=len(cluster_qubits)))

            for ordered_index, true_index in cluster_qubits_enumerated.items():
                noisy_bitstrings[sample_index][true_index] = new_bitstring_marginal[ordered_index]

    return [list(bitstring) for bitstring in noisy_bitstrings]





















##############################################################
##################### Model simulation##########
##############################################################


def construct_product_noise_model_in_correct_order(
        noise_matrices_dictionary: Dict[Tuple[int], np.ndarray],
        qubits_mapping_dictionary: Optional[Dict[int, int]] = None):
    all_clusters = list(noise_matrices_dictionary.keys())

    # print(all_clusters)
    # print([list(x) for x in all_clusters])
    number_of_qubits = len(utils.lists_sum_multiple([list(x) for x in all_clusters]))

    if qubits_mapping_dictionary is None:
        def qubits_map(x: int):
            return x
    else:
        def qubits_map(x: int):
            return qubits_mapping_dictionary[x]

    global_noise_matrix = np.eye(2 ** number_of_qubits, dtype=float)

    for cluster, noise_matrix in noise_matrices_dictionary.items():
        embeded_noise_matrix = qanf.embed_operator_in_bigger_hilbert_space(
            number_of_qubits=number_of_qubits,
            local_operator=noise_matrix,
            global_indices=[qubits_map(qi) for qi in cluster]).real

        global_noise_matrix = global_noise_matrix @ embeded_noise_matrix

    return global_noise_matrix



def get_random_clusters_division(number_of_qubits: int,
                                 minimal_cluster_size: int,
                                 maximal_cluster_size: int):
    clusters_list = []
    assigned_qubits = []
    while sorted(assigned_qubits) != list(range(number_of_qubits)):

        number_of_qubits_left = number_of_qubits - len(assigned_qubits)

        if number_of_qubits_left < maximal_cluster_size:
            maximal_cluster_size = number_of_qubits_left

        cluster_size = np.random.randint(minimal_cluster_size, maximal_cluster_size + 1)
        cluster_now = []

        while len(cluster_now) != cluster_size:
            random_qubit = np.random.randint(0, number_of_qubits)

            if random_qubit not in assigned_qubits:
                cluster_now.append(random_qubit)
                assigned_qubits.append(random_qubit)

        clusters_list.append(tuple(sorted(cluster_now)))

    return clusters_list





def get_noisy_energy_exact_CN(input_state,
                              noise_matrices_dictionary_tuples,
                              neighborhoods_dictionary_tuples,
                              weights_dictionary_tuples
                              ):
    noise_model_now = get_model_for_fixed_input_state(input_state=input_state,
                                                      noise_model_description=noise_matrices_dictionary_tuples,
                                                      neighborhoods_dictionary=neighborhoods_dictionary_tuples)

    pairs_marginals_noisy = calculate_pairs_marginals_from_tensor_model_fixed_input(
        input_state=input_state,
        noise_matrices_dictionary=noise_model_now,
        get_also_1q_marginals=True)

    energy_modeled_exact = fda.estimate_energy_from_marginals(
        weights_dictionary=weights_dictionary_tuples,
        marginals_dictionary=pairs_marginals_noisy)

    return energy_modeled_exact


##############################################################
##################### Noise model reconstruction ##########
##############################################################

def get_noisy_energy_product_noise_model(input_state,
                                         noise_matrices_dictionary,
                                         weights_dictionary_tuples,
                                         needed_pairs=None):
    pairs_marginals_noisy = calculate_pairs_marginals_from_tensor_model_fixed_input(
        input_state=input_state,
        noise_matrices_dictionary=noise_matrices_dictionary,
        needed_pairs=needed_pairs,
        get_also_1q_marginals=True)

    energy_modeled_exact = fda.estimate_energy_from_marginals(
        weights_dictionary=weights_dictionary_tuples,
        marginals_dictionary=pairs_marginals_noisy)

    return energy_modeled_exact



#JT: This is a function used when neighbors are used (it specifies noise matrix) 

def get_model_for_fixed_input_state(input_state: List[str],
                                    noise_model_description: Dict[Tuple[int],
                                                                  Dict[Union[Tuple[int], str],
                                                                       Union[np.ndarray, Dict[
                                                                           str, np.ndarray]]
                                                                  ]
                                    ],
                                    neighborhoods_dictionary: Dict[Tuple[int],
                                                                   Tuple[int]]):
    noise_matrices_now = {}
    for cluster_qubits, noise_matrices_on_cluster in noise_model_description.items():
        neighbors_now = neighborhoods_dictionary[cluster_qubits]

        # print(cluster_qubits,neighbors_now)
        #
        # print(noise_matrices_on_cluster)

        if len(neighbors_now) == 0:
            noise_matrix = noise_matrices_on_cluster['averaged']
        else:
            input_on_neighbors = ''.join([input_state[qi] for qi in neighbors_now])
            noise_matrix = noise_matrices_on_cluster[neighbors_now][input_on_neighbors]

        noise_matrices_now[cluster_qubits] = noise_matrix

    return noise_matrices_now


##############################################################
##################### Functions computing marginals ##########
##############################################################


def get_noisy_marginals_within_cluster(input_state: str,
                                       cluster_noise_matrix: np.ndarray,
                                       subsets_of_interest: List[Union[Tuple[int], List[int]]],
                                       qubits_mapping: Dict[int, int]) -> Dict[Tuple[int], np.ndarray]:
    input_state_binary = int(input_state, 2)

    probability_distribution = cluster_noise_matrix[:, input_state_binary]

    marginals_now = {}
    for qubits_of_interest in subsets_of_interest:
        marginal = fda.get_marginal_from_probability_distribution(
            global_probability_distribution=probability_distribution,
            bits_of_interest=[qubits_mapping[x] for x in qubits_of_interest])
        marginals_now[tuple(qubits_of_interest)] = marginal

    return marginals_now


def calculate_1q_marginals_from_tensor_model(input_state: str,
                                             noise_matrices_dictionary: Dict[Tuple[int],
                                                                             np.ndarray]):
    marginals_1q = {}
    for cluster, noise_matrix in noise_matrices_dictionary.items():
        cluster_input_state = ''.join([input_state[x] for x in cluster])
        enumerated_qubits = utils.map_index_to_order(list(cluster))
        marginals_now = get_noisy_marginals_within_cluster(input_state=cluster_input_state,
                                                           cluster_noise_matrix=noise_matrix,
                                                           subsets_of_interest=[[qi] for qi in
                                                                                cluster],
                                                           qubits_mapping=enumerated_qubits)

        marginals_1q = {**marginals_1q, **marginals_now}

    return marginals_1q



def calculate_pairs_marginals_from_tensor_model_all_inputs(
        noise_matrices_dictionary: Dict[Tuple[int],
                                        np.ndarray],
        needed_pairs,
        get_also_1q_marginals=True
):
    # TODO FBM: generalize to higher locality

    clusters_list = list(noise_matrices_dictionary.keys())
    number_of_qubits = sum([len(x) for x in clusters_list])

    qubits_to_clusters_map = {}
    for qubit in range(number_of_qubits):
        for cluster in clusters_list:
            if qubit in cluster:
                qubits_to_clusters_map[qubit] = cluster

    pairs_marginals = {pair: {} for pair in needed_pairs}
    if get_also_1q_marginals:
        qubits_list = utils.lists_sum_multiple(clusters_list)
        pairs_marginals = {**pairs_marginals,
                           **{(q0,): {} for q0 in qubits_list}}

    already_done = []
    for cluster, noise_matrix in noise_matrices_dictionary.items():
        pairs_in_cluster = [(qi, qj) for qi in cluster for qj in cluster
                            if qj > qi and (qi, qj) in needed_pairs]

        local_classical_register = anf.get_classical_register_bitstrings(
            qubit_indices=range(len(cluster)))
        enumerated_qubits = utils.map_index_to_order(list(cluster))
        for cluster_input_state in local_classical_register:

            marginals_dict_now = get_noisy_marginals_within_cluster(input_state=cluster_input_state,
                                                                    cluster_noise_matrix=noise_matrix,
                                                                    subsets_of_interest=pairs_in_cluster,
                                                                    qubits_mapping=enumerated_qubits)

            for pair_now, marginal_now in marginals_dict_now.items():
                pairs_marginals[pair_now][cluster_input_state] = marginal_now

            if get_also_1q_marginals:
                marginals_dict_now1q = get_noisy_marginals_within_cluster(
                    input_state=cluster_input_state,
                    cluster_noise_matrix=noise_matrix,
                    subsets_of_interest=[[qi] for qi in cluster],
                    qubits_mapping=enumerated_qubits)
                for qubit_now, marginal1q in marginals_dict_now1q.items():
                    pairs_marginals[qubit_now][cluster_input_state] = marginal1q

        # raise KeyboardInterrupt
        already_done = already_done + pairs_in_cluster

    for (q0, q1) in utils.lists_difference(needed_pairs, already_done):

        cl0, cl1 = qubits_to_clusters_map[q0], qubits_to_clusters_map[q1]

        enumerated_qubits = utils.map_index_to_order(list(cl0) + list(cl1))

        local_classical_register = anf.get_classical_register_bitstrings(
            qubit_indices=range(len(cl0 + cl1)))

        for input_bitstring in local_classical_register:
            in_cl0 = ''.join([input_bitstring[enumerated_qubits[x]] for x in cl0])
            in_cl1 = ''.join([input_bitstring[enumerated_qubits[x]] for x in cl1])
            marginal_0 = pairs_marginals[(q0,)][in_cl0]
            marginal_1 = pairs_marginals[(q1,)][in_cl1]
            pairs_marginals[(q0, q1)][input_bitstring] = np.kron(marginal_0,
                                                                 marginal_1)

    return pairs_marginals


def calculate_pairs_marginals_from_tensor_model_fixed_input(input_state: str,
                                                            noise_matrices_dictionary: Dict[Tuple[int],
                                                                                            np.ndarray],
                                                            needed_pairs,
                                                            get_also_1q_marginals=True
                                                            ):
    # TODO FBM: generalize to higher locality
    pairs_marginals = {}
    for cluster, noise_matrix in noise_matrices_dictionary.items():
        pairs_in_cluster = [(qi, qj) for qi in cluster for qj in cluster
                            if qj > qi and (qi, qj) in needed_pairs]
        cluster_input_state = ''.join([input_state[x] for x in cluster])
        enumerated_qubits = utils.map_index_to_order(list(cluster))
        marginals_now = get_noisy_marginals_within_cluster(input_state=cluster_input_state,
                                                           cluster_noise_matrix=noise_matrix,
                                                           subsets_of_interest=pairs_in_cluster,
                                                           qubits_mapping=enumerated_qubits)

        pairs_marginals = {**pairs_marginals, **marginals_now}

        if get_also_1q_marginals:
            marginals_now = get_noisy_marginals_within_cluster(input_state=cluster_input_state,
                                                               cluster_noise_matrix=noise_matrix,
                                                               subsets_of_interest=[[qi] for qi in
                                                                                    cluster],
                                                               qubits_mapping=enumerated_qubits)

            pairs_marginals = {**pairs_marginals, **marginals_now}

    already_done = list(pairs_marginals.keys())
    for pair in utils.lists_difference(needed_pairs, already_done):
        # if pair not in already_done:
        (q0, q1) = pair
        marginal_0 = pairs_marginals[(q0,)]
        marginal_1 = pairs_marginals[(q1,)]

        pairs_marginals[(q0, q1)] = np.kron(marginal_0, marginal_1)

    return pairs_marginals



# def calculate_marginals_from_noise_model(input_state: List[str],
#                                          noise_model_description,
#                                          neighborhoods_tuples,
#                                          # number_of_qubits: int,
#                                          needed_subsets: List[str],
#                                          correction_indices: Dict[str, str],
#                                          # noise_model_description: Dict[
#                                          #     Tuple[int], Dict[str, np.ndarray]],
#                                          # neighborhoods_tuples: Dict[Tuple[int], List[int]],
#                                          clusters_dictionary: Dict[int, List[int]],
#                                          neighborhoods: Dict[Tuple[int], List[int]],
#                                          noise_matrices_dictionary: dict
#                                          ):
#     noise_matrices_now = {}
#     clusters_list = []
#     for cluster_qubits, noise_matrices_on_cluster in noise_model_description.items():
#         clusters_list.append(cluster_qubits)
#         # print(neighborhoods_tuples[cluster_qubits])
#         input_on_cluster = ''.join([input_state[x] for x in cluster_qubits])

#         if len(neighborhoods_tuples[cluster_qubits]) == 0:
#             noise_matrix = noise_matrices_on_cluster['averaged']
#         else:
#             input_on_neighbors = ''.join(
#                 [input_state[x] for x in neighborhoods_tuples[cluster_qubits]])
#             noise_matrix = noise_matrices_on_cluster[input_on_neighbors]

#         noise_matrices_now[cluster_qubits] = noise_matrix
#         # prob_distro_now = noise_matrix[:, int(input_on_cluster, 2)]

# #Mocomm - why private function inside a function...
#     def _find_clusters_in_subset(qubits_list): 
#         clusters_here = []
#         for qubit in qubits_list:
#             for cl in clusters_list:
#                 if qubit in cl:
#                     if cl not in clusters_here:
#                         clusters_here.append(cl)

#         return clusters_here
#     #TODO: undestand the role of MarginalAnalyzer below
#     marginals_analyzer = MarginalsAnalyzerBase({},
#                                                False)
                                    

#     marginal_probs_big = {}
#     marginals = {}
#     for subset in needed_subsets:
#         qubits_string = 'q' + 'q'.join([str(x) for x in subset])
#         bigger_subset = convert.get_qubit_indices_from_keystring(correction_indices[qubits_string])

#         clusters_now = _find_clusters_in_subset(bigger_subset)

#         # qubits_ordering_clusters = []
#         qubits_ordering_all = []
#         for cl in clusters_now:
#             # qubits_ordering_clusters += list(cl)
#             qubits_ordering_all += list(cl)
#             for q in neighborhoods_tuples[cl]:
#                 if q not in qubits_ordering_all:
#                     qubits_ordering_all.append(q)

#         reversed_enum = utils.map_index_to_order(qubits_ordering_all)

#         # if tuple(clusters_now) in marginal_probs_big.keys():
#         #     probability_distribution = marginal_probs_big[tuple(clusters_now)]

#         # else:
#         noise_matrices_dictionary_local = {}
#         clusters_list_local = []
#         local_neighbors_dictionary = {}

#         for cl in clusters_now:
#             cluster_key_local = 'q' + 'q'.join([str(reversed_enum[x]) for x in cl])

#             neighbors_now = neighborhoods_tuples[cl]

#             noise_matrices_now = noise_model_description[cl]

#             # neighbors_enum = utils.map_index_to_order(neighbors_now)

#             # neighbors_local = [q for q in neighbors_now if q in qubits_ordering_all]
#             # neighbors_outside = [q for q in neighbors_now if q not in qubits_ordering_all]

#             # if len(neighbors_outside)!=0:

#             noise_matrices_dictionary_local[cluster_key_local] = noise_model_description[cl]
#             clusters_list_local.append([reversed_enum[x] for x in cl])
#             local_neighbors_dictionary[cluster_key_local] = [reversed_enum[x] for x in
#                                                              neighborhoods_tuples[cl]]

#         global_matrix_creator = GlobalNoiseMatrixCreator(
#             noise_matrices_dictionary=noise_matrices_dictionary_local,
#             clusters_list=clusters_list_local,
#             neighborhoods=local_neighbors_dictionary)

#         input_state_local = ''.join([input_state[x] for x in qubits_ordering_all])
#         marginal_now_big = []
#         full_register_local = anf.get_classical_register_bitstrings(range(len(qubits_ordering_all)))
#         for local_output_state in full_register_local:
#             marginal_now_big.append(
#                 global_matrix_creator.compute_matrix_element(input_state=input_state_local,
#                                                              output_state=local_output_state))

#         probability_distribution = np.array(marginal_now_big).reshape(len(marginal_now_big), 1)
#         marginal_probs_big[tuple(clusters_now)] = probability_distribution

#         bits_of_interest_now = [reversed_enum[q] for q in subset]

#         marginal_distro = marginals_analyzer.get_marginal_from_probability_distribution(
#             global_probability_distribution=probability_distribution,
#             bits_of_interest=bits_of_interest_now)

#         marginals[qubits_string] = marginal_distro

#         for q in subset:
#             marginals['q%s' % q] = marginals_analyzer.get_marginal_from_probability_distribution(
#                 global_probability_distribution=probability_distribution,
#                 bits_of_interest=[reversed_enum[q]])

#     return marginals





##############################################################
##################### DEPRECIATED? ###########################
##############################################################

def calculate_pairs_marginals_from_tensor_model_terrible(input_state: str,
                                                         noise_matrices_dictionary: Dict[Tuple[int],
                                                                                         np.ndarray],
                                                         needed_pairs,
                                                         get_also_1q_marginals=True
                                                         ):
    # TODO FBM: generalize to higher locality

    all_clusters = list(noise_matrices_dictionary.keys())

    pairs_marginals = {}

    for cluster, noise_matrix in noise_matrices_dictionary.items():
        pairs_in_cluster = [(qi, qj) for qi in cluster for qj in cluster if
                            qj > qi and (qi, qj) in needed_pairs]

        cluster_input_state = ''.join([input_state[x] for x in cluster])

        enumerated_qubits = utils.map_index_to_order(list(cluster))

        marginals_now = get_noisy_marginals_within_cluster(input_state=cluster_input_state,
                                                           cluster_noise_matrix=noise_matrix,
                                                           subsets_of_interest=pairs_in_cluster,
                                                           qubits_mapping=enumerated_qubits)

        pairs_marginals = {**pairs_marginals, **marginals_now}

        if get_also_1q_marginals:
            marginals_now = get_noisy_marginals_within_cluster(input_state=cluster_input_state,
                                                               cluster_noise_matrix=noise_matrix,
                                                               subsets_of_interest=[[qi] for qi in
                                                                                    cluster],
                                                               qubits_mapping=enumerated_qubits)

            pairs_marginals = {**pairs_marginals, **marginals_now}

        other_clusters = list(set(all_clusters).difference(set([cluster])))
        for cluster2 in other_clusters:
            pairs_between_clusters = [(qi, qj) for qi in cluster for qj in cluster2 if qj > qi]

            enumerated_qubits = utils.map_index_to_order(
                list(cluster) + list(cluster2))

            # print(cluster,cluster2)
            # print(enumerated_qubits)
            local_number_of_qubits = len(cluster) + len(cluster2)

            embeded_noise_matrix1 = qanf.embed_operator_in_bigger_hilbert_space(
                number_of_qubits=local_number_of_qubits,
                local_operator=noise_matrix,
                global_indices=[enumerated_qubits[x] for x in cluster]
            ).real
            embeded_noise_matrix2 = qanf.embed_operator_in_bigger_hilbert_space(
                number_of_qubits=local_number_of_qubits,
                local_operator=noise_matrices_dictionary[cluster2],
                global_indices=[enumerated_qubits[x] for x in cluster2]
            ).real

            local_tensor_noise_model = embeded_noise_matrix1 @ embeded_noise_matrix2

            input_state_now_on_two_clusters = ''.join([input_state[x] for x in
                                                       list(cluster) + list(cluster2)])

            marginals_now = get_noisy_marginals_within_cluster(
                input_state=input_state_now_on_two_clusters,
                cluster_noise_matrix=local_tensor_noise_model,
                subsets_of_interest=pairs_between_clusters,
                qubits_mapping=enumerated_qubits)

            pairs_marginals = {**pairs_marginals,
                               **marginals_now}

    return pairs_marginals




    # raise KeyError

    # # # big_noise_matrix = 1
    # # # for cl in clusters_now:
    # # #     noise_matrices_on_cluster = noise_model_description[cl]
    # # #     if len(neighborhoods_tuples[cl]) == 0:
    # # #         noise_matrix_local = noise_matrices_on_cluster['averaged']
    # # #     else:
    # # #         input_on_neighbors = ''.join(
    # # #             [input_state[x] for x in neighborhoods_tuples[cl]])
    # # #         noise_matrix_local = noise_matrices_on_cluster[input_on_neighbors]
    # # #
    # # #     big_noise_matrix = np.kron(big_noise_matrix, noise_matrix_local)
    # #
    # # input_on_cluster = ''.join([input_state[x] for x in cl])
    # #
    # # probability_distribution = big_noise_matrix[:, int(input_on_cluster, 2)]

    # really_needed_subsets = []
    # for subset in needed_subsets:
    #     qubits_string = 'q' + 'q'.join([str(x) for x in subset])
    #     bigger_subset = convert.get_qubit_indices_from_keystring(correction_indices[qubits_string])
    #     if bigger_subset not in really_needed_subsets:
    #         really_needed_subsets.append(sorted(bigger_subset))

    # marginals = {}
    # for subset in really_needed_subsets:
    #     clusters_now = []

    #     for qubit in subset:
    #         if clusters_dictionary[qubit] not in clusters_now:
    #             clusters_now.append(clusters_dictionary[qubit])

    #     qubits_with_neighbors = []
    #     for cluster in clusters_now:
    #         qubits_with_neighbors += neighborhoods[tuple(cluster)]
    #         qubits_with_neighbors += cluster

    #     qubits_with_neighbors = sorted(qubits_with_neighbors)
    #     enumerated_indices = utils.map_index_to_order(qubits_with_neighbors)
    #     clusters_enumerated = [[enumerated_indices[x] for x in cluster] for cluster in clusters_now]
    #     local_neighbors_dictionary = {}

    #     # print(neighborhoods)
    #     for cluster in clusters_now:
    #         key_qubits = 'q' + 'q'.join([str(enumerated_indices[x]) for x in cluster])
    #         local_neighbors_dictionary[key_qubits] = [enumerated_indices[x] for x in
    #                                                   neighborhoods[tuple(cluster)]]

    #     noise_matrices_dictionary_local = {'q' + 'q'.join([str(x) for x in cluster]): {} for cluster in
    #                                        clusters_enumerated}

    #     for cluster in clusters_now:
    #         key_qubits = 'q' + 'q'.join([str(x) for x in cluster])
    #         key_qubits_local = 'q' + 'q'.join([str(enumerated_indices[x]) for x in cluster])

    #         dict_now = noise_matrices_dictionary[key_qubits]

    #         noise_matrices_dictionary_local[key_qubits_local]['averaged'] = dict_now['averaged']

    #         if len(neighborhoods[tuple(cluster)]) != 0:
    #             qubits_neighbors_key = 'q' + 'q'.join([str(x) for x in neighborhoods[tuple(cluster)]])
    #             qubits_neighbors_local_key = 'q' + 'q'.join(
    #                 [str(enumerated_indices[x]) for x in neighborhoods[tuple(cluster)]])
    #             noise_matrices_dictionary_local[key_qubits_local][qubits_neighbors_local_key] = \
    #                 dict_now[qubits_neighbors_key]

    #     # print(noise_matrices_dictionary)
    #     # print(noise_matrices_dictionary_local)
    #     # raise KeyError
    #     register_size_local = len(qubits_with_neighbors)

    #     input_state_local = ''.join([input_state[x] for x in qubits_with_neighbors])

    #     full_register_local = anf.get_classical_register_bitstrings(
    #         qubit_indices=list(range(register_size_local)))

    #     global_matrix_creator = GlobalNoiseMatrixCreator(
    #         noise_matrices_dictionary=noise_matrices_dictionary_local,
    #         clusters_list=clusters_enumerated,
    #         neighborhoods=local_neighbors_dictionary)

    #     marginal_now = []
    #     # print(full_register_local)
    #     for local_output_state in full_register_local:
    #         marginal_now.append(
    #             global_matrix_creator.compute_matrix_element(input_state=input_state_local,
    #                                                          output_state=local_output_state))

    #     marginal_now = np.array(marginal_now).reshape(len(marginal_now), 1)

    #     qubits_key_final = 'q' + 'q'.join([str(x) for x in subset])
    #     marginals[qubits_key_final] = marginal_now

    # return marginals








