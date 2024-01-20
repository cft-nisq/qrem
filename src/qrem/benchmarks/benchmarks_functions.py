from qrem.common import probability
from qrem.common import utils 
from qrem.benchmarks import hamiltonians
from typing import Dict, Tuple, List, Type, Union
import numpy as np



def calculate_pairs_marginals_from_tensor_model_fixed_input(input_state: str,
                                                            noise_matrices_dictionary: Dict[Tuple[int],
                                                                                            np.ndarray],
                                                            needed_pairs,
                                                            get_also_1q_marginals=True
                                                            ):
    # FBM: generalize to higher locality
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

def get_noisy_marginals_within_cluster(input_state: str,
                                       cluster_noise_matrix: np.ndarray,
                                       subsets_of_interest: List[Union[Tuple[int], List[int]]],
                                       qubits_mapping: Dict[int, int]) -> Dict[Tuple[int], np.ndarray]:
    input_state_binary = int(input_state, 2)

    probability_distribution = cluster_noise_matrix[:, input_state_binary]

    marginals_now = {}
    for qubits_of_interest in subsets_of_interest:
        marginal = probability.get_marginal_from_probability_distribution(
            global_probability_distribution=probability_distribution,
            bits_of_interest=[qubits_mapping[x] for x in qubits_of_interest])
        marginals_now[tuple(qubits_of_interest)] = marginal

    return marginals_now

def get_noisy_energy_product_noise_model(input_state,
                                         noise_matrices_dictionary,
                                         weights_dictionary_tuples,
                                         needed_pairs=None):
    pairs_marginals_noisy = calculate_pairs_marginals_from_tensor_model_fixed_input(
        input_state=input_state,
        noise_matrices_dictionary=noise_matrices_dictionary,
        needed_pairs=needed_pairs,
        get_also_1q_marginals=True)

    energy_modeled_exact = hamiltonians.estimate_energy_from_marginals(
        weights_dictionary=weights_dictionary_tuples,
        marginals_dictionary=pairs_marginals_noisy)

    return energy_modeled_exact