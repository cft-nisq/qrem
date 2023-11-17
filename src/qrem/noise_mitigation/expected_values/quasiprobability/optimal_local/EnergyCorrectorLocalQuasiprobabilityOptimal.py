"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""

from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple, Union

import numpy as np

from qrem.functions_qrem import functions_data_analysis as fda, ancillary_functions as anf
from qrem.functions_qrem.functions_data_analysis import _sort_clusters_division
from qrem.noise_mitigation.base_classes.EnergyEstimatorBase import EnergyEstimatorBase
from qrem.noise_mitigation.expected_values.quasiprobability.optimal_local import \
    functions_optimal_quasiprobability as foq

from qrem.common.printer import qprint_array
from qrem.common import convert

class EnergyCorrectorLocalQuasiprobabilityOptimal(EnergyEstimatorBase):
    """
    This is class for implementing quasiprobability-like error mitigation based on CTMP model [#TODO FBM: add ref].

    """

    # TODO FBM: test whether all fo this works in toy examples

    def __init__(self,
                 results_dictionary: Dict[str, Dict[str, int]],
                 hamiltonian_dictionaries: Dict[str, Dict[Tuple[int], float]],
                 local_noise_matrices: Dict[Tuple[int], np.ndarray],
                 quasiprobability_dictionary: Optional = None

                 ) -> None:
        """
        :param results_dictionary: see parent class description
        :param hamiltonian_dictionaries: see parent class description
        """

        super().__init__(results_dictionary=results_dictionary,
                         hamiltonian_dictionaries=hamiltonian_dictionaries)

        self._energies_dictionary_corrected = {}
        self._default_max_number_of_samples = 10 ** 5
        self._clusters_list = list(local_noise_matrices.keys())

        if quasiprobability_dictionary is None:
            quasiprobability_dictionary = foq.get_quasiprobability_dictionary_from_noise_matrices(
                local_noise_matrices=local_noise_matrices)

        self._quasiprobability_dictionary = quasiprobability_dictionary

        qubits_to_clusters_map = {}
        for cluster in self._clusters_list:
            for qi in cluster:
                qubits_to_clusters_map[(qi,)] = cluster

        self._qubits_to_clusters_map = qubits_to_clusters_map

    @property
    def quasiprobability_dictionary(self) -> Dict[Tuple[int], Tuple]:
        return self._quasiprobability_dictionary

    @quasiprobability_dictionary.setter
    def quasiprobability_dictionary(self,
                                    quasiprobability_dictionary: Dict[Tuple[int], Tuple]) -> None:
        self._quasiprobability_dictionary = quasiprobability_dictionary

    def estimate_corrected_energy_local_separable(self,
                                                  experiment_key: str,
                                                  number_of_samples_dictionary: Optional[
                                                      Union[int, Dict[Tuple[int], int]]] = None,
                                                  target_estimation_accuracy: Optional[
                                                      float] = 0.01,
                                                  ):

        clusters_list_global = self._clusters_list
        quasiprobability_dictionary = self._quasiprobability_dictionary
        qubits_to_clusters_map = self._qubits_to_clusters_map

        if number_of_samples_dictionary is None:
            number_of_samples_dictionary = {}
            for i in range(len(clusters_list_global)):
                cluster_i = clusters_list_global[i]
                samples_i = quasiprobability_dictionary[cluster_i][
                                2] ** 2 / target_estimation_accuracy ** 2

                number_of_samples_dictionary[(cluster_i,)] = int(
                    np.min([samples_i, self._default_max_number_of_samples]))
                for j in range(i + 1, len(clusters_list_global)):
                    cluster_j = clusters_list_global[j]

                    sorted_tuple = _sort_clusters_division([cluster_i, cluster_j])
                    samples_ij = quasiprobability_dictionary[cluster_j][2] ** 2 * \
                                 quasiprobability_dictionary[cluster_i][
                                     2] ** 2 / target_estimation_accuracy ** 2
                    number_of_samples_dictionary[sorted_tuple] = int(
                        np.min([samples_ij, self._default_max_number_of_samples]))
            # print(number_of_samples_dictionary)


        elif isinstance(number_of_samples_dictionary, int):
            number_of_samples_dict = {}
            for i in range(len(clusters_list_global)):
                cluster_i = clusters_list_global[i]
                number_of_samples_dict[(cluster_i,)] = number_of_samples_dictionary
                for j in range(i + 1, len(clusters_list_global)):
                    cluster_j = clusters_list_global[j]
                    sorted_tuple = _sort_clusters_division([cluster_i, cluster_j])
                    number_of_samples_dict[sorted_tuple] = number_of_samples_dictionary
            number_of_samples_dictionary = number_of_samples_dict

        experimental_results = self._results_dictionary[experiment_key]
        hamiltonian_weights = self._hamiltonian_dictionaries[experiment_key]
        experimental_outcomes = list(experimental_results.keys())
        probability_distribution_over_outcomes = np.array(list(experimental_results.values()),
                                                          dtype=float)
        probability_distribution_over_outcomes /= np.sum(probability_distribution_over_outcomes)
        local_normalizations_dict = {cluster: quasiprobability_dictionary[cluster][2] for cluster in
                                     clusters_list_global}
        # local_probabilities = {cluster:quasiprobability_dictionary[cluster][0][0] for cluster in
        #                              clusters_list}

        local_registers_dictionary = {len(cluster):
                                          [tuple(convert.integer_to_bitstring(index_outcome, len(cluster)))
                                           for index_outcome in range(int(2 ** len(cluster)))]
                                      for cluster in clusters_list_global}

        local_registers_dictionary_REVERSED = {len(cluster):
                                                   {convert.integer_to_bitstring(index_outcome,
                                                                             len(cluster)):
                                                        index_outcome
                                                    for index_outcome in range(int(2 ** len(cluster)))}
                                               for cluster in clusters_list_global}

        local_bitstrings_experimental_dict = {bitstring:
                                                  {cluster: ''.join([bitstring[qi] for qi in cluster])
                                                   for cluster in clusters_list_global}
                                              for bitstring in experimental_outcomes}

        local_sets_dictionary = {}

        all_outcomes_samples_uniform = {}
        for i in range(len(clusters_list_global)):
            cluster = clusters_list_global[i]
            number_of_samples_i = number_of_samples_dictionary[(cluster,)]
            all_outcomes_samples_uniform[(cluster,)] = np.random.multinomial(n=number_of_samples_i,
                                                                             pvals=probability_distribution_over_outcomes)
            local_sets_dictionary[(cluster,)] = []

            for j in range(i + 1, len(clusters_list_global)):
                cluster2 = clusters_list_global[j]
                sorted_tuple = _sort_clusters_division([cluster, cluster2])

                number_of_samples_i_j = number_of_samples_dictionary[sorted_tuple]

                all_outcomes_samples_uniform[sorted_tuple] = np.random.multinomial(
                    n=number_of_samples_i_j,
                    pvals=probability_distribution_over_outcomes)

                local_sets_dictionary[sorted_tuple] = []

        for pair in hamiltonian_weights.keys():
            qi = pair[0]
            cluster_i = qubits_to_clusters_map[(qi,)]
            if len(pair) == 1:
                continue
            qj = pair[1]
            cluster_j = qubits_to_clusters_map[(qj,)]

            if cluster_i == cluster_j:
                local_sets_dictionary[(cluster_i,)].append(pair)
            else:
                sorted_tuple = _sort_clusters_division([cluster_i, cluster_j])
                local_sets_dictionary[sorted_tuple].append(pair)

        rng = np.random.default_rng()
        energy_estimator = 0
        for clusters_tuple, pairs_list in local_sets_dictionary.items():
            qubits_in_clusters_tuple = []
            for cluster in clusters_tuple:
                for qi in cluster:
                    qubits_in_clusters_tuple.append(qi)

            counts_dictionary_now = {}
            samples_uniform_outcomes = all_outcomes_samples_uniform[clusters_tuple]
            outcomes_range = range(len(samples_uniform_outcomes))

            # TODO FBM: temporary
            for index_outcome in outcomes_range:
                number_of_samples_from_outcome = samples_uniform_outcomes[index_outcome]
                bitstring_now = experimental_outcomes[index_outcome]

                samples_pairs = []
                for cluster in clusters_tuple:
                    local_input_state_index = local_registers_dictionary_REVERSED[len(cluster)][
                        local_bitstrings_experimental_dict[bitstring_now][cluster]]

                    sample_1 = rng.binomial(n=number_of_samples_from_outcome,
                                            p=quasiprobability_dictionary[cluster][0][0])
                    # sample_1 = rng.multinomial(n=number_of_samples_from_outcome,
                    #                          pvals=[quasiprobability_dictionary[cluster][0][0],quasiprobability_dictionary[cluster][0][1]])[0]
                    sample_2 = number_of_samples_from_outcome - sample_1

                    local_matrices_samples_dict = {0: sample_1,
                                                   1: sample_2}

                    local_samples = []
                    for index_outcome_local, number_of_outcomes_per_matrix \
                            in local_matrices_samples_dict.items():
                        if index_outcome_local == 0:
                            local_sign = 1
                        elif index_outcome_local == 1:
                            local_sign = -1
                        else:
                            raise ValueError("Not binomial??")

                        local_matrix_now = quasiprobability_dictionary[cluster][1][index_outcome_local]
                        local_distro_now = local_matrix_now[:, local_input_state_index]
                        try:
                            local_output_states = rng.multinomial(n=number_of_outcomes_per_matrix,
                                                                  pvals=local_distro_now)
                        except(ValueError):
                            qprint_array(local_matrix_now)
                            print(local_distro_now, sum(local_distro_now))
                            raise KeyboardInterrupt

                        for further_index in range(len(local_output_states)):
                            local_samples += [(local_sign,
                                               local_registers_dictionary[len(cluster)][further_index]
                                               # convert.integer_to_bitstring(index_outcome, len(cluster))
                                               )
                                              ] * local_output_states[further_index]

                    rng.shuffle(local_samples)
                    # all_samples[cluster] = local_samples

                    samples_pairs.append(local_samples)

                if len(clusters_tuple) == 1:
                    samples_pairs = list(samples_pairs[0])
                    counting = Counter(samples_pairs)
                    for tuple_outcome, amount_of_ticks in counting.items():
                        global_output_state = ['9'] * self._number_of_qubits
                        cluster_local = clusters_tuple[0]
                        for qubit_index in range(len(cluster_local)):
                            global_output_state[cluster_local[qubit_index]] = \
                                tuple_outcome[1][qubit_index]

                        global_output_state = ''.join(global_output_state)
                        if global_output_state not in counts_dictionary_now.keys():
                            counts_dictionary_now[global_output_state] = defaultdict(float)

                        counts_dictionary_now[global_output_state][
                            (tuple_outcome[0],)] += amount_of_ticks

                else:
                    samples_pairs = list(zip(*samples_pairs))
                    counting = Counter(samples_pairs)
                    for tuple_outcome, amount_of_ticks in counting.items():
                        global_output_state = ['9'] * self._number_of_qubits
                        for clust_index in range(len(clusters_tuple)):
                            cluster_local = clusters_tuple[clust_index]
                            for qubit_index in range(len(cluster_local)):
                                global_output_state[cluster_local[qubit_index]] = \
                                    tuple_outcome[clust_index][1][qubit_index]

                        global_output_state = ''.join(global_output_state)
                        if global_output_state not in counts_dictionary_now.keys():
                            counts_dictionary_now[global_output_state] = defaultdict(float)

                        counts_dictionary_now[global_output_state][
                            tuple([x[0] for x in tuple_outcome])] += amount_of_ticks

            local_weights_now = {}
            for pair in pairs_list:
                if pair not in hamiltonian_weights.keys():
                    continue
                # weight_now = hamiltonian_weights[pair]
                local_weights_now[pair] = hamiltonian_weights[pair]
            if len(clusters_tuple) == 1:
                for cluster in clusters_tuple:
                    for qi in cluster:
                        if (qi,) not in hamiltonian_weights.keys():
                            continue
                        local_weights_now[(qi,)] = hamiltonian_weights[(qi,)]

            local_energy = 0
            for global_output_state, signs_dict in counts_dictionary_now.items():
                for local_signs, value in signs_dict.items():
                    additional_multipliers_now = {}
                    # TODO FBM: careful here
                    for pair in local_weights_now.keys():
                        qi = pair[0]
                        cluster_i = qubits_to_clusters_map[(qi,)]
                        cluster_index_i = 0
                        sign_i = local_signs[cluster_index_i]

                        local_sign = sign_i
                        local_normalization_now = local_normalizations_dict[cluster_i]

                        # additional_multipliers_now[(qi,)] = local_sign*local_normalization_now*value

                        if len(pair) == 2:
                            qj = pair[1]
                            cluster_j = qubits_to_clusters_map[(qj,)]
                            if cluster_i != cluster_j:
                                cluster_index_j = 1
                                sign_j = local_signs[cluster_index_j]
                                local_sign *= sign_j
                                local_normalization_now *= local_normalizations_dict[cluster_j]

                        #TODO FBM: playing here
                        additional_multipliers_now[pair] = local_sign * local_normalization_now * value

                    energy_local_now = fda.get_energy_from_bitstring_diagonal_local(
                        bitstring=global_output_state,
                        weights_dictionary=local_weights_now,
                        additional_multipliers=additional_multipliers_now)

                    # energy_local_now *= local_sign * local_normalization_now*value
                    local_energy += energy_local_now

            energy_estimator += local_energy / number_of_samples_dictionary[clusters_tuple]

        energy_estimated = energy_estimator

        self._energies_dictionary_corrected[experiment_key] = energy_estimated

        return energy_estimated
