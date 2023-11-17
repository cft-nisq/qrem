"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""

import time
from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple, List, Union

import numpy as np

from qrem.functions_qrem import functions_data_analysis as fda, ancillary_functions as anf, \
    quantum_ancillary_functions as quanf
from qrem.functions_qrem.functions_data_analysis import _sort_clusters_division
from qrem.noise_mitigation.base_classes.EnergyEstimatorBase import EnergyEstimatorBase
from qrem.noise_model_generation.CTMP import functions_CTMP as fun_CTMP
from qrem.noise_simulation.CTMP import functions_CTMP_conditional as CTMP_cond, \
    functions_CTMP_simple as CTMP_simp

from qrem.common.printer import qprint
from qrem.common import convert

class EnergyCorrectorLocalCTMP(EnergyEstimatorBase):
    """
    This is class for implementing quasiprobability-like error mitigation based on CTMP model [#TODO FBM: add ref].

    """

    # TODO FBM: test whether all fo this works in toy examples

    def __init__(self,
                 results_dictionary: Dict[str, Dict[str, int]],
                 hamiltonian_dictionaries: Dict[str, Dict[Tuple[int], float]],
                 CTMP_rates_dictionary: Dict[Tuple[int], Dict[str, float]],
                 clusters_list: List[Tuple[int]],
                 local_noise_strengths: Optional[Dict[Tuple[int], float]] = None,
                 local_noise_matrices: Optional[Dict[Tuple[int], np.ndarray]] = None

                 ) -> None:
        """
        :param results_dictionary: see parent class description
        :param hamiltonian_dictionaries: see parent class description
        :param CTMP_rates_dictionary: dictionary of error rates
        """

        super().__init__(results_dictionary=results_dictionary,
                         hamiltonian_dictionaries=hamiltonian_dictionaries)

        self._CTMP_rates_dictionary = CTMP_rates_dictionary
        self._energies_dictionary_corrected = {}
        self._default_max_number_of_samples = 10 ** 5
        self._clusters_list = clusters_list

        local_rates_dictionaries = fun_CTMP._convert_to_local_noise_rates_dictionary(
            clusters_list=clusters_list,
            CTMP_rates_dictionary=CTMP_rates_dictionary)

        if local_noise_strengths is None:
            local_noise_strengths = fun_CTMP.calculate_noise_strengths_local(
                local_rates_dictionaries=local_rates_dictionaries,
                method='bruteforce')
        if local_noise_matrices is None:
            local_noise_matrices = fun_CTMP.get_local_noise_matrices(
                local_rates_dictionaries=local_rates_dictionaries,
                local_noise_strengths=local_noise_strengths)
        self._local_noise_strengths = local_noise_strengths
        self._local_noise_matrices = local_noise_matrices

        matrix_powers_dictionary = {cluster: {} for cluster in clusters_list}
        for cluster in clusters_list:
            for matrix_power in range(11):
                matrix_powers_dictionary[cluster][matrix_power] = np.linalg.matrix_power(
                    local_noise_matrices[cluster], matrix_power)

        self._matrix_powers_dictionary = matrix_powers_dictionary

        qubits_to_clusters_map = {}
        for cluster in clusters_list:
            for qi in cluster:
                qubits_to_clusters_map[(qi,)] = cluster

        self._qubits_to_clusters_map = qubits_to_clusters_map

        self.__CTMP_generator_matrices = fun_CTMP.get_generators_dictionary_from_rates(
            rates_dictionary=CTMP_rates_dictionary)

    @property
    def CTMP_rates_dictionary(self) -> Dict[Tuple[int], Dict[str, float]]:
        return self._CTMP_rates_dictionary

    @CTMP_rates_dictionary.setter
    def CTMP_rates_dictionary(self, CTMP_rates_dictionary: Dict[Tuple[int], Dict[str, float]]) -> None:
        self._CTMP_rates_dictionary = CTMP_rates_dictionary

    def estimate_corrected_energy_CTMP_local_separable(self,
                                                       experiment_key: str,
                                                       number_of_samples_dictionary: Optional[
                                                           Union[int, Dict[Tuple[int], int]]] = None,
                                                       target_estimation_accuracy: Optional[
                                                           float] = 0.01,
                                                       ):


        local_noise_strengths = self._local_noise_strengths
        clusters_list_global = self._clusters_list
        qubits_to_clusters_map = self._qubits_to_clusters_map
        matrix_powers_dictionary = self._matrix_powers_dictionary
        local_noise_matrices = self._local_noise_matrices

        # import time
        t0 = time.time()
        if number_of_samples_dictionary is None:
            number_of_samples_dictionary = {}
            for i in range(len(clusters_list_global)):
                cluster_i = clusters_list_global[i]
                samples_i = fun_CTMP.get_theoretical_number_of_samples_CTMP(
                    noise_strength_CTMP=local_noise_strengths[cluster_i],
                    target_estimation_accuracy=target_estimation_accuracy)
                number_of_samples_dictionary[(cluster_i,)] = int(
                    np.min([samples_i, self._default_max_number_of_samples]))
                for j in range(i + 1, len(clusters_list_global)):
                    cluster_j = clusters_list_global[j]

                    sorted_tuple = _sort_clusters_division([cluster_i, cluster_j])
                    samples_ij = fun_CTMP.get_theoretical_number_of_samples_CTMP(
                        noise_strength_CTMP=local_noise_strengths[cluster_i] + local_noise_strengths[
                            cluster_j],
                        target_estimation_accuracy=target_estimation_accuracy)
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

        # for key, value in number_of_samples_dictionary.items():
        #     print(key,value)
        # raise KeyboardInterrupt
        # print(number_of_samples_dictionary)

        experimental_results = self._results_dictionary[experiment_key]
        hamiltonian_weights = self._hamiltonian_dictionaries[experiment_key]

        # global_output_state_template = ['9'] * self._number_of_qubits

        experimental_outcomes = list(experimental_results.keys())
        probability_distribution_over_outcomes = np.array(list(experimental_results.values()),
                                                          dtype=float)
        probability_distribution_over_outcomes /= np.sum(probability_distribution_over_outcomes)

        local_normalizations_dict = {cluster: np.exp(2 * local_noise_strengths[cluster]) for cluster in
                                     clusters_list_global}

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

                # sorted_tuple = _sort_clusters_division([cluster, cluster2])
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
        # t1=time.time()
        # qprint("PREPROCESSING:",t1-t0)
        # raise KeyboardInterrupt

        # t1=  time.time()
        rng = np.random.default_rng()
        energy_estimator = 0
        t_sampling, t_formatting, t_energy, t_filling = 0, 0, 0, 0
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

                    noise_strength_now = local_noise_strengths[cluster]

                    t_sampling_start0 = time.time()
                    local_matrix_powers = rng.poisson(lam=noise_strength_now,
                                                      size=number_of_samples_from_outcome)
                    local_matrix_powers_dict = defaultdict(float)
                    t_sampling += time.time() - t_sampling_start0
                    for power in local_matrix_powers:
                        local_matrix_powers_dict[power] += 1
                    local_samples = []
                    for local_matrix_power, number_of_outcomes_per_power \
                            in local_matrix_powers_dict.items():
                        if (local_matrix_power % 2) == 0:
                            local_sign = 1
                        else:
                            local_sign = -1
                        try:
                            local_matrix_now = matrix_powers_dictionary[cluster][local_matrix_power]
                        except(KeyError):
                            matrix_powers_dictionary[cluster][
                                local_matrix_power] = np.linalg.matrix_power(
                                local_noise_matrices[cluster], local_matrix_power)
                            local_matrix_now = matrix_powers_dictionary[cluster][local_matrix_power]

                        local_distro_now = local_matrix_now[:, local_input_state_index]
                        t_sampling_start1 = time.time()
                        local_output_states = rng.multinomial(n=number_of_outcomes_per_power,
                                                              pvals=local_distro_now)
                        t_sampling += time.time() - t_sampling_start1

                        t_filling0 = time.time()
                        for further_index in range(len(local_output_states)):
                            local_samples += [(local_sign,
                                               local_registers_dictionary[len(cluster)][further_index]
                                               )
                                              ] * local_output_states[further_index]
                        t_filling += time.time() - t_filling0

                    # rng = np.random.default_rng()
                    t_sampling_start2 = time.time()
                    rng.shuffle(local_samples)
                    # all_samples[cluster] = local_samples
                    samples_pairs.append(local_samples)
                    t_sampling += time.time() - t_sampling_start2
                #
                t3 = time.time()


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
                    # print(samples_pairs[0])

                    # raise KeyboardInterrupt
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

                t4 = time.time()
                t_formatting += t4 - t3
                #
                # t5= time.time()
                # for i in range(number_of_samples_from_outcome):
                #     global_output_state = ['9']*self._number_of_qubits
                #     local_signs = [all_samples[cluster][i][0] for cluster in clusters_tuple]
                #     for cluster in clusters_tuple:
                #         local_output_state = all_samples[cluster][i][1]
                #         for qubit_index in range(len(cluster)):
                #             global_output_state[cluster[qubit_index]] = local_output_state[qubit_index]
                #
                #     global_output_state = ''.join(global_output_state)
                #     if global_output_state not in counts_dictionary_now.keys():
                #         counts_dictionary_now[global_output_state] = defaultdict(float)
                #
                #     counts_dictionary_now[global_output_state][tuple(local_signs)] += 1
                # t6=time.time()
                # t_formatting+=t6-t5

            t5 = time.time()
            # t23+=t3-t2
            #TODO FBM: Make sure it does take into account every possible case
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

                        # else:
                        #     print('hej',pair)

                        additional_multipliers_now[pair] = local_sign * local_normalization_now * value

                        # for qk in pair:
                        #     additional_multipliers_now[(qk,)]

                    energy_local_now = fda.get_energy_from_bitstring_diagonal_local(
                        bitstring=global_output_state,
                        weights_dictionary=local_weights_now,
                        additional_multipliers=additional_multipliers_now)

                    # energy_local_now *= local_sign * local_normalization_now*value
                    local_energy += energy_local_now

            energy_estimator += local_energy / number_of_samples_dictionary[clusters_tuple]

            t6 = time.time()
            t_energy += t6 - t5

        qprint("(sampling, formatting counts,filling counts, energy estimation):",
                       (t_sampling, t_formatting, t_filling, t_energy))
        energy_estimated = energy_estimator

        self._energies_dictionary_corrected[experiment_key] = energy_estimated

        return energy_estimated

