"""
@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple, List

from tqdm import tqdm
import numpy as np

from qrem.functions_qrem import functions_data_analysis as fda, ancillary_functions as anf, quantum_ancillary_functions as quanf
from qrem.noise_mitigation.base_classes.EnergyEstimatorBase import EnergyEstimatorBase
from qrem.noise_model_generation.CTMP import functions_CTMP as fun_CTMP
from qrem.noise_simulation.CTMP import functions_CTMP_conditional as CTMP_cond, \
    functions_CTMP_simple as CTMP_simp


from qrem.common.printer import qprint

class EnergyCorrectorGlobalCTMP(EnergyEstimatorBase):
    """
    This is class for implementing quasiprobability-like error mitigation based on CTMP model [#TODO FBM: add ref].

    """


    def __init__(self,
                 results_dictionary: Dict[str, Dict[str, int]],
                 hamiltonian_dictionaries: Dict[str, Dict[Tuple[int], float]],
                 CTMP_rates_dictionary: Dict[Tuple[int], Dict[str, float]],
                 noise_strength_CTMP: Optional[float] = None,
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
        self._default_max_number_of_samples = 10**5

        self._number_of_qubits = len(list(list(list(results_dictionary.values())[0].keys())[0]))
        if noise_strength_CTMP is None:
            method_searching = "SDP_relaxation"
            if self._number_of_qubits<=20:
                method_searching='bruteforce'

            self._noise_strength_CTMP = fun_CTMP.find_optimal_noise_strength_CTMP(ctmp_rates_dictionary=CTMP_rates_dictionary,
                                                                                  number_of_qubits=self._number_of_qubits,
                                                                                  method_name=method_searching)

        self.__CTMP_generator_matrices = fun_CTMP.get_generators_dictionary_from_rates(
            rates_dictionary=CTMP_rates_dictionary)

    @property
    def CTMP_rates_dictionary(self) -> Dict[Tuple[int], Dict[str, float]]:
        return self._CTMP_rates_dictionary

    @CTMP_rates_dictionary.setter
    def CTMP_rates_dictionary(self, CTMP_rates_dictionary: Dict[Tuple[int], Dict[str, float]]) -> None:
        self._CTMP_rates_dictionary = CTMP_rates_dictionary


    def estimate_corrected_energy_CTMP_global(self,
                                              experiment_key=str,
                                              sampling_method='conditional',
                                              number_of_samples: Optional[int] = None,
                                              target_estimation_accuracy:Optional[float]=0.01,
                                              print_progress_bar=True
                                              ):
        """

        :param experiment_key:
        :type experiment_key:
        :param sampling_method:
        :type sampling_method:
        :param number_of_samples:
        :type number_of_samples:
        :param print_progress_bar:
        :type print_progress_bar:
        :return:
        :rtype:
        """

        # Choose sampling method.
        if sampling_method.upper() == 'CONDITIONAL':
            # If noise strength was not provided, calculate it here
            if self._noise_strength_CTMP is None:
                qprint("Noise strength not provided! Calculating now using SDP relaxation!")
                self._noise_strength_CTMP = fun_CTMP.find_optimal_noise_strength_CTMP(
                    ctmp_rates_dictionary=self._CTMP_rates_dictionary,
                    number_of_qubits=self._number_of_qubits,
                    method_name='SDP_relaxation',
                    method_kwargs={'hierarchy_level': 1})
                qprint('DONE!')
            noise_strength_CTMP = self._noise_strength_CTMP

        elif sampling_method.upper() == 'SIMPLE':
            noise_strength_CTMP = fun_CTMP.get_CTMP_error_rates_sum(
                rates_dictionary=self._CTMP_rates_dictionary)
            simple_noise_matrices_flat, probability_over_matrices_flat = CTMP_simp._get_inputs_to_sampling_algorithm(
                rates_dictionary=self._CTMP_rates_dictionary)

        else:
            raise ValueError(f"Sampling method '{sampling_method}' not recognized!")

        # theoretical_overhead = fun_CTMP.get_theoretical_overhead_CTMP(noise_strength_CTMP=noise_strength_CTMP)



        if number_of_samples is None:
            number_of_samples = fun_CTMP.get_theoretical_number_of_samples_CTMP(noise_strength_CTMP=noise_strength_CTMP,
                                                                                target_estimation_accuracy=target_estimation_accuracy)

            if number_of_samples > self._default_max_number_of_samples:
                number_of_samples = self._default_max_number_of_samples

        normalization_quasiprobability = np.exp(2 * noise_strength_CTMP)

        experimental_results = self._results_dictionary[experiment_key]
        hamiltonian_weights = self._hamiltonian_dictionaries[experiment_key]

        experimental_outcomes = list(experimental_results.keys())
        probability_distribution_over_outcomes = np.array(list(experimental_results.values()),
                                                          dtype=float)
        probability_distribution_over_outcomes /= np.sum(probability_distribution_over_outcomes)
        samples_uniform_outcomes = np.random.multinomial(n=number_of_samples,
                                                         pvals=probability_distribution_over_outcomes)

        outcomes_range = range(len(samples_uniform_outcomes))

        if print_progress_bar:
            outcomes_range = tqdm(outcomes_range)

        counts_dictionary = defaultdict(float)
        for index_outcome in outcomes_range:
            number_of_samples_from_outcome = samples_uniform_outcomes[index_outcome]
            bitstring_now = experimental_outcomes[index_outcome]
            matrix_powers_samples = np.random.poisson(lam=noise_strength_CTMP,
                                                      size=number_of_samples_from_outcome)

            for matrix_power in matrix_powers_samples:
                if (matrix_power % 2) == 0:
                    sign = 1.0
                else:
                    sign = -1.0

                if sampling_method.upper() == 'CONDITIONAL':
                    output_state = CTMP_cond.sample_from_power_of_B_matrix(
                        matrix_power=matrix_power,
                        input_state=bitstring_now,
                        generators_dictionary=self.__CTMP_generator_matrices,
                        noise_strength_CTMP=noise_strength_CTMP
                    )


                elif sampling_method.upper() == 'SIMPLE':
                    output_state = CTMP_simp.sample_from_power_of_B_matrix_simple(
                        global_input_state=bitstring_now,
                        matrix_power=matrix_power,
                        probability_over_matrices_list=probability_over_matrices_flat,
                        local_noise_matrices_list=simple_noise_matrices_flat)

                counts_dictionary[output_state] += sign

        energy_estimated = fda.estimate_energy_from_counts_dictionary(
            counts_dictionary=dict(counts_dictionary),
            weights_dictionary=hamiltonian_weights,
            normalize=False) * normalization_quasiprobability / number_of_samples

        self._energies_dictionary_corrected[experiment_key] = energy_estimated

        return energy_estimated



    def estimate_corrected_energy_CTMP_global_separable(self,
                                                        experiment_key: str,
                                                        clusters_list: List[Tuple[int]],
                                                        sampling_method='conditional',
                                                        number_of_samples: Optional[int] = None,
                                                        target_estimation_accuracy:Optional[float]=0.01,
                                                        print_progress_bar=True
                                                        ):
        """

        :param experiment_key:
        :type experiment_key:
        :param clusters_list:
        :type clusters_list:
        :param sampling_method:
        :type sampling_method:
        :param number_of_samples:
        :type number_of_samples:
        :param print_progress_bar:
        :type print_progress_bar:
        :return:
        :rtype:
        """

        # raise ValueError(f"Local sampling method not yet implemented")

        experimental_results = self._results_dictionary[experiment_key]
        hamiltonian_weights = self._hamiltonian_dictionaries[experiment_key]

        local_rates_dictionaries = self.__convert_to_local_noise_rates_dictionary(
            clusters_list=clusters_list)

        if sampling_method.upper() == 'SIMPLE':
            sampling_algorithm_inputs = {}
            for cluster, local_rates_dictionary in local_rates_dictionaries.items():
                sampling_algorithm_inputs[cluster] = CTMP_simp._get_inputs_to_sampling_algorithm(
                    rates_dictionary=local_rates_dictionary)

            noise_strength_method = 'SUM_OF_RATES'


        elif sampling_method.upper() == 'CONDITIONAL':

            local_generators_dictionary = {}
            for cluster, local_rates_dictionary in local_rates_dictionaries.items():
                local_generators_dictionary[cluster] = fun_CTMP.get_generators_dictionary_from_rates(
                    rates_dictionary=local_rates_dictionary)

            noise_strength_method = 'bruteforce'

        else:
            raise ValueError(
                f"Sampling method '{sampling_method}' unrecognized!\n Please use one of the following:"
                f" - 'SIMPLE'"
                f" - 'CONDITIONAL'")

        local_noise_strengths = self._calculate_noise_strengths_local(clusters_list=clusters_list,
                                                                      method=noise_strength_method)

        total_noise_strength = sum(list(local_noise_strengths.values()))
        normalization_quasiprobability = np.exp(2 * total_noise_strength)

        if number_of_samples is None:
            number_of_samples = fun_CTMP.get_theoretical_number_of_samples_CTMP(noise_strength_CTMP=total_noise_strength,
                                                                                target_estimation_accuracy=target_estimation_accuracy)

            if number_of_samples > self._default_max_number_of_samples:
                number_of_samples = self._default_max_number_of_samples







        experimental_outcomes = list(experimental_results.keys())
        probability_distribution_over_outcomes = np.array(list(experimental_results.values()),
                                                          dtype=float)
        probability_distribution_over_outcomes /= np.sum(probability_distribution_over_outcomes)
        samples_uniform_outcomes = np.random.multinomial(n=number_of_samples,
                                                         pvals=probability_distribution_over_outcomes)
        outcomes_range = range(len(samples_uniform_outcomes))

        if print_progress_bar:
            outcomes_range = tqdm(outcomes_range)

        counts_dictionary = defaultdict(float)
        for index_outcome in outcomes_range:
            number_of_samples_from_outcome = samples_uniform_outcomes[index_outcome]
            bitstring_now = experimental_outcomes[index_outcome]

            for _ in range(number_of_samples_from_outcome):
                global_output_state = [None for __ in range(self._number_of_qubits)]
                global_sign = 1
                for cluster in clusters_list:
                    # rates_dictionary_now = local_rates_dictionaries[cluster]
                    noise_strength_now = local_noise_strengths[cluster]

                    local_bitstring = [bitstring_now[qi] for qi in cluster]
                    local_matrix_power = np.random.poisson(lam=noise_strength_now,
                                                           size=1)[0]

                    if (local_matrix_power % 2) == 0:
                        global_sign *= 1.0
                    else:
                        global_sign *= -1.0

                    # print(rates_dictionary_now)

                    if sampling_method.upper() == 'CONDITIONAL':

                        generators_now = local_generators_dictionary[cluster]
                        local_output_state = CTMP_cond.sample_from_power_of_B_matrix(
                            matrix_power=local_matrix_power,
                            input_state=local_bitstring,
                            generators_dictionary=generators_now,
                            noise_strength_CTMP=noise_strength_now
                        )


                    elif sampling_method.upper() == 'SIMPLE':
                        local_output_state = CTMP_simp.sample_from_power_of_B_matrix_simple(
                            global_input_state=bitstring_now,
                            matrix_power=local_matrix_power,
                            probability_over_matrices_list=sampling_algorithm_inputs[cluster][1],
                            local_noise_matrices_list=sampling_algorithm_inputs[cluster][0])

                    for qubit_index in range(len(cluster)):
                        global_output_state[cluster[qubit_index]] = local_output_state[qubit_index]

                global_output_state = ''.join(global_output_state)
                counts_dictionary[global_output_state] += global_sign

        energy_estimated = fda.estimate_energy_from_counts_dictionary(
            counts_dictionary=dict(counts_dictionary),
            weights_dictionary=hamiltonian_weights,
            normalize=False) * normalization_quasiprobability / number_of_samples

        self._energies_dictionary_corrected[experiment_key] = energy_estimated

        return energy_estimated

