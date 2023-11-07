import copy
from typing import Optional, Dict, Union, Tuple

import numpy as np
import scipy as sc
from tqdm import tqdm

from qrem.functions_qrem import ancillary_functions as anf
from qrem.noise_characterization.tomography_design.overlapping.DOTMarginalsAnalyzer import DOTMarginalsAnalyzer
from qrem.noise_model_generation.CTMP import functions_CTMP as fun_CTMP

import qrem.common.utils as qrem_utils
# from qrem.common.printer import qprint_array, qprint

class ModelGeneratorCTMP(DOTMarginalsAnalyzer):

    # TODO Significant refactoring needed
    #

    def __init__(self,
                 results_dictionary_ddot: Dict[str, Dict[str, int]],

                 number_of_qubits: int,
                 bitstrings_right_to_left: Optional[bool] = False,
                 marginals_dictionary: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                 noise_matrices_dictionary: Optional[
                     Dict[str, Union[np.ndarray, Dict[str, Dict[str, np.ndarray]]]]] = None,
                 pairs_complements_dictionary=None
                 ) -> None:
        """

        :param results_dictionary_ddot:
        :type results_dictionary_ddot:
        :param bitstrings_right_to_left:
        :type bitstrings_right_to_left:
        :param number_of_qubits:
        :type number_of_qubits:
        :param marginals_dictionary:
        :type marginals_dictionary:
        :param noise_matrices_dictionary:
        :type noise_matrices_dictionary:

        :param pairs_complements_dictionary:
        Dictionary with structure:

        key: pair tuple
        value: list/tuple of qubits which are correlated to it

        For given clusters' structure, this can be generated via function:
        noise_model_generation.CTMP.fucntions_CTMP.get_pairs_complement_dictionary_from_clusters_list


        :type pairs_complements_dictionary:
        """

        super().__init__(results_dictionary_ddot,
                         bitstrings_right_to_left,
                         marginals_dictionary,
                         noise_matrices_dictionary
                         )
        self._number_of_qubits = number_of_qubits
        self._qubit_indices = list(range(number_of_qubits))

        self._correlations_table_pairs = None
        self._rates_dictionary = {}

        if pairs_complements_dictionary is None:
            pairs_complements_dictionary = {}
            single_qubits = list(range(self._number_of_qubits))
            for qi in range(self._number_of_qubits):
                for qj in range(qi + 1, self._number_of_qubits):
                    pair = (qi, qj)
                    pair_complement = sorted(list(set(single_qubits).difference(set(pair))))
                    pairs_complements_dictionary[pair] = pair_complement

        self._pairs_complements_dictionary = pairs_complements_dictionary
        self._noise_strength_CTMP = None

        if noise_matrices_dictionary is None:
            # single_qubits = list(range(number_of_qubits))
            # pairs = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]

            self._noise_matrices_dictionary = {}

        else:
            self._noise_matrices_dictionary = noise_matrices_dictionary

    @property
    def rates_dictionary(self) -> Dict[Tuple[int], Dict[str, float]]:
        return self._rates_dictionary

    @rates_dictionary.setter
    def rates_dictionary(self, rates_dictionary: Dict[Tuple[int], Dict[str, float]]) -> None:
        self._rates_dictionary = rates_dictionary


    def find_optimal_noise_strength_CTMP(self):
        method_searching = "SDP_relaxation"
        if self._number_of_qubits<=22:
            method_searching='bruteforce'
        self._noise_strength_CTMP = fun_CTMP.find_optimal_noise_strength_CTMP(ctmp_rates_dictionary=self._rates_dictionary,
                                                                              number_of_qubits=self._number_of_qubits,
                                                                              method_name=method_searching)
        return self._noise_strength_CTMP


    def _get_local_noise_matrix_postselection(self,
                                              pair):

        # single_qubits = list(range(self._number_of_qubits))
        # print(pair)
        pair_complement = self._pairs_complements_dictionary[pair]
        # print(pair_complement)

        local_noise_matrix = np.zeros((4,4), dtype=float)
        for global_input_state, counts_dict in self._results_dictionary.items():
            marginal_input_state = ''.join([global_input_state[x] for x in pair])
            input_state_complement = [global_input_state[x] for x in pair_complement]

            # print(global_input_state)
            for global_output_state, ticks in counts_dict.items():
                marginal_output_state = ''.join([global_output_state[x] for x in pair])
                output_state_complement = [global_output_state[x] for x in pair_complement]

                # print(global_output_state)
                if output_state_complement == input_state_complement:
                    # print(output_state_complement,input_state_complement,ticks)
                    # check if this is their convention!

                    local_noise_matrix[int(marginal_output_state, 2),
                                       int(marginal_input_state, 2)] += ticks
        # raise KeyboardInterrupt
        # qprint_array(local_noise_matrix)
        # normalize to stochastic matrix
        for k in range(4):
            sum_column = sum(local_noise_matrix[:, k])
            if sum_column==0:
                local_noise_matrix[k,k] = 1
            else:
                local_noise_matrix[:, k] /= sum(local_noise_matrix[:, k])

        # qprint_array(local_noise_matrix)
        # raise KeyboardInterrupt
        return local_noise_matrix

    def _get_local_noise_matrix_postselection_1q(self,
                                                 qubit):

        local_noise_matrix = np.zeros((2, 2), dtype=float)
        for global_input_state, results_dictionary in self._results_dictionary.items():
            marginal_input_state = global_input_state[qubit]

            for global_output_state, ticks in results_dictionary.items():
                marginal_output_state = global_output_state[qubit]

                local_noise_matrix[int(marginal_output_state, 2),
                                   int(marginal_input_state, 2)] += ticks
        # normalize to stochastic matrix
        for k in range(2):
            local_noise_matrix[:, k] /= sum(local_noise_matrix[:, k])

        return local_noise_matrix

    def _get_local_noise_matrix(self,
                                pair,
                                method='postselection'):

        if method.upper() == 'POSTSELECTION':
            return self._get_local_noise_matrix_postselection(pair=pair)
        else:
            raise ValueError(f"Method '{method}' not yet supported")

    def _calculate_multiple_noise_matrices(self,
                                           pairs_list,
                                           method='postselection'):
        pairs_range = copy.deepcopy(pairs_list)
        if len(pairs_list)>50:
            pairs_range = tqdm(pairs_range)
            # print(pairs_list)

        for pair in pairs_range:
            local_noise_matrix = self._get_local_noise_matrix(pair=pair,
                                                              method=method)
            # print(local_noise_matrix,'hej')

            self._noise_matrices_dictionary[pair] = local_noise_matrix


    # @staticmethod
    # def _get_1q_G_matrix(stochastic_map):
    #
    #     epsilon = stochastic_map[1,0]
    #     eta = stochastic_map[0,1]
    #
    #     G_matrix = -np.log(1-epsilon-eta)/(epsilon)
    #

    def _get_CTMP_rates_from_results_postselection(self,
                                                   pairs_list=None
                                                   ):

        number_of_qubits = self._number_of_qubits
        single_qubits = self._qubit_indices

        if pairs_list is None:
            pairs_list = [(qi, qj) for qi in single_qubits for qj in single_qubits if qj > qi]

        # print(pairs_list)


        qubits_in_pairs = []
        for pair in pairs_list:
            for qubit in pair:
                qubits_in_pairs.append(qubit)
        unique_qubits_in_pairs = sorted(list(np.unique(qubits_in_pairs)))
        unique_qubits_alone = qrem_utils.lists_difference(list(range(number_of_qubits)),
                                                   unique_qubits_in_pairs)




        self._calculate_multiple_noise_matrices(pairs_list=pairs_list,
                                                method='postselection')

        # Get G matrices
        G_matrices = {pair: sc.linalg.logm(self._noise_matrices_dictionary[pair]) for pair in
                      pairs_list}

        # ancillary function
        def __chop_negatives(M):
            (m, n) = M.shape

            chopped_M = copy.deepcopy(M)
            for i in range(m):
                for j in range(n):
                    if i != j and M[i, j] < 0:
                        chopped_M[i, j] = 0

            return chopped_M

        # Get G' matrices
        G_prime_matrices = {pair: __chop_negatives(G_matrices[pair]) for pair in pairs_list}


        rates_dictionary_1q_new_format = {(q,): {'0': 0, '1': 0} for q in single_qubits}


        for qj in unique_qubits_in_pairs:
            # qprint("hey",qj)
            r0, r1 = 0, 0

            counter_normalization = 0
            for q_other in unique_qubits_in_pairs:
                if q_other != qj:
                    # qprint("hey", q_other,'red')

                    if tuple(sorted([qj, q_other])) in G_prime_matrices.keys():
                        # try:
                        G_prime_matrix_now = G_prime_matrices[tuple(sorted([qj, q_other]))]

                        if qj<q_other:

                            r0 += G_prime_matrix_now[2, 0] + G_prime_matrix_now[3, 1]
                            r1 += G_prime_matrix_now[0, 2] + G_prime_matrix_now[1, 3]
                        else:
                            r0 += +G_prime_matrix_now[1,0]+G_prime_matrix_now[3,2]
                            r1 += G_prime_matrix_now[0, 1] + G_prime_matrix_now[2, 3]

                        counter_normalization+=2
                        # except(KeyError):
                            #     pass


            r0 /= counter_normalization
            r1 /= counter_normalization

            rates_dictionary_1q_new_format[(qj,)]['0'] = r0
            rates_dictionary_1q_new_format[(qj,)]['1'] = r1


        # for averaged_1q matrix
        G_matrices_1q = {
            (qi,): sc.linalg.logm(self._get_local_noise_matrix_postselection_1q(qubit=qi))
            for qi in unique_qubits_alone}
        G_prime_matrices_1q = {qi: __chop_negatives(G_matrix) for qi, G_matrix in
                               G_matrices_1q.items()}



        for qi, G_prime_1q in G_prime_matrices_1q.items():
            r0 = G_prime_1q[1, 0]
            r1 = G_prime_1q[0, 1]

            rates_dictionary_1q_new_format[qi]['0'] = r0
            rates_dictionary_1q_new_format[qi]['1'] = r1



        rates_dictionary_2q_new_format = {pair: {'00': 0,
                                                 '01': 0,
                                                 '10': 0,
                                                 '11': 0} for pair in pairs_list}

        for pair in pairs_list:
            G_prime_matrix_now = G_prime_matrices[pair]

            rates_dictionary_2q_new_format[pair]['00'] = G_prime_matrix_now[3, 0]
            rates_dictionary_2q_new_format[pair]['01'] = G_prime_matrix_now[2, 1]
            rates_dictionary_2q_new_format[pair]['10'] = G_prime_matrix_now[1, 2]
            rates_dictionary_2q_new_format[pair]['11'] = G_prime_matrix_now[0, 3]

        rates_dictionary_new_format = {**rates_dictionary_1q_new_format,
                                       **rates_dictionary_2q_new_format}

        return rates_dictionary_new_format

    def get_CTMP_rates_from_results(self,
                                    pairs_list=None,
                                    method='postselection'):

        if method.upper() == 'POSTSELECTION':
            rates_dictionary = self._get_CTMP_rates_from_results_postselection(pairs_list=pairs_list)
        else:
            raise ValueError(f"Method '{method}' is not yet supported.")

        self._rates_dictionary = rates_dictionary

        return self._rates_dictionary




