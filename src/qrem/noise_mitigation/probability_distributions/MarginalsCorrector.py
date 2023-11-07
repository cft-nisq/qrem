
from typing import Optional, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from qrem.noise_characterization.base_classes.marginals_analyzer_base import MarginalsAnalyzerBase
from qrem.functions_qrem import povmtools as pt

from qrem.common import probability
from qrem.common.printer import qprint #,qprint_array

class MarginalsCorrector(MarginalsAnalyzerBase):
    """
    This is the main class that uses correction data to reduce noise on the level of marginals_dictionary.
    Main functionalities are to correct marginal distributions of some experiments.

    NOTE: please see parent class MarginalsAnalyzerBase for conventions of experimental results storage

    """

    def __init__(self,
                 experimental_results_dictionary: Dict[str, Dict[str, int]],

                 correction_data_dictionary: dict,
                 bitstrings_right_to_left: Optional[bool] = False,
                 marginals_dictionary: Optional[Dict[str, Dict[Tuple[int], np.ndarray]]] = None
                 ) -> None:
        """
        :param experimental_results_dictionary: dictionary of results of experiments we wish to correct
        (see class' description for conventions)

        :param bitstrings_right_to_left: specify whether bitstrings
                                    should be read from right to left (when interpreting qubit labels)
        :param correction_data_dictionary: dictionary that contains information needed for noise
                                           mitigation on marginal probability distributions.


        :param marginals_dictionary: in case we pre-computed some marginal distributions
                                     (see class' description for conventions)

        """

        super().__init__(results_dictionary=experimental_results_dictionary,
                         bitstrings_right_to_left=bitstrings_right_to_left,
                         marginals_dictionary=marginals_dictionary
                         )
        if 'noise_matrices' in correction_data_dictionary.keys():
            self._noise_matrices = correction_data_dictionary['noise_matrices']
        else:
            qprint('No noise matrices provided!','','red')
        # self._correction_matrices = correction_data_dictionary['correction_matrices']
        if 'correction_matrices' in correction_data_dictionary.keys():
            self._correction_matrices = correction_data_dictionary['correction_matrices']
            # print('hejka')
        else:
            qprint('No correction matrices provided!','','red')



        self._correction_indices = correction_data_dictionary['correction_indices']

        self._corrected_marginals = {}

    @property
    def correction_indices(self) -> Dict[str, str]:
        return self._correction_indices

    @correction_indices.setter
    def correction_indices(self, correction_indices) -> None:
        self._correction_indices = correction_indices

    @property
    def corrected_marginals(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self._corrected_marginals

    def compute_marginals_of_marginals(self,
                                       keys_of_interest: List[str],
                                       corrected=True) -> \
            Dict[str, Dict[str, np.ndarray]]:
        """From dictionary of marginals_dictionary take only those which are in "marginals_labels_hamiltonian".
        Furthermore, for all keys, calculate also two-qubit and single-qubit marginals_dictionary for qubits
        inside those marginals_dictionary.

        :param keys_of_interest: list of strings representing qubit indices, e.g., 'q1q3q15'

        :return: marginals_of_interest : dictionary with marginal distributions for marginals_labels_hamiltonian
        """
        # self._corrected_marginals = dict(self._corrected_marginals)
        # self._marginals_dictionary = dict(self._marginals_dictionary)

        if corrected:
            marginals_dictionary = self._corrected_marginals
        else:
            marginals_dictionary = self._marginals_dictionary

        marginals_of_interest = {}

        for key in marginals_dictionary.keys():
            distribution_now = marginals_dictionary[key]
            if key in keys_of_interest:
                marginals_of_interest[key] = distribution_now

            qubits_here = key
            enumerated_qubits = dict(enumerate(key))
            rev_map = {}
            for kkk, vvv in enumerated_qubits.items():
                rev_map[vvv] = kkk

            
            #Here is one source of a problem

            # print(qubits_here)
            for qi in qubits_here:
                if (qi,) in keys_of_interest:# and not marginals_dictionary.keys():
                    marginals_of_interest[(qi,)] = \
                        self.get_marginal_from_probability_distribution(
                            distribution_now, [rev_map[qi]])

            for qi in qubits_here:
                for qj in qubits_here:
                    if (qi, qj) in keys_of_interest:# and not marginals_dictionary.keys():
                        marginals_of_interest[(qi, qj)] = self.get_marginal_from_probability_distribution(
                            distribution_now, sorted([rev_map[qi], rev_map[qj]]))

        if corrected:
            self._corrected_marginals = {**marginals_dictionary,**marginals_of_interest}

            #return marginals_of_interest
            return self._corrected_marginals
        else:
            self._marginals_dictionary = {**marginals_dictionary,**marginals_of_interest}

            return self._marginals_dictionary



    @staticmethod
    def correct_distribution_T_matrix(distribution: np.ndarray,
                                      correction_matrix: np.ndarray,
                                      ensure_physicality=True,
                                      norm_minimization='l2'):
        """Correct probability distribution using multiplication via inverse of noise matrix.
        See Refs. [0], [0.5].

        :param distribution: noisy distribution
        :param correction_matrix: correction matrix (inverse of noise matrix)
        :param ensure_physicality: if True, then after correcting marginal distribution it ensures that
                                    resulting vector has elements from [0,1]

        :return: array representing corrected distribution

        #TODO FBM: add option to return mitigation errors
        """

        # Consistent formatting

        #JT: in the definition distribution is specified as np.ndarray, do we need the ib below?
        if isinstance(distribution, list):
            d = len(distribution)
        elif isinstance(distribution, np.ndarray):
            d = distribution.shape[0]
        else:
            raise TypeError("Wrong distribution format")

        distribution = np.array(distribution).reshape(d, 1)

        # correct distribution using inverse of noise matrix
        quasi_distribution = correction_matrix.dot(distribution)

        # do we want to ensure resulting distribution is physical?
        if ensure_physicality:
            # if yes, check if it is physical
            if probability.is_valid_probability_vector(quasi_distribution):
                # if it is physical, no further action is required
                return quasi_distribution
            else:
                # if it is not physical, find closest physical one
                if norm_minimization.lower() in ['l2']:
                    return np.array(pt.find_closest_prob_vector_l2(quasi_distribution)).reshape(d, 1)
                elif norm_minimization.lower() in ['l1']:
                    return np.array(pt.find_closest_prob_vector_l1(quasi_distribution)).reshape(d, 1)
        else:
            # if we don't care about physicality, we don't do anything and just return vector
            return quasi_distribution

    @staticmethod
    def correct_distribution_IBU(estimated_distribution: np.ndarray,
                                 noise_matrix: np.ndarray,
                                 iterations_number: Optional[int] = 10,
                                 prior: Optional[np.ndarray] = None):
        """Correct probability distribution using Iterative Bayesian Unfolding (IBU)
        See Ref. [1]

        :param estimated_distribution: noisy distribution (to be corrected)
        :param noise_matrix: noise matrix
        :param iterations_number: number of iterations in unfolding
        :param prior: ansatz for ideal distribution, default is uniform

        :return: array representing corrected distribution
        """

        # Consistent formatting
        if isinstance(estimated_distribution, list):
            d = len(estimated_distribution)
        elif isinstance(estimated_distribution, np.ndarray):
            d = estimated_distribution.shape[0]
        else:
            raise TypeError("Wrong distribution format")

        # default iterations number
        if iterations_number is None:
            iterations_number = 10

        # If no prior is provided, we take uniform distribution
        if prior is None:
            prior = np.full((d, 1), 1 / d, dtype=float)
        elif isinstance(prior,str):
            if prior in ['uniform']:
                prior = np.full((d, 1), 1 / d, dtype=float)

            # max_index = np.argmax(estimated_distribution)
            # prior = np.ones((d,1))
            # prior[max_index] = 10
            # prior/=np.sum(prior)


        # raise KeyboardInterrupt
        # initialize distribution for recursive loop
        distribution_previous = prior
        # print(noise_matrix)
        # go over iterations
        for iteration_index in range(iterations_number):
            distribution_new = np.zeros((d, 1),
                                        dtype=float)

            # go over measurement outcomes
            for outcome_index in range(d):
                # calculating IBU estimate
                distribution_new[outcome_index] = sum(
                    [noise_matrix[j, outcome_index] * distribution_previous[outcome_index] *
                     estimated_distribution[j] / sum(
                        [noise_matrix[j, k] * distribution_previous[k] for k in range(d)])
                     for j in
                     range(d)])
            # update distribution for recursion
            distribution_previous = distribution_new

        return distribution_previous

    # @staticmethod
    def correct_distribution_hybrid_T_IBU(self,
                                          estimated_distribution,
                                          noise_matrix,
                                          correction_matrix: Optional[np.ndarray] = None,
                                          unphysicality_threshold: Optional[float] = 0.0,
                                          iterations_number: Optional[int] = None,
                                          prior: Optional[np.ndarray] = None):
        """
        Correct distribution using method_name that is hybrid of T_matrix correction (see Refs. [0,0.5])
        and Iterative Bayesian Unfolding (IBU) (see Ref. [1]).

        Algorithm goes like this:
        - Correct distribution using inverse of noise matrix.
        - If distribution is physical (i_index.e., elements are from (0,1)), return it.
        - Otherwise, perform IBU.
        - [Optional] if parameter "unphysicality_threshold" is provided,
           then it projects the result of "T_matrix" correction onto probability simplex
           and if Total Variation Distance between this projected vector
           and original unphysical one is belowthis cluster_threshold, then it returns the projection.
           Otherwise, it performs IBU.

        :param estimated_distribution: noisy distribution (to be corrected)
        :param noise_matrix: noise matrix
        :param correction_matrix: correction matrix (inverse of noise matrix)
                                 if not provided, it is calculated from noise_matrix
        :param unphysicality_threshold: cluster_threshold to decide whether unphysical "T_matrix"
                                        correction is acceptable. See description of the function
        :param iterations_number: number of iterations in IBU
        :param prior: ansatz for ideal distribution in IBU, default is uniform

        """

        if isinstance(estimated_distribution, list):
            d = len(estimated_distribution)
        elif isinstance(estimated_distribution, np.ndarray):
            d = estimated_distribution.shape[0]
        else:
            raise TypeError("Wrong distribution format")

        if correction_matrix is None:
            correction_matrix = np.linalg.inv(noise_matrix)

        distribution = np.array(estimated_distribution).reshape(d, 1)

        quasi_distribution = correction_matrix.dot(distribution)

        is_physical = probability.is_valid_probability_vector(quasi_distribution)

        if is_physical:
            return quasi_distribution
        else:
            if unphysicality_threshold == 0:
                closest_physical = pt.find_closest_prob_vector_l2(quasi_distribution)
                return self.correct_distribution_IBU(distribution,
                                                     noise_matrix,
                                                     iterations_number,
                                                     prior=closest_physical)
            else:
                closest_physical = pt.find_closest_prob_vector_l2(quasi_distribution)

                distance_now = 1 / 2 * np.linalg.norm(quasi_distribution - closest_physical, ord=1)
                # distance_now = 1 / 2 * np.linalg.norm(quasi_distribution - closest_physical, ord=1)



                #TODO FBM: original
                if distance_now <= unphysicality_threshold:
                    return closest_physical

                #
                # if distance_now >= unphysicality_threshold:
                #     return closest_physical

                else:
                    return self.correct_distribution_IBU(distribution,
                                                         noise_matrix,
                                                         iterations_number,
                                                         prior=closest_physical)

    def correct_marginals(self,
                          marginals_dictionary: Optional[Dict[str, np.ndarray]] = None,
                          method='lambda_inverse',
                          method_kwargs=None) -> Dict[str, Dict[str, np.ndarray]]:

        """Return dictionary of corrected marignal distributions
        :param marginals_dictionary: dictionary of (noisy) marginal distributions
        :param method: method_name to be used for correction of marginal probability distributions

        possible values:
        - 'T_matrix' - uses inverse of noise matrix as correction (see Refs. [0,0.5])
        - 'IBU' - uses Iterative Bayesian Unfolding (see Ref. [1])
        - 'hybrid_T_IBU' - uses hybrid method_name between 'T_matrix' and 'IBU',
                          see description of self.correct_distribution_hybrid_T_IBU

        :param method_kwargs:  keyword arguments passed to function using chosen method_name.
                               See description of specific functions_qrem.

        :return: corrected_marginals : dictionary of corrected marginal distributions
        """

        # TODO FBM: make it consistent with conventions used in parent class
        if marginals_dictionary is None:
            marginals_dictionary = self._marginals_dictionary

        corrected_marginals = {}

        for key in marginals_dictionary.keys():

            noisy_marginal_now = marginals_dictionary[key]

            # print(key)
            # qprint_array(self._correction_matrices[key])
            # qprint_array(noisy_marginal_now)

            if method.lower() in ['lambda_inverse',
                                  # 'linear_inversion',
                                  't_matrix']:
                if method_kwargs is None:
                    method_kwargs = {'ensure_physicality': True,
                                     'norm_minimization':'l2'}
                else:
                    if 'ensure_physicality' not in method_kwargs.keys():
                        method_kwargs['ensure_physicality'] = True

                    # for unwanted_key in

                corrected_marginal_now = self.correct_distribution_T_matrix(noisy_marginal_now,
                                                                            self._correction_matrices[
                                                                                key],
                                                                            **method_kwargs)
            elif method.lower() in  ['ibu']:
                if method_kwargs is None:
                    method_kwargs = {'iterations_number': 2,
                                     'prior': None}
                else:
                    if 'iterations_number' not in method_kwargs.keys():
                        method_kwargs['iterations_number'] = 2
                    if 'prior' not in method_kwargs.keys():
                        method_kwargs['prior']=None


                #TODO FBM: some problems with KEY in self._noise_matrices
                corrected_marginal_now = self.correct_distribution_IBU(noisy_marginal_now,
                                                                       self._noise_matrices[
                                                                           key]['averaged'],
                                                                       **method_kwargs)
            elif method.lower() in ['hybrid_lambda_ibu']:
                if method_kwargs is None:
                    method_kwargs = {'unphysicality_threshold': 0.0,
                                     'iterations_number': 2,
                                     'prior': None}
                else:
                    if 'iterations_number' not in method_kwargs.keys():
                        method_kwargs['iterations_number'] = 2
                    if 'prior' not in method_kwargs.keys():
                        method_kwargs['prior'] = None
                    if 'unphysicality_threshold' not in method_kwargs.keys():
                        method_kwargs['unphysicality_threshold'] = 0.0

                corrected_marginal_now = self.correct_distribution_hybrid_T_IBU(noisy_marginal_now,
                                                                                self._noise_matrices[
                                                                                    key]['averaged'],
                                                                                self._correction_matrices[
                                                                                    key],
                                                                                **method_kwargs)
            else:
                raise ValueError('Wrong method_name name')

            corrected_marginals[key] = corrected_marginal_now

        self._corrected_marginals = corrected_marginals
        return corrected_marginals
