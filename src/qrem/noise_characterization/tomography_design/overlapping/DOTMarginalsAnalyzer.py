import copy
from collections import defaultdict
from typing import Optional, Dict, List, Union, Tuple

import numpy as np
from tqdm import tqdm


 
from qrem.noise_characterization.base_classes.marginals_analyzer_base import MarginalsAnalyzerBase

import qrem.common.math as qrem_math
from qrem.common import utils, convert
from qrem.common.printer import qprint, qprint_array
import qrem.common. utils as qrem_utils


#(PP): DELETE, should be removed from package everywhere it is used (no human input needed rule)
def query_yes_no(question):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    yes_answers = {'yes',
                   'y',
                   'ye',
                   'tak',
                   'sure',
                   'of course',
                   'Yes',
                   'yeah'}
    no_answers = {'no',
                  'n',
                  'nope',
                  'nah',
                  'nie',
                  'noo',
                  'nooo',
                  'noooo',
                  'No'}

    choice = 0
    print(question + ' [y/n] ')
    choice = input().lower()
    if choice in yes_answers:
        return True
    elif choice in no_answers:
        return False
    else:
        qprint('Please:', "respond with 'yes' or 'no'")
        return query_yes_no(question)


class DOTMarginalsAnalyzer(MarginalsAnalyzerBase):
    """
        Class that handles results of Diagonal Detector Tomography.
        Main functionalities allow to calculate noise matrices on subsets_list of qubits.
        This includes averaged noise matrices, i_index.e., averaged over states off all other qubits,
        as well as state-dependent, i_index.e., conditioned on the particular
        input classical state of some other qubits.

        In this class, and all its children, we use the following convention
        for storing marginal noise matrices:

        :param noise_matrices_dictionary: nested dictionary with following structure:

        noise_matrices_dictionary[qubits_subset_string]['averaged']
        = average noise matrix on qubits subset
        and
        noise_matrices_dictionary[qubits_subset_string][other_qubits_subset_string][input_state_bitstring]
        = noise matrix on qubits subset depending on input state of other qubits.

        where:
        - qubits_subset_string - is string labeling qubits subset (e.g., 'q1q2q15...')
        - other_qubits_subset_string - string labeling other subset
        - input_state_bitstring - bitstring labeling
                                                   input state of qubits in other_qubits_subset_string

    """

    def __init__(self,
                 results_dictionary_ddot: Dict[str, Dict[str, int]],
                 bitstrings_right_to_left: Optional[bool] = False,
                 marginals_dictionary: Optional[Dict[str, Dict[Tuple[int], np.ndarray]]] = None,
                 noise_matrices_dictionary: Optional[
                     Dict[Tuple[int], Dict[Union[str,Tuple[int]], Dict[str, np.ndarray]]]] = None
                 ) -> None:

        """
        :param results_dictionary_ddot: see description of MarginalsAnalyzerBase.
        Here we use classical input states (bitstrings) of qubits as LABELS for experiments.
        :param bitstrings_right_to_left: specify whether bitstrings
                                 should be read from right to left (when interpreting qubit labels)
        :param marginals_dictionary: see description of MarginalsAnalyzerBase.
        :param noise_matrices_dictionary: nested dictionary with following structure:
        """

        super().__init__(results_dictionary=results_dictionary_ddot,
                         bitstrings_right_to_left=bitstrings_right_to_left,
                         marginals_dictionary=marginals_dictionary
                         )


        self.__fill_missing_columns = False



        if noise_matrices_dictionary is None:
            noise_matrices_dictionary = {}
            # FBM: Make sure whether this helps in anything
            # FBM: (because we anyway perform checks in the functions later)
            if marginals_dictionary is not None:
                for experiment_key, dictionary_of_marginals in marginals_dictionary.items():
                    for marginal_key in dictionary_of_marginals.keys():
                        if marginal_key not in noise_matrices_dictionary.keys():
                            noise_matrices_dictionary[marginal_key] = {}

        self._noise_matrices_dictionary = noise_matrices_dictionary

    @property
    def noise_matrices_dictionary(self) -> Dict[
        Tuple[int], Dict[Union[str, Tuple[int]], Dict[str, np.ndarray]]]:
        return self._noise_matrices_dictionary

    @noise_matrices_dictionary.setter
    def noise_matrices_dictionary(self,
                                  noise_matrices_dictionary: Dict[Tuple[int],
                                                                  Dict[Union[str, Tuple[int]],
                                                                       Dict[
                                                                           str, np.ndarray]]] = None) -> None:

        self._noise_matrices_dictionary = noise_matrices_dictionary

    @staticmethod
    def get_noise_matrix_from_counts_dict(
            results_dictionary: Union[Dict[str, np.ndarray], defaultdict]) -> np.ndarray:
        """Return noise matrix from counts dictionary.
        Assuming that the results are given only for qubits of interest.
        :param results_dictionary: dictionary with experiments of the form:

        results_dictionary[input_state_bitstring] = probability_distribution

        where:
        - input_state_bitstring is bitstring denoting classical input state
        - probability_distribution - estimated vector of probabilities for that input state

        :return: noise_matrix: the array representing noise on qubits
        on which the experiments were performed
        """

        number_of_qubits = len(list(results_dictionary.keys())[0])
        # print('hej',number_of_qubits)
        noise_matrix = np.zeros((2 ** number_of_qubits, 2 ** number_of_qubits))
        for input_state, probability_vector in results_dictionary.items():
            numbers_input = [int(x) for x in list(input_state)]

            #FBM: this should never happen, right? Control it
            if np.any(np.array(numbers_input) > 1):
                continue


            try:
                noise_matrix[:, int(input_state, 2)] = probability_vector[:]
            except(ValueError):
                noise_matrix[:, int(input_state, 2)] = probability_vector[:, 0]



        return noise_matrix

    @staticmethod
    def average_noise_matrices_over_some_qubits(matrices_cluster: Dict[str, np.ndarray],
                                                all_neighbors: List[int],
                                                qubits_to_be_left: List[int]) -> Dict[str, np.ndarray]:
        """
          Given dictionary of noise matrices, average them over some qubits.

         :param matrices_cluster: dictionary for which KEY is classical INPUT state of neighbors,
                                  and VALUE is stochastic_matrix noise matrix
         :param all_neighbors: list of neighbors of given cluster
         :param qubits_to_be_left: qubits which we are interested in and we do not average over them

         :return: dictionary of noise matrices
                    depending on the state of neighbors MINUS qubits_to_be_averaged_over

         """

        if all_neighbors is None or len(all_neighbors) == 0:
            return {'averaged': matrices_cluster['averaged']}

        reversed_enumerated = utils.map_index_to_order(all_neighbors)
        averaging_normalization = int(2 ** (len(all_neighbors) - len(qubits_to_be_left)))

        states_after_averaging = povmtools.get_classical_register_bitstrings(range(len(qubits_to_be_left)),
                                                                       len(qubits_to_be_left), False)
        averaged_dimension = list(matrices_cluster.values())[0].shape[0]

        averaged_matrices_cluster = {
            state: np.zeros((averaged_dimension, averaged_dimension), dtype=float) for state in
            states_after_averaging}

        qubits_to_be_averaged_over = list(set(all_neighbors).difference(set(qubits_to_be_left)))
        qubits_to_be_averaged_over_mapped = [reversed_enumerated[q_index] for q_index in
                                             qubits_to_be_averaged_over]

        for neighbors_state, conditional_noise_matrix in matrices_cluster.items():
            list_string_neighbors = list(copy.deepcopy(neighbors_state))

            list_string_neighbors_to_be_left = list(np.delete(list_string_neighbors,
                                                              qubits_to_be_averaged_over_mapped))

            string_neighbors = ''.join(list_string_neighbors_to_be_left)

            averaged_matrices_cluster[string_neighbors] += conditional_noise_matrix / averaging_normalization

        return averaged_matrices_cluster

    def _compute_noise_matrix_averaged(self,
                                       subset: Tuple[int]) -> np.ndarray:
        """Noise matrix for subset of qubits, averaged over all other qubits

            :param subset: subset of qubits we are interested in

           By default takes data from self._marginals_dictionary. If data is not present, then it
           calculates marginals_dictionary for given subset
           and updates the class's property self.marginals_dictionary
        """

        # FBM: Perhaps add possibility of using existing marginals_dictionary for bigger subset that includes target subset
        #

        # subset_key = 'q' + 'q'.join([str(s) for s in subset])
        subset_key = tuple(subset)



        marginal_dict_now = self.get_averaged_marginal_for_subset(subset)

        # print('hej')
        # print(marginal_dict_now)
        noise_matrix_averaged = self.get_noise_matrix_from_counts_dict(marginal_dict_now)


        # fill_missing = False

        if not qrem_math.is_matrix_stochastic(noise_matrix_averaged):
            # qprint_array(noise_matrix_averaged)
            message_now = f"\nNoise matrix not stochastic for subset: {subset}.\n" \
                          f"This most likely means that DDOT collection was not complete " \
                          f"for locality {len(subset)} and some " \
                          f"states were not implemented."
            qprint(message_now,'','red')
            qprint("\nThat matrix looks like this:")
            qprint_array(noise_matrix_averaged)

            print('\n')
            if self.__fill_missing_columns:
                qprint("\nAdding missing columns.",'','red')
                for column_index in range(noise_matrix_averaged.shape[0]):
                    column_now = noise_matrix_averaged[:, column_index]

                    if np.all(column_now == 0):
                        noise_matrix_averaged[column_index, column_index] = 1.0

                qprint("\nNow the matrix looks like this:")
                qprint_array(noise_matrix_averaged)

            elif query_yes_no("Do you wish to assume that missing columns are ideal?\n"
                                "Otherwise, we will raise ValueError."):

                for column_index in range(noise_matrix_averaged.shape[0]):
                    column_now = noise_matrix_averaged[:,column_index]
                    
                    if np.all(column_now==0):
                        noise_matrix_averaged[column_index,column_index] = 1.0

                qprint("\nNow the matrix looks like this:")
                qprint_array(noise_matrix_averaged)


                if query_yes_no("Do you wish to fill all missing columns in the future?\n"
                                    "Otherwise, we will ask each time where there's need."):
                    self.__fill_missing_columns = True
                    qprint("OK, from now on we will assume missing columns are ideal.\n")
                else:
                    qprint("OK, we will ask again if needed.")

            else:
                raise ValueError()

        if subset_key in self._noise_matrices_dictionary.keys():
            self._noise_matrices_dictionary[subset_key]['averaged'] = noise_matrix_averaged
        else:
            self._noise_matrices_dictionary[subset_key] = {'averaged': noise_matrix_averaged}

        return noise_matrix_averaged

    def get_noise_matrix_averaged(self,
                                  subset: Tuple[int]) -> np.ndarray:
        """
            Like self._compute_noise_matrix_averaged but if matrix is already in class' property,
            does not calculate it again.

            :param subset: subset of qubits we are interested in
        """
        # subset_key = 'q' + 'q'.join([str(s) for s in subset])
        subset_key = tuple(subset)

        # print('hej',subset)
        # print(self._noise_matrices_dictionary)
        if subset_key in self._noise_matrices_dictionary.keys():
            if 'averaged' in self._noise_matrices_dictionary[subset_key].keys():
                return self._noise_matrices_dictionary[subset_key]['averaged']
            else:
                return self._compute_noise_matrix_averaged(subset)
        else:
            return self._compute_noise_matrix_averaged(subset)
#BUG WARNING
    def _compute_noise_matrix_dependent(self,
                                        qubits_of_interest: Tuple[int],
                                        neighbors_of_interest: Union[Tuple[int], None]) \
            -> Dict[str, np.ndarray]:
        """Return lower-dimensional effective noise matrices acting on qubits_of_interest"
                    conditioned on input states of neighbors_of_interest
            :param qubits_of_interest: labels of qubits in marginal  we are interested in
            :param neighbors_of_interest: labels of qubits that affect noise matrix on qubits_of_interest

            :return conditional_noise_matrices_dictionary: dictionary with structure

            conditional_noise_matrices_dictionary['averaged'] =
            noise matrix on qubits_of_interest averaged over input states of other qubits

            and

            conditional_noise_matrices_dictionary[input_state_neighbors_bitstring] =
            noise matrix on qubits_of_interest conditioned on input state of neighbors being
            input_state_neighbors_bitstring

        """

        # If there are no all_neighbors,
        # then this corresponds to averaging over all qubits except qubits_of_interest
        if len(neighbors_of_interest) == 0 or neighbors_of_interest is None:
            cluster_tuple_now = convert.qubit_indices_to_keystring(qubits_of_interest)
            cluster_tuple_now = tuple(qubits_of_interest)
            if 'averaged' in self._noise_matrices_dictionary[cluster_tuple_now].keys():
                return {'averaged': self._noise_matrices_dictionary[cluster_tuple_now]['averaged']}
            else:
                noise_matrix = self.get_noise_matrix_averaged(qubits_of_interest)
                return {'averaged': noise_matrix}

        # check if there is no collision between qubits_of_interest and neighbors_of_interest
        # (if there is, then the method_name won't be consistent)
        if len(utils.lists_intersection(qubits_of_interest, neighbors_of_interest)) != 0:
            print(qubits_of_interest, neighbors_of_interest)
            raise ValueError('Qubits of interest and neighbors overlap')

        # first, get averaged noise matrix on qubits of interest and all_neighbors of interest
        # FBM: make sure that qubit indices are correct (I think they are)
        qubits_of_interest = list(qubits_of_interest)
        neighbors_of_interest = list(neighbors_of_interest)
        all_qubits = sorted(qubits_of_interest + neighbors_of_interest)
        all_qubits_enumerated = utils.map_index_to_order(all_qubits)

        # we will get noise matrix on all of the qubits first, and then we will process it to get
        # conditional marginal noise matrices on qubits_of_interest
        big_lambda = self.get_noise_matrix_averaged(tuple(all_qubits))

        total_number_of_qubits = int(np.log2(big_lambda.shape[0]))
        total_dimension = int(2 ** total_number_of_qubits)
        number_of_qubits_of_interest = len(qubits_of_interest)
        number_of_neighbors = len(neighbors_of_interest)

        # Normalization when averaging over states of non-neighbours (each with the same probability)
        normalization = 2 ** (
                total_number_of_qubits - number_of_neighbors - number_of_qubits_of_interest)

        # classical register on all qubits
        classical_register_all_qubits = ["{0:b}".format(i).zfill(total_number_of_qubits) for i in
                                         range(total_dimension)]

        # classical register on neighbours
        classical_register_neighbours = ["{0:b}".format(i).zfill(number_of_neighbors) for i in
                                         range(2 ** number_of_neighbors)]

        # create dictionary of the marginal states of qubits_of_interest and neighbors_of_interest
        # for the whole register (this function is storing data which could also be calculated in situ
        # in the loops later, but this is faster)
        indices_dictionary_small = {}
        for neighbors_state_bitstring in classical_register_all_qubits:
            small_string = ''.join([list(neighbors_state_bitstring)[all_qubits_enumerated[b]] for b in
                                    qubits_of_interest])
            neighbours_tuple_now = ''.join(
                [list(neighbors_state_bitstring)[all_qubits_enumerated[b]] for b in
                 neighbors_of_interest])
            # first place in list is label for state of qubits_of_interest
            # and second for neighbors_of_interest
            indices_dictionary_small[neighbors_state_bitstring] = [small_string, neighbours_tuple_now]

        # initiate dictionary for which KEY is input state of all_neighbors
        # and VALUE will the the corresponding noise matrix on qubits_of_interest
        conditional_noise_matrices = {
            s: np.zeros((2 ** number_of_qubits_of_interest, 2 ** number_of_qubits_of_interest)) for s
            in
            classical_register_neighbours}

        # go through all classical states
        for measured_state_integer in range(total_dimension):
            for input_state_integer in range(total_dimension):
                lambda_element = big_lambda[measured_state_integer, input_state_integer]

                # input state of all qubits in binary format
                input_state_bitstring = classical_register_all_qubits[input_state_integer]
                # measured state of all qubits in binary format
                measured_state_bitstring = classical_register_all_qubits[measured_state_integer]

                # input state of qubits_of_interest in binary format
                input_state_small = indices_dictionary_small[input_state_bitstring][0]
                # measured state of qubits_of_interest in binary format
                measured_state_small = indices_dictionary_small[measured_state_bitstring][0]

                # input state of neighbors_of_interest in binary format
                input_state_neighbours = indices_dictionary_small[input_state_bitstring][1]

                # element of small lambda labeled by (measured state | input state),
                # and the lambda itself is labeled by input state of all_neighbors
                conditional_noise_matrices[input_state_neighbours][
                    int(measured_state_small, 2), int(input_state_small, 2)] += lambda_element

        # normalize matrices
        for neighbors_state_bitstring in classical_register_neighbours:
            conditional_noise_matrices[neighbors_state_bitstring] /= normalization

        # conditional_noise_matrices['all_neighbors'] = neighbors_of_interest

        # cluster_string = 'q' + 'q'.join(str(s) for s in qubits_of_interest)
        # neighbours_string = 'q' + 'q'.join(str(s) for s in neighbors_of_interest)

        cluster_tuple_now = tuple(qubits_of_interest)
        neighbours_tuple_now = tuple(neighbors_of_interest)


        if cluster_tuple_now not in self._noise_matrices_dictionary.keys():
            # If there is no entry for our cluster in the dictionary, we create it and add
            # averaged noise matrix
            averaged_noise_matrix = np.zeros(
                (2 ** number_of_qubits_of_interest, 2 ** number_of_qubits_of_interest))
            for neighbors_state_bitstring in conditional_noise_matrices.keys():
                averaged_noise_matrix += conditional_noise_matrices[neighbors_state_bitstring]
            averaged_noise_matrix /= 2 ** number_of_qubits_of_interest
            self._noise_matrices_dictionary[cluster_tuple_now] = {'averaged': averaged_noise_matrix}

        self._noise_matrices_dictionary[cluster_tuple_now][neighbours_tuple_now] = conditional_noise_matrices

        return self._noise_matrices_dictionary[cluster_tuple_now][neighbours_tuple_now]


    def get_noise_matrix_dependent(self,
                                   qubits_of_interest: Tuple[int],
                                   neighbors_of_interest: Tuple[int]) -> dict:
        """Description:
        like self._compute_noise_matrix_dependent
        but checks whether matrices were already calculated to prevent multiple computations of the
        same matrices

        :param qubits_of_interest: labels of qubits in marginal  we are interested in
        :param neighbors_of_interest: labels of qubits that affect noise matrix on qubits_of_interest

        :return conditional_noise_matrices_dictionary:

        """

        # cluster_key = convert.qubit_indices_to_keystring(qubits_of_interest)

        cluster_key = tuple(qubits_of_interest)
        if cluster_key not in self._noise_matrices_dictionary.keys():
            self.compute_subset_noise_matrices_averaged([qubits_of_interest])
        # print('hey')

        if len(neighbors_of_interest) == 0 or neighbors_of_interest is None:
            neighbors_key = 'averaged'

            if neighbors_key in self._noise_matrices_dictionary[cluster_key]:
                # if not qrem_math.is_matrix_stochastic(self._noise_matrices_dictionary[cluster_key]['averaged']):
                #     qprint('Bug is here')
                #     print(cluster_key, neighbors_key)
                #     # FBM: SOMETHING IS BROKEN
                #     self._noise_matrices_dictionary[cluster_key][
                #         'averaged'] = self._compute_noise_matrix_averaged(qubits_of_interest)
                #     if not qrem_math.is_matrix_stochastic(self._noise_matrices_dictionary[cluster_key]['averaged']):
                #         qprint('And I cant fix it')

                # qprint_array(self._noise_matrices_dictionary[cluster_key]['averaged'])
                return {'averaged': self._noise_matrices_dictionary[cluster_key]['averaged']}
            else:

                #note that this is already a dict anyway
                averaged_matrix_dictionary = self._compute_noise_matrix_dependent(qubits_of_interest,
                                                                       neighbors_of_interest)

                if not qrem_math.is_matrix_stochastic(averaged_matrix_dictionary['averaged']):
                    raise ValueError(f"Noise matrix not stochastic for qubits: "
                                     f"cluster = {cluster_key}, neighbors = {neighbors_of_interest}.")

                return averaged_matrix_dictionary

        else:
            # neighbors_key = 'q' + 'q'.join([str(s) for s in neighbors_of_interest])

            neighbors_key = tuple(neighbors_of_interest)

            if neighbors_key in self._noise_matrices_dictionary[cluster_key]:
                deps_matrix = self._noise_matrices_dictionary[cluster_key][neighbors_key]
            else:
                deps_matrix = self._compute_noise_matrix_dependent(qubits_of_interest,
                                                                   neighbors_of_interest)

            for matrix_test in deps_matrix.values():
                if not qrem_math.is_matrix_stochastic(matrix_test):
                    raise ValueError(f"Noise matrix not stochastic for qubits: "
                                     f"cluster = {cluster_key}, neighbors = {neighbors_of_interest}.")

            return deps_matrix

    def compute_subset_noise_matrices_averaged(self,
                                               subsets_list: List[Tuple[int]],
                                               show_progress_bar: Optional[bool] = False) -> None:
        """Description:
        computes averaged (over all other qubits) noise matrices on subsets_list of qubits

        :param subsets_list: subsets_list of qubit indices
        :param show_progress_bar: whether to show animated progress bar. requires tqdm package

        """

        for subset_index in tqdm(subsets_list, disable = not show_progress_bar):
            
            #JT There is a bug in the line below, at least when it comes to data structures 
            #that we use as for now. For now I comment that line and replace it with a one that
            #works 
            #self._compute_noise_matrix_averaged(subsets_list[subset_index])

            self._compute_noise_matrix_averaged(subset_index)


