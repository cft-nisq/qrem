"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com

REFERENCES:
[0] Filip B. Maciejewski, Zoltán Zimborás, Michał Oszmaniec,
"Mitigation of readout noise in near-term quantum devices
by classical post-processing based on detector tomography",
Quantum 4, 257 (2020)

[0.5] Filip B. Maciejewski, Flavio Baccari, Zoltán Zimborás, Michał Oszmaniec,
"Modeling and mitigation of cross-talk effects in readout noise
with applications to the Quantum Approximate Optimization Algorithm",
Quantum 5, 464 (2021).

"""

from typing import Optional, Dict, Union, List, Tuple

import numpy as np

from qrem.functions_qrem import povmtools, ancillary_functions as anf
from qrem.noise_model_generation.CN.GlobalNoiseMatrixCreator import \
    GlobalNoiseMatrixCreator
from qrem.noise_model_generation.CN.NoiseModelGenerator import NoiseModelGenerator

from qrem.common import utils
from qrem.common.printer import qprint

class CorrectionDataGenerator(NoiseModelGenerator):
    """
        Main class used to calculate data needed for noise-mitigation on marginals_dictionary, based on provided
        noise model.

        NOTE: Currently it handles properly only two-qubit marginals_dictionary (e.g., experiments involving
              estimation of 2-local Hamiltonians)

        The correction data consists of the following:
        - :param: 'correction_indices' - for each marginal of interest (e.g., 'q0q1'), specify label for
                                 marginal that needs to be corrected and then coarse-grained in order
                                 to perform noise-mitigation. For example, if q0 is strongly correlated
                                 with q5 (i.e., they are in the same cluster), it is advisable to first
                                 perform noise mitigation on marginal 'q0q1q5', and then coarse-grain
                                 it to obtain 'q0q1'.

                                 The format we use here is dictionary where KEY is label for marginal
                                 of interest, and VALUE is label for marginal that needs to be
                                 calculated first.
                                 So the entry for example above would look like:
                                 correction_indices['q0q1'] = 'q0q1q5'


        - :param: 'noise_matrices' - the noise matrices representing effective noise matrix acting on marginals_dictionary
                            specified by values of correction_indices dictionary

                            This is dictionary where KEY is subset label and VALUE is noise matrix


        - :param: 'correction_matrices' - inverses of noise_matrices, convention used is the same
    """

    # TODO FBM: generalize for more than two-qubit subsets
    # TODO FBM: add mitigation errors

    def __init__(self,
                 results_dictionary_ddt: Dict[str, Dict[str, int]],
                 number_of_qubits: int,
                 bitstrings_right_to_left: Optional[bool] = False,
                 marginals_dictionary: Optional[Dict[str, Dict[Tuple[int], np.ndarray]]] = None,
                 clusters_list: Optional[List[Tuple[int]]] = None,
                 neighborhoods: Optional[Dict[Tuple[int], Tuple[int]]] = None,
                 noise_matrices_dictionary: Optional[
                     Dict[Tuple[int], Dict[Union[str, Tuple[int]], Dict[str, np.ndarray]]]] = None,
                 correction_matrices_dictionary: Optional[Dict[Tuple[int], np.ndarray]] = None
                 ) -> None:

        super().__init__(results_dictionary_ddt,
                         bitstrings_right_to_left,
                         marginals_dictionary,
                         noise_matrices_dictionary,
                         clusters_list,
                         neighborhoods
                         )
        self._number_of_qubits = number_of_qubits
        self._qubit_indices = range(number_of_qubits)

        if clusters_list is None:
            clusters_lists_dictionary, clusters_labels_dictionary = {}, {}

        #JT: clusters_lists_dictionary: a dictionary with keys corresponding to qubits and values to clusters 
        else:
            clusters_lists_dictionary = {(qi,): () for qi in self._qubit_indices}
            for cluster in self._clusters_list:
                for qi in cluster:
                    clusters_lists_dictionary[(qi,)] = cluster


            #JT: how this is different from clusters_lists dictionary
            clusters_labels_dictionary = {
                (qi,): clusters_lists_dictionary[(qi,)] for qi in
                self._qubit_indices}

        # self._noise_matrices = noise_matrices_dictionary
        self._correction_matrices = correction_matrices_dictionary

        if noise_matrices_dictionary is None:
            self._noise_matrices_dictionary = {}

        if correction_matrices_dictionary is None:
            self._correction_matrices = {}

        self._mitigation_errors = {}
        self._correction_indices = {}

        self._clusters_lists_dictionary = clusters_lists_dictionary
        self._clusters_labels_dictionary = clusters_labels_dictionary

    # @property
    # def correction_matrices(self)->Dict[Tuple[int],np.ndarray]:
    #

    def _set_clusters_dictionary(self) -> None:
        """
        Function that updates class' properties needed (later) to calculate correction data
        """

        clusters_lists_dictionary = {(qi,): () for qi in self._qubit_indices}
        for cluster in self._clusters_list:
            for qi in cluster:
                clusters_lists_dictionary[(qi,)] = cluster
        clusters_labels_dictionary = {
            (qi,): clusters_lists_dictionary[(qi,)] for qi in
            self._qubit_indices}
        self._clusters_lists_dictionary = clusters_lists_dictionary
        self._clusters_labels_dictionary = clusters_labels_dictionary

    def _construct_local_full_noise_model(self,
                                          qubit_i: int,
                                          qubit_j: int,
                                          intersection_i: Optional[List[int]] = None,
                                          intersection_j: Optional[List[int]] = None
                                          ) -> np.ndarray:
        """
        This is function to obtain local noise model for two clusters to which both qubits belong.
        It is slightly tedious, the main issue here is to properly sort qubits and format input data
        for GlobalNoiseMatrixCreator (different class)

        :param qubit_i: qubit that belongs to cluster i
        :param qubit_j: as above
        :param intersection_i: qubits that are neighbors of cluster i and belong to cluster j
        :param intersection_j: as above
        :return:
        """

        # get needed information from class' properties
        cluster_i = self._clusters_lists_dictionary[(qubit_i,)]
        cluster_j = self._clusters_lists_dictionary[(qubit_j,)]

        string_cluster_i = self._clusters_labels_dictionary[(qubit_i,)]
        string_cluster_j = self._clusters_labels_dictionary[(qubit_j,)]

        neighbors_i = self._neighborhoods[string_cluster_i]
        neighbors_j = self._neighborhoods[string_cluster_j]

        # if not provided, calculate it
        if intersection_i is None:
            intersection_i = utils.lists_intersection(neighbors_i, cluster_j)
        if intersection_j is None:
            intersection_j = utils.lists_intersection(neighbors_j, cluster_i)

        # Take cluster noise matrices depending on neighbors states
        matrices_cluster_i = self.get_noise_matrix_dependent(cluster_i,
                                                             neighbors_i)
        matrices_cluster_j = self.get_noise_matrix_dependent(cluster_j,
                                                             neighbors_j)

        # average over neighbours of first cluster
        # that do not include the members of second cluster
        averaged_matrices_cluster_i = self.average_noise_matrices_over_some_qubits(
            matrices_cluster_i,
            neighbors_i,
            intersection_i
        )
        # and vice versa
        averaged_matrices_cluster_j = self.average_noise_matrices_over_some_qubits(
            matrices_cluster_j,
            neighbors_j,
            intersection_j)

        """
        Here we will need to properly relabel qubits because noise matrix creator requires 
        numbering them from 0 to number of qubits.
        """
        # Label qubits from 0
        enumerated_map_reversed = utils.map_index_to_order(
            cluster_i + cluster_j)

        qubit_indices_enumerated = []
        for cluster_now in [cluster_i, cluster_j]:
            qubit_indices_enumerated.append(
                [enumerated_map_reversed[ci] for ci in cluster_now])

        cluster_key_for_construction_i = anf.get_qubits_keystring(
            qubit_indices_enumerated[0])
        cluster_key_for_construction_j = anf.get_qubits_keystring(
            qubit_indices_enumerated[1])

        # this is format of noise matrices dictionary accepted by GlobalNoiseMatrixCreator
        properly_formatted_lambdas = {cluster_key_for_construction_i: averaged_matrices_cluster_i,
                                      cluster_key_for_construction_j: averaged_matrices_cluster_j
                                      }

        # this is format for neighbors dictionary accepted by GlobalNoiseMatrixCreator
        neighbors_for_construction = {
            cluster_key_for_construction_i: [enumerated_map_reversed[ci] for ci in
                                             intersection_i],
            cluster_key_for_construction_j: [enumerated_map_reversed[cj] for cj in
                                             intersection_j]}

        # get instance of GlobalNoiseMatrixCreator
        big_lambda_creator_now = GlobalNoiseMatrixCreator(
            properly_formatted_lambdas,
            qubit_indices_enumerated,
            neighbors_for_construction)

        # get noise matrix
        local_noise_matrix = \
            big_lambda_creator_now.compute_global_noise_matrix()

        return local_noise_matrix

    @staticmethod
    def _sort_qubits_matrix(
            local_noise_matrix: np.ndarray,
            cluster_i: List[int],
            cluster_j: List[int]) -> np.ndarray:
        """
        :param local_noise_matrix: noise matrix acting on cluster_i and cluster_j
        :param cluster_i: list of qubits' indices
        :param cluster_j: list of qubits' indices
        :return: permuted local_noise_matrix
        """

        all_qubits_list = list(cluster_i + cluster_j)

        # This is our target
        sorted_qubits_dictionary = utils.enumerate_dict(sorted(all_qubits_list))
        # This is what we have
        qubits_dictionary = utils.enumerate_dict(all_qubits_list)

        # While what we have is not equal to target we sort qubits.
        # The following loop performs series of SWAPs to properly order qubits.
        while qubits_dictionary != sorted_qubits_dictionary:
            for index_qubit_here in range(len(all_qubits_list) - 1):
                if all_qubits_list[index_qubit_here + 1] < all_qubits_list[index_qubit_here]:
                    # if two qubits are not sorted in ascending order, we permute matrix
                    # this corresponds to exchanging place of two qubits in the Hilbert space
                    local_noise_matrix = povmtools.permute_matrix(local_noise_matrix,
                                                                  len(all_qubits_list),
                                                                  [index_qubit_here + 1,
                                                                   index_qubit_here + 2])
                    # update indices to keep track of already made swaps
                    all_qubits_list[index_qubit_here], all_qubits_list[
                        index_qubit_here + 1] = all_qubits_list[index_qubit_here + 1], all_qubits_list[
                        index_qubit_here]
            # update whole dictionary
            qubits_dictionary = utils.enumerate_dict(all_qubits_list)
        return local_noise_matrix

    def _compute_pair_noise_matrix(self,
                                   pair: Tuple[int]):

        qubit_i, qubit_j = pair[0], pair[1]

        cluster_i = self._clusters_lists_dictionary[(qubit_i,)]
        cluster_j = self._clusters_lists_dictionary[(qubit_j,)]

        if cluster_i == cluster_j:
            # Check if qubits are in the same cluster. If yes, we just take average
            # noise matrix on that cluster.
            averaged_matrix_clusters_i_j = self.get_noise_matrix_averaged(
                sorted(utils.lists_sum(cluster_i, cluster_j)))

        else:
            string_cluster_i = self._clusters_labels_dictionary[(qubit_i,)]
            string_cluster_j = self._clusters_labels_dictionary[(qubit_j,)]
            # If qubits are in different clusters, we have two options to consider below...
            intersection_i, intersection_j = \
                utils.lists_intersection(
                    self._neighborhoods[string_cluster_i],
                    cluster_j), \
                utils.lists_intersection(
                    self._neighborhoods[string_cluster_j],
                    cluster_i)

            if len(intersection_i) == 0 and len(intersection_j) == 0:
                # Check if clusters contain each others neighbors.
                # If not, the noise matrix is simply a
                # tensor product of cluster matrices
                averaged_matrix_cluster_i = self.get_noise_matrix_averaged(cluster_i)
                averaged_matrix_cluster_j = self.get_noise_matrix_averaged(cluster_j)

                # print(averaged_matrix_cluster_i,averaged_matrix_cluster_j)

                averaged_matrix_clusters_i_j = np.kron(averaged_matrix_cluster_i,
                                                       averaged_matrix_cluster_j)

            else:
                # Check if clusters are each others neighbors.
                # If yes, the noise matrix needs to be constructed using
                # cluster-neighborhoods noise model with treating
                # some members of clusters as neighbours
                averaged_matrix_clusters_i_j = self._construct_local_full_noise_model(
                    qubit_i=qubit_i,
                    qubit_j=qubit_j,
                    intersection_i=intersection_i,
                    intersection_j=intersection_j)

        local_noise_matrix = averaged_matrix_clusters_i_j
        # check whether qubits are properly sorted
        if cluster_i != cluster_j and cluster_i + cluster_j != sorted(cluster_i + cluster_j):
            # if qubits are not sorted, noise matrix needs to be permuted
            local_noise_matrix = self._sort_qubits_matrix(
                local_noise_matrix=averaged_matrix_clusters_i_j,
                cluster_i=cluster_i,
                cluster_j=cluster_j)

        return local_noise_matrix

    def _compute_pair_correction_data(self,
                                      pair: Tuple[int]) -> None:
        """
        For given pair of qubits, get correction data required
        to correct corresponding two-qubit marginal.
        NOTE: see class' description

        :param pair: list of qubit indices
        """

        qubit_i, qubit_j = pair[0], pair[1]

        cluster_i = self._clusters_lists_dictionary[(qubit_i,)]
        cluster_j = self._clusters_lists_dictionary[(qubit_j,)]

        whole_marginal = sorted(utils.lists_sum(cluster_i, cluster_j))
        marginal_indices_tuple = tuple(whole_marginal)

        # print(marginal_indices_tuple)

        # print(self._noise_matrices_dictionary.keys())
        if marginal_indices_tuple in self._noise_matrices_dictionary.keys():

            if 'averaged' in self._noise_matrices_dictionary[marginal_indices_tuple].keys():
                local_noise_matrix = self._noise_matrices_dictionary[marginal_indices_tuple]['averaged']
                # print('not computing noise matrix!')
            else:
                local_noise_matrix = self._compute_pair_noise_matrix(pair)

                self._noise_matrices_dictionary[marginal_indices_tuple]['averaged'] = local_noise_matrix
        else:
            #JT: A potential source of the problem
            local_noise_matrix = self._compute_pair_noise_matrix(pair)
            self._noise_matrices_dictionary[marginal_indices_tuple] = {'averaged': local_noise_matrix}

        # JT this is a quic try to fix problem with inverse, it's non-optimal and should be changed in the future, it will not work eg. in the case when nieghbours are considered
        # the overall matriox is constructed as a kronecker product of two inverses
        # firstly we calculate average noise matrices for the two qubits
        averaged_matrix_cluster_i = self.get_noise_matrix_averaged(cluster_i)

        #if the two qubits belong to the same cluster, the inverse is computed
        if cluster_i == cluster_j:
            try:
                correction_matrix = np.linalg.inv(averaged_matrix_cluster_i)

            except:
                qprint(
                    f"MATRIX i FOR SUBSET: {marginal_indices_tuple} is not invertible, computing pseudoinverse", '',
                    'red')
                correction_matrix = np.linalg.pinv(averaged_matrix_cluster_i)

        #if the qubits are in different clusters we need the two inverses
        else:
            averaged_matrix_cluster_j = self.get_noise_matrix_averaged(cluster_j)
            try:
                correction_matrix_cluster_i = np.linalg.inv(averaged_matrix_cluster_i)
            except:
                qprint(f"MATRIX i FOR SUBSET: {marginal_indices_tuple} is not invertible, computing pseudoinverse",'','red')
                correction_matrix_cluster_i = np.linalg.pinv(averaged_matrix_cluster_i)
            try:
                correction_matrix_cluster_j = np.linalg.inv(averaged_matrix_cluster_j)
            except:
                qprint(f"MATRIX j FOR SUBSET: {marginal_indices_tuple} is not invertible, computing pseudoinverse",
                           '', 'red')
                correction_matrix_cluster_j = np.linalg.pinv(averaged_matrix_cluster_j)
            correction_matrix = np.kron(correction_matrix_cluster_i, correction_matrix_cluster_j)
            #the order of qubits is checked, reordering is performed when needed
            if cluster_i + cluster_j != sorted(cluster_i+ cluster_j):
                correction_matrix = self._sort_qubits_matrix(local_noise_matrix=correction_matrix,cluster_i=cluster_i,cluster_j=cluster_j)
        #here the modification of inverse computation ends

        #JT uncimment this try except if the above does not work
        #try:
        #    correction_matrix = np.linalg.inv(local_noise_matrix)
        #except:
        #    qprint(f"MATRIX FOR SUBSET: {marginal_indices_tuple} is not invertible, computing pseudoinverse",'','red')
        #    qprint(f"DET: {np.linalg.det(local_noise_matrix)} ",'', 'red')
        #    correction_matrix = np.linalg.pinv(local_noise_matrix)

        #here modification ends


        # correction_matrix = np.linalg.inv(local_noise_matrix)

        #Perhaps do not do this
        self._correction_indices[(qubit_i,)] = marginal_indices_tuple
        self._correction_indices[(qubit_j,)] = marginal_indices_tuple
        self._correction_indices[(qubit_i, qubit_j)] = marginal_indices_tuple
        self._correction_matrices[marginal_indices_tuple] = correction_matrix

    def get_pairs_correction_data(self,
                                  pairs_list: List[Tuple[int]],
                                  show_progress_bar: Optional[bool] = False,
                                  reset_everything: Optional[bool] = False
                                  ) -> dict:
        """
        For pairs of qubits in the list, get correction data required
        to correct corresponding two-qubit marginals_dictionary.
        NOTE: see class' description for details

        :param pairs_list:
        :param show_progress_bar:
        :return: correction_data_dictionary
        """

        # TODO FBM: possibly change resetting
        self._set_clusters_dictionary()

        if reset_everything:
            # self._noise_matrices = {}
            self._correction_matrices = {}

        self._mitigation_errors = {}
        self._correction_indices = {}

        range_pairs = range(len(pairs_list))

        if show_progress_bar:
            from tqdm import tqdm
            range_pairs = tqdm(range_pairs)

        for pair_index in range_pairs:
            pair = pairs_list[pair_index]
            pair_key = tuple(pair)
            if pair_key not in self._correction_indices.keys():
                self._compute_pair_correction_data(pair)

        for qubit in range(self._number_of_qubits):
            # TODO FBM: perhaps modify this

            cluster_now = self._clusters_labels_dictionary[(qubit,)]
            self._correction_indices[(qubit,)] = cluster_now

            if len(cluster_now) == 1:
                if (qubit,) not in self._correction_matrices.keys():
                    # local_noise_matrix = self._compute_noise_matrix_averaged(subset=(qubit,))

                    local_noise_matrix = self.get_noise_matrix_averaged(subset=(qubit,))
                    # self.noise_matrices_dictionary[(qubit,)]['averaged'] = local_noise_matrix
                    try:
                        self._correction_matrices[(qubit,)] = np.linalg.inv(local_noise_matrix)
                    except:
                        qprint(f"MATRIX FOR SUBSET: {(qubit,)} is not invertible, computing pseudinverse",'','red')
                        self._correction_matrices[(qubit,)] = np.linalg.pinv(local_noise_matrix)


        return {'correction_matrices': self._correction_matrices,
                'noise_matrices': self.noise_matrices_dictionary,
                'correction_indices': self._correction_indices}
