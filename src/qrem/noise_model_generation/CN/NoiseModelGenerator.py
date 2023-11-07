import numpy as np
import copy
from typing import Optional, List, Dict, Union, Tuple

from tqdm import tqdm

from qrem.functions_qrem import ancillary_functions as anf

#JT: Check whwter we need this here
from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.functions.functions_noise_model_heuristic import partition_algorithm_v1_cummulative

from qrem.noise_characterization.data_analysis.InitialNoiseAnalyzer import InitialNoiseAnalyzer


from qrem.common.printer import qprint
from qrem.common import convert, utils

class NoiseModelGenerator(InitialNoiseAnalyzer):
    """
        This is class that uses results of Diagonal Detector Tomography (DDT) to construct stochastic_matrix noise
        model for a measuring device. The model is classical and based on Ref. [0.5].

        The main functionalities include computing sets of strongly correlated qubits (clusters)
        and for each such set, computing the qubits which affect the exact form of the noise
        on those clusters (hence, neighborhoods of cluster)
    """

    def __init__(self,
                 results_dictionary: Dict[str, Dict[str, int]],
                 bitstrings_right_to_left: Optional[bool] = False,
                 marginals_dictionary: Optional[Dict[str, Dict[Tuple[int], np.ndarray]]] = None,
                 noise_matrices_dictionary: Optional[
                     Dict[Tuple[int], Dict[Union[str,Tuple[int]], Dict[str, np.ndarray]]]] = None,
                 clusters_list: Optional[List[Tuple[int]]] = None,
                 neighborhoods: Dict[Tuple[int], Tuple[int]] = None,
                 correlations_data:Optional[np.ndarray] = None
                 ) -> None:

        super().__init__(results_dictionary=results_dictionary,
                         bitstrings_right_to_left=bitstrings_right_to_left,
                         marginals_dictionary=marginals_dictionary,
                         noise_matrices_dictionary=noise_matrices_dictionary,
                         correlations_data=correlations_data
                         )

        # self._correlations_table_pairs = correlations_table_pairs

        if clusters_list is None:
            clusters_list = []

        if neighborhoods is None:
            neighborhoods = {}

        self._clusters_list = clusters_list

        self._neighborhoods = neighborhoods

    @property
    def clusters_list(self) -> List[Tuple[int]]:
        return self._clusters_list

    @clusters_list.setter
    def clusters_list(self, clusters_list: List[Tuple[int]]) -> None:
        for cluster in clusters_list:
            # cluster_string = anf.get_qubits_keystring(cluster)
            cluster_string = tuple(cluster)
            if cluster_string not in self._noise_matrices_dictionary.keys():
                average_noise_matrix_now = self._compute_noise_matrix_averaged(cluster)
                dictionary_now = {'averaged': average_noise_matrix_now}

                if cluster_string in self._neighborhoods.keys():
                    neighborhood_now = self._neighborhoods[cluster_string]
                    dependent_noise_matrices = self._compute_noise_matrix_dependent(cluster,
                                                                                    neighborhood_now)
                    dictionary_now = {**dictionary_now, **dependent_noise_matrices}
                    qprint('im doing this')

                self._noise_matrices_dictionary[tuple(cluster)] = dictionary_now

        self._clusters_list = clusters_list

    @property
    def neighborhoods(self) -> Dict[Tuple[int], Tuple[int]]:
        return self._neighborhoods

    @neighborhoods.setter
    def neighborhoods(self, neighborhoods: Dict[Tuple[int], Tuple[int]]) -> None:
        self._neighborhoods = neighborhoods
        self.clusters_list = [cluster for cluster in neighborhoods.keys()]

        for cluster_string in neighborhoods.keys():
            dictionary_now = self._noise_matrices_dictionary[cluster_string]
            neighborhood_now = neighborhoods[cluster_string]
            # print(dictionary_now.keys())
            neighbors_key = anf.get_qubits_keystring(neighborhood_now)
            if neighbors_key not in dictionary_now.keys():
                cluster = convert.get_qubit_indices_from_keystring(cluster_string)

                dependent_noise_matrices = self._compute_noise_matrix_dependent(cluster,
                                                                                neighborhood_now)

                self._noise_matrices_dictionary[cluster_string] = {**dictionary_now,
                                                                   **dependent_noise_matrices}

    def _compute_clusters_pairwise(self,
                                   maximal_size: int,
                                   cluster_threshold: float
                                   ) -> list:
        """
            Get partition of qubits in a device into disjoint "clusters". This function uses "naive"
            method_name by assigning qubits to the same cluster if correlations between them are higher
            than some "neighbors_threshold". It restricts size of the cluster to "maximal_size"
            by disregarding the lowest correlations (that are above neighbors_threshold).
            It uses table of correlations from class property self._correlations_table_pairs

          :param cluster_threshold: correlations magnitude above which qubits are assigned
                 to the same cluster
          :param maximal_size: maximal allowed size of the cluster

          :return: clusters_labels_list: list of lists, each representing a single cluster
             """
        self._clusters_list = []

        qubit_indices = self._qubit_indices
        # number_of_qubits = len(qubit_indices)

        clusters = {'q%s' % qi: [[qi, 0., 0.]] for qi in qubit_indices}
        for qi in qubit_indices:
            for qj in qubit_indices:
                if qj > qi:
                    corr_j_i, corr_i_j = self._correlations_table[qj, qi], \
                                         self._correlations_table[qi, qj]

                    # if any of the qubit affects the other strongly enough,
                    # we assign them to the same cluster
                    if corr_j_i >= cluster_threshold or corr_i_j >= cluster_threshold:
                        clusters['q%s' % qi].append([qj, corr_i_j, corr_j_i])
                        clusters['q%s' % qj].append([qi, corr_i_j, corr_j_i])

        # Merge clusters containing the same qubits
        #(PP) this seems quite inefficient approach

        #TODO BEGIN improve efficiency of this check
        new_lists = []

        for key, value in clusters.items():
            clusters[key] = sorted(value, key=lambda arg: arg[0])
            new_lists.append([vi[0] for vi in clusters[key]])

        # print(new_lists)
        while utils.check_for_multiple_occurences(new_lists):
            for i in range(len(new_lists)):
                cl0 = new_lists[i]
                for j in range(len(new_lists)):
                    cl1 = new_lists[j]
                    if len(utils.lists_intersection(cl0, cl1)) != 0:
                        new_lists[i] = utils.lists_sum(cl0, cl1)

            uniques = np.unique(new_lists)

            if not isinstance(uniques[0], list):
                uniques = [uniques]

            unique_stuff = [sorted(lis) for lis in uniques]
            new_lists = copy.deepcopy(unique_stuff)
        #END ------------------------------------------------

        clusters_list = new_lists


        # Chop clusters if they exceed max size
        while np.any(np.array([len(cluster) > maximal_size for cluster in clusters_list])):
            chopped_clusters = []
            for cluster in clusters_list:
                if len(cluster) > maximal_size:
                    correlations_sorting = []
                    for qi in cluster:
                        # as figure of merit, we will sum all correlations that are between
                        # given qubit and other guys in its cluster.
                        x = 0.0
                        for list_now in clusters['q%s' % qi]:
                            x += np.max([list_now[1], list_now[2]])

                        correlations_sorting.append([qi, x])

                    correlations_sorted = sorted(correlations_sorting,
                                                 key=lambda arg: arg[1],
                                                 reverse=True)

                    # choose only "maximal_size" qubits to belong to given cluster
                    qubits_sorted = [correlations_sorted[index][0] for index in range(maximal_size)]
                    qubits_left = [correlations_sorted[index][0] for index in
                                   range(maximal_size, len(correlations_sorted))]
                    chopped_clusters.append(sorted(qubits_left))
                else:
                    qubits_sorted = cluster
                chopped_clusters.append(sorted(qubits_sorted))

            clusters_list = sorted(chopped_clusters, key=lambda y: y[0])

        self._clusters_list = clusters_list

        return clusters_list

    def _find_neighbors_of_cluster_holistic(self,
                                            cluster: List[int],
                                            maximal_size: int,
                                            chopping_threshold: Optional[float] = 0.) -> List[int]:
        """
        For a given cluster of qubits, find qubits which are their neighbors, i.e., they affect the
        noise matrix of cluster significantly. Figure of merit for correlations here is:

        c_{j -> cluster} = 1/2 || Lambda_{cluster}^{Y_j='0'}- Lambda_{cluster}^{Y_j='1'}||_{l1}

        where Lambda_{cluster}^{Y_j} is the noise matrix describing noise on qubits in "cluster"
        provided that input state of qubit "j" was "Y_j".
        See also description of self._compute_clusters_pairwise.


        :param cluster: list of labels of qubits in a cluster
        :param maximal_size: maximal allowed size of the set "cluster+neighborhood"
        :param chopping_threshold: numerical value, for which correlations lower than
              chopping_threshold are set to 0.
              If not provided, it adds all_neighbors until maximal_size is met.



        :return: neighbors_list: list of lists, each representing a single cluster
        """

        size_cut = maximal_size - len(cluster)

        potential_neighbours = []

        #JT this is added as a treshold

        correlation_threshold = 0.009

        for qi in self._qubit_indices:
            if qi not in cluster:
                compute_noise_matrix = False

                #JT this is added to avoid extensive computation of noise matrices

                for qj in cluster: 
                    corr_i_j = self._correlations_data['worst_case']['classical'][qj, qi]

                    #
                    if corr_i_j > correlation_threshold:
                        
                        compute_noise_matrix = True
                        
                        break

                #if the corresponding correlation coefficient is above threshold computation of the noise matrix will be performed 
                if compute_noise_matrix:    
                    lam_ci_j = self.get_noise_matrix_dependent(cluster,
                                                           [qi])
                    diff_ci_j = lam_ci_j['0'] - lam_ci_j['1']
                    correlation_ci_j = 1 / 2 * np.linalg.norm(diff_ci_j, ord=1)
                    potential_neighbours.append([qi, correlation_ci_j])

        sorted_neighbours = sorted(potential_neighbours, key=lambda x: x[1], reverse=True)

        neighbors_list = sorted(
            [sorted_neighbours[i][0] for i in range(int(np.min([size_cut, len(sorted_neighbours)]))) if
             chopping_threshold < sorted_neighbours[i][1]])

        cluster_key = anf.get_qubits_keystring(cluster)

        self._neighborhoods[cluster_key] = neighbors_list

        return neighbors_list

    def _find_all_neighborhoods_holistic(self,
                                         maximal_size,
                                         chopping_threshold: float,
                                         show_progress_bar: Optional[bool] = False) \
            -> Dict[str, List[int]]:
        """
                Run self._find_neighbors_of_cluster_holistic for all clusters.

                :param maximal_size: maximal allowed size of the set "cluster+neighborhood"
                :param chopping_threshold: numerical value, for which correlations lower than
                      chopping_threshold are set to 0.
                      If not provided, it adds all_neighbors until maximal_size is met.
                :param show_progress_bar: specify whether to show progress bar


                :return: neighbors_dictionary: dictionary where KEY is label for cluster,
                                               and VALUE is list of its neighbors
        """

        self._neighborhoods = {}
        clusters_list = self._clusters_list
        range_clusters = range(len(clusters_list))

        if show_progress_bar:
            range_clusters = tqdm(range_clusters)

        for index_cluster in range_clusters:
            cluster = clusters_list[index_cluster]
            self._neighborhoods[
                anf.get_qubits_keystring(cluster)] = self._find_neighbors_of_cluster_holistic(
                cluster,
                maximal_size,
                chopping_threshold)

        return self._neighborhoods

    def _find_neighbors_of_cluster_pairwise(self,
                                            cluster: List[int],
                                            maximal_size: int,
                                            neighbors_threshold: float
                                            ) -> List[int]:
        """
        Like self._find_neighbors_of_cluster_holistic but looks how noise on qubits in given cluster
        depend on input state of other qubits (potential neighbors) *separately*.

        NOTE: see description of self._find_neighbors_of_cluster_holistic
                                            for definition of correlations' measure we use

        :param cluster: list of labels of qubits in a cluster
        :param maximal_size: maximal allowed size of the set "cluster+neighborhood"
        :param neighbors_threshold: numerical value, for which correlations higher than
              neighbors_threshold assign qubit to the neighborhood of other qubit

        :return: neighbors_list: list of lists, each representing a single cluster
        """

        qubit_indices = self._qubit_indices
        potential_neighbors = []

        for qj in qubit_indices:
            affections_qj = []
            for qi in cluster:
                if qj not in cluster:
                    corr_j_i = self._correlations_table[qi, qj]
                    affections_qj.append(corr_j_i)

            if qj not in cluster:
                corr_j_i = np.max(affections_qj)

                if corr_j_i >= neighbors_threshold:
                    potential_neighbors.append([qj, corr_j_i])
        sorted_neighbors = sorted(potential_neighbors, key=lambda x: x[1], reverse=True)

        target_size = maximal_size - len(cluster)
        range_final = int(np.min([len(sorted_neighbors), target_size]))

        return sorted([sorted_neighbors[index][0] for index in
                       range(range_final)])

    def _find_all_neighborhoods_pairwise(self,
                                         maximal_size: int,
                                         neighbors_threshold: float,
                                         show_progress_bar: Optional[bool] = False
                                         ) -> Dict[str, List[int]]:
        """
        Like self._find_neighbors_of_cluster_holistic but looks how noise on qubits in given cluster
        depend on input state of other qubits (potential neighbors) *separately*.

        NOTE: see description of self._find_neighbors_of_cluster_holistic
                                            for definition of correlations' measure we use

        :param maximal_size: maximal allowed size of the set "cluster+neighborhood"
        :param neighbors_threshold: numerical value, for which correlations higher than
              neighbors_threshold assign qubit to the neighborhood of other qubit

        :return: neighbors_dictionary: dictionary where KEY is label for cluster, and VALUE is list of
                                       its neighbors
        """
        if self._correlations_table is None:
            self._compute_correlations_data_pairs_old()
#why do we use old version of the method if we have a new one?
        self._neighborhoods = {}

        clusters_list = self._clusters_list
        range_clusters = range(len(clusters_list))

        if show_progress_bar:
            range_clusters = tqdm(range_clusters)

        for index_cluster in range_clusters:
            cluster = clusters_list[index_cluster]
            self._neighborhoods[tuple(cluster)] = self._find_neighbors_of_cluster_pairwise(
                cluster, maximal_size=maximal_size, neighbors_threshold=neighbors_threshold)

        return self._neighborhoods

    def compute_clusters(self,
                         maximal_size: int,
                         correlations_table:np.ndarray,
                         method: Optional[str] = 'CSP',
                         method_kwargs: Optional[dict] = None) -> list:
        """
        Get partition of qubits in a device into disjoint "clusters".
        This function uses various heuristic methods, specified via string "version".
        It uses table of correlations from class property self._correlations_table_pairs

        :param maximal_size: maximal allowed size of the cluster
        :param method: string specifying stochasticity_type of heuristic
        Possible values:
            'pairwise' - heuristic that uses Algorithm 3 from Ref.[]
            'CSP' - heuristic Cluster Size Penalization


        :param method_kwargs: potential arguments that will be passed to clustering function.
                           For possible parameters see descriptions of particular functions_qrem.
        :return: clusters_labels_list: list of lists, each representing a single cluster
        """

        self._clusters_list = []

        if method == 'pairwise':
            if method_kwargs is None:
                default_kwargs = {'maximal_size': maximal_size,
                                  'cluster_threshold': 0.02
                                  }

                method_kwargs = default_kwargs
            elif 'maximal_size' in method_kwargs.keys():
                if method_kwargs['maximal_size'] != maximal_size:
                    raise ValueError('Disagreement between maximal size argument and method_name kwargs')
            else:
                method_kwargs['maximal_size'] = maximal_size
            clusters_list = self._compute_clusters_pairwise(**method_kwargs)


        elif method.upper()=='CSP':

            from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.child_classes.heuristic_clustering_algorithm import HeuristicClusteringAlgorithm

            if method_kwargs is None:
                method_kwargs = {'maximal_size': maximal_size,
                                  'version': 'v4',
                                  'number_of_algorithm_iterations':1000,
                                  'alpha_multiplier':1.0
                                  }

                # method_kwargs = default_kwargs

            if 'version' not in method_kwargs.keys():
                method_kwargs['version'] = 'v4'

            if 'number_of_algorithm_iterations' not in method_kwargs.keys():
                method_kwargs['number_of_algorithm_iterations'] = 1000

            if 'alpha_multiplier' not in method_kwargs.keys():
                method_kwargs['alpha_multiplier'] = 1.0

            if 'maximal_size' not in method_kwargs.keys():
                method_kwargs['maximal_size'] = maximal_size

            #v0 is interpreted sa "pairwise"
            if method_kwargs['version'] == 'v0':

                clustering_algorithm = HeuristicClusteringAlgorithm(
                    correlations_table=correlations_table,
                    version='v1',
                    C_maxsize=maximal_size)

                clustering_algorithm.suggest_alpha_heuristic()

                alpha = 2*clustering_algorithm.alpha*method_kwargs['alpha_multiplier']
                kwargs_pairwise = {'maximal_size': maximal_size,
                                  'cluster_threshold': alpha
                                  }

                clusters_list = self._compute_clusters_pairwise(**kwargs_pairwise)

                self._alpha_parameter = alpha

            #JT Here clustering algorithm is chosen

            else:
                clustering_algorithm = HeuristicClusteringAlgorithm(correlations_table=correlations_table,
                                                                    version=method_kwargs['version'],
                                                                    C_maxsize=maximal_size,
                                                                    N_alg = method_kwargs['number_of_algorithm_iterations'])#, N_rand = method_kwargs['N_rand'])

                clustering_algorithm.suggest_alpha_heuristic()

                # print(clustering_algorithm.alpha)
                #TODO someone: make sure that this 2 factor is consitent with the paper
                clustering_algorithm.alpha = 2*clustering_algorithm.alpha*method_kwargs['alpha_multiplier']


                clusters_list, score = clustering_algorithm.clusterize()

                # qprint('Current partitioning got score:', score)

                self._alpha_parameter = clustering_algorithm.alpha




        elif method == 'holistic_v1':
            if method_kwargs is None:
                alpha = 1
                algorithm_runs = 1000
                default_kwargs = {'alpha': alpha,
                                  'N_alg': algorithm_runs,
                                  'printing': False,
                                  'drawing': False,
                                  'C_maxsize': maximal_size}

                method_kwargs = default_kwargs

            elif 'C_maxsize' in method_kwargs.keys():
                # TODO FBM, OS: this variable should have name consistent with rest of functions_qrem
                if method_kwargs['C_maxsize'] != maximal_size:
                    raise ValueError('Disagreement between maximal size argument and method_name kwargs')
            else:
                method_kwargs['C_maxsize'] = maximal_size

            clusters_list, score = partition_algorithm_v1_cummulative(correlations_table,
                                                                      **method_kwargs)

            qprint('Current partitioning got score:', score)
        elif method == 'external':
            if 'clustering_algorithm' not in method_kwargs.keys():
                raise ValueError('No HeuristicClusteringAlgorithmBase instance provided in method kwargs')
            clustering_algorithm = method_kwargs['clustering_algorithm']
            if clustering_algorithm.C_maxsize != maximal_size:
                raise ValueError(
                    'Disagreement between maximal size argument and method_name kwargs')
            clusters_list, score = clustering_algorithm.clusterize()
        else:
            raise ValueError('No heuristic with that name: ' + method)

        clusters_list_formatted = [tuple(sorted(x)) for x in clusters_list]
        clusters_list_formatted = tuple(sorted(clusters_list_formatted, key=lambda x: x[0]))
        from qrem.functions_qrem import functions_data_analysis as fda

        clusters_list_formatted = fda._sort_clusters_division(clusters_list_formatted)

        self._clusters_list = clusters_list_formatted

#apart from clusters list also value of the clustering function is returned
        return clusters_list_formatted, score

    def find_all_neighborhoods(self,
                               maximal_size: int,
                               method: Optional[str] = 'holistic',
                               method_kwargs: Optional[dict] = None):

        if method == 'pairwise':
            if method_kwargs is None:
                default_kwargs = {'neighbors_threshold': 0.01}
                method_kwargs = default_kwargs

            method_kwargs['maximal_size'] = maximal_size
            neighborhoods = self._find_all_neighborhoods_pairwise(**method_kwargs)

        elif method == 'holistic':
            if method_kwargs is None:
                default_kwargs = {'chopping_threshold': 0.01,
                                  'show_progress_bar': True}
                method_kwargs = default_kwargs
            method_kwargs['maximal_size'] = maximal_size
            neighborhoods = self._find_all_neighborhoods_holistic(**method_kwargs)

        else:
            raise ValueError('Wrong method_name name')

        return neighborhoods
#JT : This can be safely removed 
    def print_properties(self):
        # TODO FBM, OS: add this

        return None

    def draw_noise_model(self):
        # TODO FBM, OS: add this

        return None
