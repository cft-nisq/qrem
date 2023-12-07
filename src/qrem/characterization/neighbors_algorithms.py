

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

        cluster_key = convert.qubit_indices_to_keystring(cluster)

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
                convert.qubit_indices_to_keystring(cluster)] = self._find_neighbors_of_cluster_holistic(
                cluster,
                maximal_size,
                chopping_threshold)

        return self._neighborhoods