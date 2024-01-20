import numpy as np
from typing import Dict, Optional, Tuple, List, Union 
from tqdm import tqdm 

from qrem.common import convert 
from qrem.common import math as qmath  
from qrem.common import utils
from qrem.characterization import characterization 
import numpy as np 

## What is needed
## data: qubit indices
## data: correlations data
## data: _neighborhoods
## data: _noise_matrices_dictionary
## functions: get_noise_matrix_dependent
## function: compute_subset_noise_matrices_averaged
## functions: _compute_noise_matrix_dependent



def find_neighbors_of_cluster_holistic(     cluster: List[int],
                                            characterization_data,
                                            qubit_indices,
                                            
                                            maximal_size: int,
                                            chopping_threshold: Optional[float] = 0.,
                                            correlation_threshold:float = 0.001) -> List[int]:
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

        correlations_data = characterization_data.correlation_coefficients_dictionary

        neighborhoods ={}

        #JT this is added as a treshold


        for qi in qubit_indices:
            if qi not in cluster:
                compute_noise_matrix = False

                #JT this is added to avoid extensive computation of noise matrices

                for qj in cluster: 
                    corr_i_j = correlations_data['worst_case']['classical'][qj, qi]

                    #
                    if corr_i_j > correlation_threshold:
                        
                        compute_noise_matrix = True
                        
                        break

                #if the corresponding correlation coefficient is above threshold computation of the noise matrix will be performed 
                if compute_noise_matrix:    
                    lam_ci_j = characterization.compute_noise_matrix_dependent(characterization_data=characterization_data,qubits_of_interest=cluster,neighbors_of_interest=[qi])[(qi,)]  
                    diff_ci_j = lam_ci_j['0'] - lam_ci_j['1']
                    correlation_ci_j = 1 / 2 * np.linalg.norm(diff_ci_j, ord=1)
                    potential_neighbours.append([qi, correlation_ci_j])

        sorted_neighbours = sorted(potential_neighbours, key=lambda x: x[1], reverse=True)

        neighbors_list = sorted(
            [sorted_neighbours[i][0] for i in range(int(np.min([size_cut, len(sorted_neighbours)]))) if
             chopping_threshold < sorted_neighbours[i][1]])

        cluster_key = convert.qubit_indices_to_keystring(cluster)

        neighborhoods[cluster_key] = neighbors_list

        return neighborhoods

def find_all_neighborhoods_holistic(     characterization_data,
                                         qubits_indices,
                                         clusters_tuple,    
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

        neighborhoods = {}
        
        range_clusters = range(len(clusters_tuple))

        if show_progress_bar:
            range_clusters = tqdm(range_clusters)

        for index_cluster in range_clusters:
            cluster = clusters_tuple[index_cluster]
            #neighborhoods[
            #    convert.qubit_indices_to_keystring(cluster)] = find_neighbors_of_cluster_holistic(
            #    cluster=cluster, characterization_data=characterization_data, maximal_size=maximal_size, chopping_threshold=chopping_threshold,qubit_indices=qubits_indices)
            neighborhoods[(cluster)] = tuple(find_neighbors_of_cluster_holistic(
               cluster=cluster, characterization_data=characterization_data, maximal_size=maximal_size, chopping_threshold=chopping_threshold,qubit_indices=qubits_indices)[convert.qubit_indices_to_keystring(cluster)])

        return neighborhoods



      




def find_all_neighborhoods( characterization_data,
                           clusters_tuple,
                            qubit_indices,
                            maximal_size: int,
                            method: Optional[str] = 'holistic',
                            method_kwargs: Optional[dict] = None):

    #if method == 'pairwise':
    #    if method_kwargs is None:
    #        default_kwargs = {'neighbors_threshold': 0.01}
    #        method_kwargs = default_kwargs

    #    method_kwargs['maximal_size'] = maximal_size
    #    method_kwargs['cluster_list'] = cluster_list
    #    neighborhoods = find_all_neighborhoods_pairwise(**method_kwargs)

    if method == 'holistic':
        if method_kwargs is None:
            default_kwargs = {'chopping_threshold': 0.01,
                                'show_progress_bar': True}
            method_kwargs = default_kwargs
        method_kwargs['maximal_size'] = maximal_size
        method_kwargs['characterization_data'] = characterization_data
        neighborhoods = find_all_neighborhoods_holistic(characterization_data=characterization_data,qubits_indices=qubit_indices,maximal_size=maximal_size,chopping_threshold=0.01,clusters_tuple=clusters_tuple)                                                  

    return neighborhoods    