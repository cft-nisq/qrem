"""
refactored functions with new marginals format
"""
from typing import Dict, Tuple, List, Union, Optional, Type
from pathlib import Path
import numpy as np
from qrem.common import math
from qrem.qtypes.characterization_data import CharacterizationData 
import time
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from qrem.characterization import characterization

from qrem.common import io, probability, printer,convert, math as qmath
import qrem.common.povmtools as povmtools
from qrem.common.printer import qprint 
import numpy as np 
from qrem.qtypes.cn_noise_model import CNModelData 
from qrem.characterization.clustering_algorithms import partition_algorithm_v4_cummulative 
from qrem.characterization import neighbors_algorithms
from qrem.common import utils 


def compute_all_marginals(   results_dictionary: Dict[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]],
                                    subsets_of_qubits:List[Tuple],
                                    backup: Union[bool,str,Path]=False,
                                    overwrite: bool = False,
                                    verbose_log: bool = False):
        
    """
    #FBM: add multiprocessing for this task
    #FBM: change subsets_list to be dictionary
    Implements self.compute_marginals for all experimental keys.

    :param subsets_dictionary: list of subsets of qubits for which marginals_dictionary should be calculated
    :param show_progress_bar: if True, shows progress bar. Requires "tqdm" package
    """


    t0 = time.time()
    #[1] Compute normalized marginals for each experimental setting (each circuit label)
    marginals_dictionary = probability.compute_marginals( results_dictionary=results_dictionary,
                                                        subsets_list=subsets_of_qubits,
                                                        print_progress_bar = True,
                                                        normalization=True,
                                                        multiprocessing= True) 
    

    t1 = time.time()

    #[2] Output printing and backups
    if verbose_log:
        printer.qprint("\nCalculating marginals took:",t1-t0)


    if backup:
        path_to_backup =  io.prepare_outfile(   outpath= backup,
                                                overwrite= overwrite,
                                                default_filename = "marginals_dictionary.pkl")
        
        io.save(dictionary_to_save=marginals_dictionary, directory=path_to_backup.parent, custom_filename=path_to_backup.name, overwrite = overwrite)    
    
        if verbose_log:
            printer.qprint("\nBacked up the calculated marginals dictionary into: ",path_to_backup)

    return marginals_dictionary



def averaged_marginals_to_noise_matrix_ddot(
        averaged_marginals: Dict[str, np.ndarray]) -> np.ndarray:
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

    # Register length
    number_of_qubits_of_interest = len(list(averaged_marginals.keys())[0])
    
    noise_matrix = np.zeros((2 ** number_of_qubits_of_interest, 2 ** number_of_qubits_of_interest))

    for input_state, probability_vector in averaged_marginals.items():
        numbers_input = [int(x) for x in list(input_state)]

        #Bad data check - qubits should be in state 0 or 1 for DDoT
        if np.any(np.array(numbers_input) > 1):
            raise ValueError("Input state should be given as a bitstring of 0s and 1s for DDoT")

        try:
            noise_matrix[:, int(input_state, 2)] = probability_vector[:]
        except(ValueError):
            noise_matrix[:, int(input_state, 2)] = probability_vector[:, 0]



    return noise_matrix

def compute_single_noise_matrix_ddot(   experiment_results:Dict[str,Dict[str,int]],
                                        normalized_marginals:Dict[str,Dict[str,int]] ,
                                        subset: Tuple,
                                        fill_missing_columns_with_ideal_result = True,
                                        verbose_log: bool = False) -> Dict[str, np.ndarray]:
    """Noise matrix for subset of qubits, averaged over all other qubits

        :param subset: subset of qubits we are interested in

        By default takes data from self._marginals_dictionary. If data is not present, then it
        calculates marginals_dictionary for given subset
        and updates the class's property self.marginals_dictionary
    """

    averaged_marginals = probability.compute_average_marginal_for_subset(subset=subset,
                                                                            experiment_results=experiment_results, 
                                                                            normalized_marginals = normalized_marginals)
    
    noise_matrix_averaged = averaged_marginals_to_noise_matrix_ddot(averaged_marginals)
    
    
    if not qmath.is_matrix_stochastic(noise_matrix_averaged):
        # qprint_array(noise_matrix_averaged)
        message_now = f"\nNoise matrix not stochastic for subset: {subset}.\n" \
                        f"This most likely means that DDOT collection was not complete " \
                        f"for locality {len(subset)} and some " \
                        f"states were not implemented."
        
        if verbose_log:
            printer.warprint(message_now)
            printer.warprint("\nThat matrix looks like this:")
            printer.warprint(noise_matrix_averaged)

        if fill_missing_columns_with_ideal_result:
            printer.warprint("\nAdding missing columns.")
            for column_index in range(noise_matrix_averaged.shape[0]):
                column_now = noise_matrix_averaged[:, column_index]

                if np.all(column_now == 0):
                    noise_matrix_averaged[column_index, column_index] = 1.0

            printer.warprint("\nNow the matrix looks like this:")
            printer.warprint(noise_matrix_averaged)

        else:
            printer.warprint("\nNoise matrix not stochastic and missing collumns not filled in:")
    return noise_matrix_averaged

def compute_single_noise_matrix_qdot(   experiment_results:Dict[str,Dict[str,int]],
                                        normalized_marginals:Dict[str,Dict[str,int]] ,
                                        subset: Tuple,
                                        estimation_method:str='pls',
                                        fill_missing_columns_with_ideal_result = True,
                                        verbose_log: bool = False) -> Dict[str, np.ndarray]:
    """Noise matrix for subset of qubits, averaged over all other qubits

        :param subset: subset of qubits we are interested in

        By default takes data from self._marginals_dictionary. If data is not present, then it
        calculates marginals_dictionary for given subset
        and updates the class's property self.marginals_dictionary
    """

    POVM= povmtools.compute_subset_POVM(results_dictionary =experiment_results,marginals_dictionary=normalized_marginals,subset=subset,
                                    estimation_method=estimation_method)
        
    
    noise_matrix_averaged = povmtools.get_stochastic_map_from_povm(POVM)
    
    if not qmath.is_matrix_stochastic(noise_matrix_averaged):
        # qprint_array(noise_matrix_averaged)
        message_now = f"\nNoise matrix not stochastic for subset: {subset}.\n" \
                        f"This most likely means that DDOT collection was not complete " \
                        f"for locality {len(subset)} and some " \
                        f"states were not implemented."
        
        if verbose_log:
            printer.warprint(message_now)
            printer.warprint("\nThat matrix looks like this:")
            printer.warprint(noise_matrix_averaged)

        if fill_missing_columns_with_ideal_result:
            printer.warprint("\nAdding missing columns.")
            for column_index in range(noise_matrix_averaged.shape[0]):
                column_now = noise_matrix_averaged[:, column_index]

                if np.all(column_now == 0):
                    noise_matrix_averaged[column_index, column_index] = 1.0

            printer.warprint("\nNow the matrix looks like this:")
            printer.warprint(noise_matrix_averaged)

        else:
            printer.warprint("\nNoise matrix not stochastic and missing collumns not filled in:")
    return noise_matrix_averaged



def compute_noise_matrices_ddot( experiment_results:Dict[str,Dict[str,int]],
                                normalized_marginals:Dict[str,Dict[str,int]] ,
                                subset_of_qubits: List[Tuple] =[],
                                backup: Union[bool,str,Path]=False,
                                overwrite: bool = False,
                                verbose_log: bool = False) -> Dict[Tuple, Dict[str, np.ndarray]]:

    noise_matrices_dictionary = {}

    #Loop over all subsets of qubits. Can be multiprocessed in future if needed
    for subset in tqdm(subset_of_qubits):
        

        noise_matrix_averaged = compute_single_noise_matrix_ddot(   experiment_results=experiment_results,
                                                                    normalized_marginals = normalized_marginals,
                                                                    subset=subset,
                                                                    fill_missing_columns_with_ideal_result = True,
                                                                    verbose_log=verbose_log)
        
        noise_matrices_dictionary[subset] = {'averaged': noise_matrix_averaged}

    return noise_matrices_dictionary


def compute_errors_POVMs(    qubits_subsets: List[Tuple[int]],
                             POVMs_dictionary: Dict[Tuple[int], np.array],
                             distances_types=[('wc', 'classical')],
                             chopping_threshold: Optional[float] = 0) -> Dict[
        str, Dict[str, Dict[Tuple[int], float]]]:

    #JT: a list contaning uniqe sizes of qubits subsets passed to the class

    unique_subsets_sizes = list(np.unique([len(x) for x in qubits_subsets]))

    #JT: a dictionary contaning all ideal, projetive measurements in the computational basis
    # for dimensions corresponding to the unique dimensions passed to the input

    POVMs_compuational_basis = {x: povmtools.computational_projectors(d=2, n=x) for x in
                                unique_subsets_sizes}

    #JT: errors_data is a dictionary that stores results of computation
    #the keys are the first entry of a tuple corresponding to chosen distances (ac/wc)
    #this is a nested dictionary - values will be also dictionaries

    errors_data = {distance_tuple[0]: {} for distance_tuple in distances_types}

    #JT: the second key in errors data is a string classical/quantum"
    #JT: We need to discuess if we want to store it this way


    for distance_tuple in distances_types:
        errors_data[distance_tuple[0]][distance_tuple[1]] = {}

    #JT: a loop to compute distances

    for distance_tuple in distances_types:



        #JT: information about computation is printed

        qprint("\nCalculating errors of type:", distance_tuple, 'red')

        #JT: an internal loop over subsets

        for subset in tqdm(qubits_subsets):

           
            #JT: function calculating specified distances, detailed comments in function distances and povmtools
            #functions involved include average_distance_POVMs and  operational_distance_POVMs

            distance_now = povmtools.calculate_distance_between_POVMs(POVM_1=POVMs_dictionary[subset],
                                                                        POVM_2=
                                                                        POVMs_compuational_basis[
                                                                            len(subset)],
                                                                        distance_type_tuple=distance_tuple)

            #JT: treshold check

            if distance_now >= chopping_threshold:
                errors_data[distance_tuple[0]][distance_tuple[1]][subset] = distance_now

        qprint("DONE")

    return errors_data


def compute_correlations_data_pairs(qubit_indices,
                                    POVMs_dictionary: Dict[Tuple[int], np.array],
                                    distances_types=[('ac', 'classical')],
                                    chopping_threshold: Optional[float] = 0) -> Dict[
    str, Dict[str, np.array]]:

    qubit_pairs = [(qi, qj) for qi in qubit_indices for qj in qubit_indices if qj > qi]

    number_of_qubits = len(qubit_indices)

    if np.max(qubit_indices) > number_of_qubits:
        mapping = povmtools.get_enumerated_rev_map_from_indices(qubit_indices)
    else:
        mapping = {qi: qi for qi in qubit_indices}

    correlations_data = {distance_tuple[0]: {} for distance_tuple in distances_types}

    for distance_tuple in distances_types:
        correlations_data[distance_tuple[0]][distance_tuple[1]] = np.zeros((number_of_qubits,
                                                                            number_of_qubits),
                                                                            dtype=float)

    for distance_tuple in distances_types:
        qprint("\nCalculating correlations of type:", distance_tuple, 'red')

        (distance_type, correlations_type) = distance_tuple

        classical_correlations = False
        if correlations_type.lower() in ['classical', 'diagonal']:
            classical_correlations = True

        for (qi, qj) in tqdm(qubit_pairs):
            qi_mapped, qj_mapped = mapping[qi], mapping[qj]

            povm_2q_now = POVMs_dictionary[(qi, qj)]

            c_i_j, c_j_i = povmtools.find_correlations_coefficients(povm_2q=povm_2q_now,
                                                                    distance_type=distance_type,
                                                                    classical=classical_correlations,
                                                                    direct_optimization=True)

            if c_i_j >= chopping_threshold:
                correlations_data[distance_type][correlations_type][qi_mapped, qj_mapped] = c_i_j

            if c_j_i >= chopping_threshold:
                correlations_data[distance_type][correlations_type][qj_mapped, qi_mapped] = c_j_i
        qprint("DONE")

    return correlations_data 

def compute_reduced_POVMs_and_noise_matrices(characterization_data: Type[CharacterizationData] ,subset_of_qubits:List):

    if characterization_data.experiment_type.upper() == 'DDOT':
   

        noise_matrices_dictionary = characterization.compute_noise_matrices_ddot(experiment_results=characterization_data.results_dictionary,normalized_marginals=characterization_data.marginals_dictionary,subset_of_qubits=subset_of_qubits)

        POVMs_dictionary = povmtools.establish_POVMs_dictionary_from_noise_matrices_dictionary(noise_matrices_dictionary=noise_matrices_dictionary)
 
      #marginals and POVM calculation for QDOT
    elif characterization_data.experiment_type.upper() == 'QDOT':


        POVMs_dictionary = povmtools.compute_subsets_POVMs_averaged(results_dictionary=characterization_data.results_dictionary,marginals_dictionary=characterization_data.marginals_dictionary,subsets_of_qubits=subset_of_qubits,
                                                           show_progress_bar=True,
                                                           estimation_method='PLS')
        
        
        noise_matrices_dictionary = povmtools.compute_noise_matrices_from_POVMs(subsets_of_qubits=subset_of_qubits,POVM_dictionary=POVMs_dictionary)

    return noise_matrices_dictionary, POVMs_dictionary

       


      


    return characterization_data

def Overlap(s1:str,s2:str):
    """
    Function used to create dictionary of overlapes between different Pauli states. It's needed to compute coherence indicator
    Input: s1,s2 two symbols, each corresponding to one of the Pauli states, as in the SeparableCircuitsCreator class 

    Output: value of squared scalar product between the states     
    """
    if s1==s2:
        return 1
    elif (s1=='2' and s2=='3') or (s2=='2' and s1=='3') or (s1=='4' and s2 =='5') or (s2=='4' and s1 =='5'):
        return 0
    else:
        return 0.5

def compute_indicator_normalization(dim,setting1,setting2,overlap_dic):
    """
    A function computing normalization factor of coherence indicator. As for now works for two-qubit reduced POVMs
    
    Input: 
    dim - dimension of the reduced subspace
    setting1,setting2 - strings consisting of two symbols each, corresponding to an inout Pauli eigenstate
    overlap_dic - a dictionary consisting of squared scalar product of two Pauli eigenstates   
    """
    return dim*np.sqrt((2*(1- overlap_dic[setting1[1]+setting2[1]]*overlap_dic[setting1[0]+setting2[0]])))

def compute_pauli_marginals(marginals_dictionary, subsets_list):
    """
    Dictionaries needed to perform computations

    setting dictionary - keys correspond to alphabet enocding input states
                         items correspond to unnormalized probability distributions for a given input state on subset of qubits
    normalization dictionary
                         keys correspond to alphabet encoding input states
                         items correspond to number of times that a particular input state appear in marginals

    """
    setting_dictionary = {}
    normalisation_dictionary = {}
    for i in range(2, 6):
        for j in range(2, 6):
            setting_dictionary[str(i) + str(j)] = np.array([0., 0., 0., 0.])
            normalisation_dictionary[str(i) + str(j)] = 0

    measurement_settings = marginals_dictionary.keys()
    """
    Computation of marginal probability distributions for X,Y Pauli input states  
    """
    Pauli_subset_dictionary = {}
    for subset in subsets_list:
        setting_dictionary = setting_dictionary.fromkeys(setting_dictionary, [0., 0., 0., 0.])
        normalisation_dictionary = normalisation_dictionary.fromkeys(normalisation_dictionary, 0)
        for setting in measurement_settings:
            s1 = setting[subset[0]]
            s2 = setting[subset[1]]
            if s1 != '0' and s1 != '1' and s2 != '0' and s2 != '1':
                setting_dictionary[s1 + s2] += marginals_dictionary[setting][subset]
                normalisation_dictionary[s1 + s2] += 1

        Pauli_subset_dictionary[subset] = [setting_dictionary, normalisation_dictionary]
    return Pauli_subset_dictionary



def compute_coherence_indicator(marginals_dictionary, subset_list,settings_list=['2','3','4','5']):
    """
    Computation of coherence indicator 

    TVD between probability distributions generated by different input X,Y Pauli eigenstates of two qubits
    """
    pauli_subset_dictionary = compute_pauli_marginals(marginals_dictionary,subset_list)

    """
    Creation of overlap dicitionary for the Pauli case
    """
   

    overlap_dictionary={}
    settings_list_coherence_indicator = []
    for i in settings_list:
        for j in settings_list:
            overlap_dictionary[i+j] = Overlap(i,j)
            settings_list_coherence_indicator.append(i+j)

   

    # dictionary storing values of coherence indicator
    indicator_dic = {}

  

    for keys, elements in pauli_subset_dictionary.items():
        tvd_value = []
        tvd_settings = []

        for i in range(len(settings_list_coherence_indicator)):
            for j in range(i + 1, len(settings_list_coherence_indicator)):
                s1 = settings_list_coherence_indicator[i]
                s2 = settings_list_coherence_indicator[j]
                indicator = math.compute_TVD(elements[0][s1] / elements[1][s1], elements[0][s2] / elements[1][s2])
                indicator = indicator / (compute_indicator_normalization(2, s1, s2, overlap_dictionary))
                tvd_value.append(indicator)
                tvd_settings.append((settings_list_coherence_indicator[i], settings_list_coherence_indicator[j]))
        indicator_dic[keys] = [tvd_value, tvd_settings]

    return  indicator_dic



def perform_noise_model_reconstruction_routine(characterization_data_container:Type[CharacterizationData],experiment_type:str = 'DDOT',find_neighbors:bool =False):

    
        
    noise_model_list =[]

    all_clusters_sets_list = list(characterization_data_container.clusters_neighbors_sets_dictionary.keys())
        
    for cluster_list in all_clusters_sets_list:

        model_noise_matrix_dictionary = {}

        cn_arrangement = characterization_data_container.clusters_neighbors_sets_dictionary[tuple(cluster_list)]

        if experiment_type =='DDOT':
                
       



            for subset in cluster_list:

                if cn_arrangement[subset]  == None:

                    if subset in characterization_data_container.noise_matrices_dictionary.keys():

                        model_noise_matrix_dictionary[subset]  = characterization_data_container.noise_matrices_dictionary[subset]
                    
                    else:

                        model_noise_matrix_dictionary[subset] = {'averaged': compute_single_noise_matrix_ddot(experiment_results=characterization_data_container.results_dictionary,normalized_marginals=characterization_data_container.marginals_dictionary,subset=subset) }


                else:

                    model_noise_matrix_dictionary[subset] =  compute_noise_matrix_dependent(characterization_data=characterization_data_container,qubits_of_interest=subset,neighbors_of_interest=cn_arrangement[subset]) 

                        
                        


          

               
        elif experiment_type =='QDOT':

            

            #POVMs_dictionary = povmtools.compute_subsets_POVMs_averaged(subsets_of_qubits=cluster_list,
                                                           #show_progress_bar=True,
                                                           #estimation_method='PLS')
        
        
            #noise_matrices_dictionary = povmtools.compute_noise_matrices_from_POVMs(subsets_of_qubits=cluster_list,POVM_dictionary=POVMs_dictionary)

            for subset in cluster_list:

                if cn_arrangement[subset]  == None:

                    if subset in characterization_data_container.noise_matrices_dictionary.keys():

                        model_noise_matrix_dictionary[subset]  = characterization_data_container.noise_matrices_dictionary[subset]
                    
                    else:

                        model_noise_matrix_dictionary[subset] =  {'averaged':compute_single_noise_matrix_qdot(experiment_results=characterization_data_container.results_dictionary,normalized_marginals=characterization_data_container.marginals_dictionary,subset=subset)}


                else:

                    model_noise_matrix_dictionary[subset] =  compute_noise_matrix_dependent(characterization_data=characterization_data_container,qubits_of_interest=subset,neighbors_of_interest=cn_arrangement[subset],experiment_type='qdot') 

           
       

        noise_model = CNModelData(next(iter(characterization_data_container.results_dictionary.keys())))

        noise_model.set_clusters_tuple(clusters_tuple=tuple(cluster_list))

        noise_model.clusters_neighborhoods = cn_arrangement

        noise_model.set_noise_matrices_dictionary(noise_matrices_dictionary=model_noise_matrix_dictionary)

       


            
        noise_model_list.append(noise_model)

    return noise_model_list



def perform_clustering_routine(characterization_data, number_of_qubits:int,clustering_functions_parameters: Dict ,perform_neighbors_search:bool = False):

    all_clusters_sets_dictionary = {tuple([(qi,) for qi in range(number_of_qubits)]):None}
  
    sizes_clusters = clustering_functions_parameters['sizes_clusters']

    distance_type = clustering_functions_parameters['distance_type']
    
    correlations_type = clustering_functions_parameters['correlations_type']

    alpha_hyperparameters = clustering_functions_parameters['alpha_hyperparameters'] 
    
    correlations_table = characterization_data.correlation_coefficients_dictionary[distance_type][correlations_type]

    all_clusters_sets_neighbors_dictionary = {tuple((qi,) for qi in range(number_of_qubits)):{(qj,) :None for qj in range(number_of_qubits)}}

    
        #product noise model is added to clusters list
    for max_cluster_size in sizes_clusters:
        qprint("\nCurrent max cluster size:", max_cluster_size, 'red')

        for alpha_multiplier in alpha_hyperparameters:
                
            clustering_function_arguments = {'alpha_multiplier': alpha_multiplier}

                  
            clusters_tuples_now, score = partition_algorithm_v4_cummulative(correlations_table = correlations_table,alpha=alpha_multiplier,C_maxsize=max_cluster_size,N_alg=1000,printing=False,disable_pb=True, drawing=False) 

            clusters_tuples_now = tuple(tuple(cluster) for cluster in clusters_tuples_now)

            if perform_neighbors_search:
                    
                qprint('find_all_neighborhoods starts')

                qubit_indices= [i for i in range(number_of_qubits)]
                                
                neighbors = neighbors_algorithms.find_all_neighborhoods(characterization_data=characterization_data,qubit_indices=qubit_indices,maximal_size= max_cluster_size+2,clusters_tuple=clusters_tuples_now)

                qprint('find_all_neighborhoods ends')
               
                        
            if clusters_tuples_now in all_clusters_sets_dictionary.keys():
                    
                all_clusters_sets_dictionary[clusters_tuples_now].append(
                        (max_cluster_size, clustering_function_arguments['alpha_multiplier'], score))
            else:
                
                all_clusters_sets_dictionary[clusters_tuples_now] = [
                        (max_cluster_size, clustering_function_arguments['alpha_multiplier'], score)]
                
            if perform_neighbors_search:
                    
                all_clusters_sets_neighbors_dictionary[clusters_tuples_now] = neighbors

           
            else:
                all_clusters_sets_neighbors_dictionary[clusters_tuples_now] = {cluster_tuple: None for cluster_tuple in clusters_tuples_now}
    

    return all_clusters_sets_neighbors_dictionary


def compute_noise_matrix_dependent(characterization_data,
                                        qubits_of_interest: Tuple[int],
                                        neighbors_of_interest: Union[Tuple[int], None],
                                        experiment_type:str='ddot') \
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

        noise_matrices_dictionary= {}

        if len(neighbors_of_interest) == 0 or neighbors_of_interest is None:
            cluster_tuple_now = convert.qubit_indices_to_keystring(qubits_of_interest)
            cluster_tuple_now = tuple(qubits_of_interest)
            if cluster_tuple_now in  characterization_data.noise_matrices_dictionary.keys() and 'averaged' in characterization_data.noise_matrices_dictionary[cluster_tuple_now].keys():
                return {'averaged':characterization_data.noise_matrices_dictionary[cluster_tuple_now]['averaged']}
            else:
                if experiment_type == 'ddot': 
                    noise_matrix =  compute_single_noise_matrix_ddot(experiment_results=characterization_data.results_dictionary,normalized_marginals=characterization_data.marginals_dictionary,subset=cluster_tuple_now)
                elif experiment_type == 'qdot':
                    noise_matrix =  compute_single_noise_matrix_qdot(experiment_results=characterization_data.results_dictionary,normalized_marginals=characterization_data.marginals_dictionary,subset=cluster_tuple_now)
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
        if experiment_type == 'ddot': 
            big_lambda =  compute_single_noise_matrix_ddot(experiment_results=characterization_data.results_dictionary,normalized_marginals=characterization_data.marginals_dictionary,subset=tuple(all_qubits))
        elif experiment_type=='qdot':
            noise_matrix =  compute_single_noise_matrix_qdot(experiment_results=characterization_data.results_dictionary,normalized_marginals=characterization_data.marginals_dictionary,subset=tuple(all_qubits))
             

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

        
        
        if cluster_tuple_now not in characterization_data.noise_matrices_dictionary.keys():
            # If there is no entry for our cluster in the dictionary, we create it and add
            # averaged noise matrix
            averaged_noise_matrix = np.zeros(
                (2 ** number_of_qubits_of_interest, 2 ** number_of_qubits_of_interest))
            for neighbors_state_bitstring in conditional_noise_matrices.keys():
                averaged_noise_matrix += conditional_noise_matrices[neighbors_state_bitstring]
            averaged_noise_matrix /= 2 ** number_of_qubits_of_interest
            noise_matrices_dictionary[cluster_tuple_now] = {'averaged': averaged_noise_matrix}
        else:
            noise_matrices_dictionary[cluster_tuple_now] ={'averaged': characterization_data.noise_matrices_dictionary[cluster_tuple_now]['averaged']}
        noise_matrices_dictionary[cluster_tuple_now][neighbours_tuple_now] = conditional_noise_matrices

        #return noise_matrices_dictionary[cluster_tuple_now][neighbours_tuple_now]
        
        return noise_matrices_dictionary[cluster_tuple_now]





