import sys
import os
import random
import pickle 

from tqdm import tqdm
import numpy as np

import qrem.common.providers.ibmutils.data_converters


from qrem.common import io 
import qrem.functions_qrem.functions_data_analysis as fdt 

from qrem.noise_characterization.tomography_design.overlapping.DOTMarginalsAnalyzer import DOTMarginalsAnalyzer 
from qrem.noise_characterization.tomography_design.overlapping.QDTMarginalsAnalyzer import QDTMarginalsAnalyzer

from qrem.noise_characterization.base_classes.OverlappingTomographyBase import OverlappingTomographyBase
from qrem.noise_characterization.tomography_design.overlapping.SeparableCircuitsCreator import SeparableCircuitsCreator
from qrem.noise_characterization.data_analysis.InitialNoiseAnalyzer import InitialNoiseAnalyzer
from typing import Dict, Tuple, List


#from qrem.backends_support.qiskit import qiskit_utilities
from qrem.functions_qrem import functions_data_analysis as fdt

from qrem.noise_model_generation.CN.NoiseModelGenerator import NoiseModelGenerator
from qrem.functions_qrem import functions_coherence_analysis as fca

import qrem.common.math as qrem_math
from qrem.common.printer import qprint, qprint_array
from qrem.types import CNModelData
from qrem.cn import simulation as cnsimulation 
from datetime import date
from typing import Tuple, Dict, List, Optional
import copy
import time
import re
from qrem.common.printer import qprint


import statistics 

def change_dictionary_format(noise_matrix:Dict) -> Dict:

    new_format ={}
    
    for key, matrix in noise_matrix.items():

        if key != 'averaged':
            
        
                   
            new_index = [int(character) for character in key]
        
            new_format[tuple(new_index)] = matrix
        
        else:

            new_format[key] = matrix

    return new_format


def change_state_dependent_noise_matrix_format(noise_matrix:Dict) -> Dict:

    """
    Function transforming old to new format of state dependent matrix 

    """

    state_dependent_noise_matrix_in_new_format = {}


    for key in noise_matrix.keys():
        
        if key != 'averaged':
            
        
            for neighbors_state, state_noise_matrix in noise_matrix[key].items():
        
                new_index = [int(character) for character in neighbors_state]
        
                state_dependent_noise_matrix_in_new_format[tuple(new_index)] = state_noise_matrix
            
        elif key == 'averaged':

            state_dependent_noise_matrix_in_new_format[key] = noise_matrix[key]
    
    return(state_dependent_noise_matrix_in_new_format)
    


def change_state_dependent_noise_matrices_dictionary_format(noise_matrices_dictionary:Dict) -> Dict:

    """
    Function transforming old to new format of state dependent matrix 

    """

    state_dependent_noise_matrices_dictionary_in_new_format = {}


    for key, item in noise_matrices_dictionary.items():

        state_dependent_noise_matrices_dictionary_in_new_format[key] = change_state_dependent_noise_matrix_format(noise_matrix=item)
        
        
    
    return(state_dependent_noise_matrices_dictionary_in_new_format)

def change_format_of_cn_dictionary(cn_dictionary:Dict)->Dict:

    pattern='q[0-9]+'
    cn_dictionary_new = {}
    for key,entry in cn_dictionary.items():
        index_list = re.findall(pattern,key)

        cluster_arrangement = []
        for index in index_list:
            cluster_arrangement.append(int(index[1:]))
        if entry == []:
            new_key = None
        else:
            new_key = tuple(entry)
        cn_dictionary_new[tuple(cluster_arrangement)] = new_key
    return cn_dictionary_new

def divide_data_into_characterization_benchmark_coherence_witness_data(results_dictionary:Dict[str,Dict[str,int]],marginals_dictionary:Dict[str,Dict[str,int]],ground_states_list=None, coherence_witnesses_list = None ):

    characterization_data_dictionary = {}
    
    
    if ground_states_list != None:
        benchmarks_results_dictionary = {}
        benchmarks_marginals_dictionary = {}
        characterization_results_dictionary = copy.copy(results_dictionary)
        characterization_marginals_dictionary = copy.copy(marginals_dictionary)
        
        for state in ground_states_list:
            benchmarks_results_dictionary[state] = copy.copy(results_dictionary[state])
            benchmarks_marginals_dictionary[state] =copy.copy( marginals_dictionary[state])
            del characterization_results_dictionary[state]
            del characterization_marginals_dictionary[state]
        
        characterization_data_dictionary['benchmarks_results_dictionary'] = benchmarks_results_dictionary
        characterization_data_dictionary['benchmarks_marginals_dictionary'] = benchmarks_marginals_dictionary 

    else:
        characterization_results_dictionary = results_dictionary
        characterization_marginals_dictionary = marginals_dictionary
        benchmarks_marginals_dictionary  = {}
        benchmarks_results_dictionary = {}
    
    if coherence_witnesses_list != None:
        coherence_witness_results_dictionary = {}
        coherence_witness_marginals_dictionary = {}
        for setting in coherence_witnesses_list:
            setting_string = ''
            
            for element in setting:
                setting_string = setting_string+ str(element)
            coherence_witness_results_dictionary[setting_string] = copy.copy(results_dictionary[setting_string])
            coherence_witness_marginals_dictionary[setting_string] =copy.copy( marginals_dictionary[setting_string])
            del characterization_results_dictionary[setting_string]
            del characterization_marginals_dictionary[setting_string]
        characterization_data_dictionary['coherence_witnesses_results_dictionary'] = coherence_witness_results_dictionary
        characterization_data_dictionary['coherence_witnesses_marginals_dictionary'] = coherence_witness_marginals_dictionary
    
    characterization_data_dictionary['characterization_results_dictionary'] = characterization_results_dictionary
    characterization_data_dictionary['characterization_marginals_dictionary'] = characterization_marginals_dictionary

    return characterization_data_dictionary 

def compute_marginals_dictionary(results_dictionary:Dict[str,Dict[str,int]],marginals_list:List,experiment_type:str='DDOT', data_directory:str ='', name_string=''):

    if experiment_type == 'DDOT':
        marginals_analyzer = DOTMarginalsAnalyzer(results_dictionary_ddot=results_dictionary)
        marginals_analyzer.compute_all_marginals(subsets_dictionary=marginals_list,show_progress_bar=True,multiprocessing=True)
        
        
            #marginals  calculation for QDOT
    elif experiment_type == 'QDOT':
        marginals_analyzer = QDTMarginalsAnalyzer(results_dictionary,experiment_name='QDT')
        marginals_analyzer.initialize_labels_interpreter(interpreter='PAULI')
        marginals_analyzer.compute_all_marginals(subsets_dictionary=marginals_list,show_progress_bar=True,multiprocessing=True)
        
    marginals_dictionary=marginals_analyzer.marginals_dictionary
  
    
    return marginals_dictionary

def compute_reduced_POVMs_and_noise_matrices(results_dictionary:Dict[str,Dict[str,int]],marginals_dictionary:Dict[str,Dict[str,int]] ,subset_of_qubits:List,experiment_type:str='DDOT'):

    if experiment_type == 'DDOT':
        marginals_analyzer = DOTMarginalsAnalyzer(results_dictionary_ddot=results_dictionary,marginals_dictionary=marginals_dictionary)


        marginals_analyzer.compute_subset_noise_matrices_averaged(subset_of_qubits)

        POVMs_dictionary_diag= marginals_analyzer._noise_matrices_dictionary

        #POVMs are established form noise matrices
        POVMs_dictionary = {}
        for key, entry in POVMs_dictionary_diag.items():
            list_of_matrices=[]
            for matrix in POVMs_dictionary_diag[key]['averaged']:
                list_of_matrices.append(np.diag(matrix))

            POVMs_dictionary[key]=list_of_matrices

      #marginals and POVM calculation for QDOT
    elif experiment_type == 'QDOT':
        marginals_analyzer = QDTMarginalsAnalyzer(results_dictionary=results_dictionary,experiment_name='QDT',marginals_dictionary=marginals_dictionary)

        marginals_analyzer.initialize_labels_interpreter(interpreter='PAULI')

        #calculate reduced POVMs
        marginals_analyzer.compute_subsets_POVMs_averaged(subsets_of_qubits=subset_of_qubits,
                                                           show_progress_bar=True,
                                                           estimation_method='PLS')
        POVMs_dictionary = marginals_analyzer._POVM_dictionary

        marginals_analyzer.compute_noise_matrices_from_POVMs(subsets_of_qubits=subset_of_qubits,
                                                     show_progress_bar=True)

    noise_matrices_dictionary = marginals_analyzer._noise_matrices_dictionary

    return POVMs_dictionary, noise_matrices_dictionary

def perform_clustering_routine(results_dictionary:Dict[str,Dict[str,int]],marginals_dictionary:Dict[str,Dict[str,int]], noise_matrices_dictionary:Dict[str,Dict[str,np.array]],correlations_data:np.array, number_of_qubits:int,clustering_functions_parameters: Dict ,perform_neighbors_search:bool = False):

    noise_model_analyzer = NoiseModelGenerator(results_dictionary=results_dictionary,
                                           marginals_dictionary=marginals_dictionary,
                                           noise_matrices_dictionary=noise_matrices_dictionary,
                                           correlations_data=correlations_data
                                           )

    all_clusters_sets_dictionary = {tuple([(qi,) for qi in range(number_of_qubits)]):None}

    #clustering starts 

    
    sizes_clusters = clustering_functions_parameters['sizes_clusters']

    distance_type = clustering_functions_parameters['distance_type']
    
    correlations_type = clustering_functions_parameters['correlations_type']

    alpha_hyperparameters = clustering_functions_parameters['alpha_hyperparameters'] 
    
    correlations_table = correlations_data[distance_type][correlations_type]

    all_clusters_sets_neighbors_dictionary = {tuple((qi,) for qi in range(number_of_qubits)):{(qj,) :None for qj in range(number_of_qubits)}}

    
        #product noise model is added to clusters list
    for max_cluster_size in sizes_clusters:
        qprint("\nCurrent max cluster size:", max_cluster_size, 'red')

        for alpha_multiplier in alpha_hyperparameters:
                
            clustering_function_arguments = {'alpha_multiplier': alpha_multiplier}

            clusters_tuples_now, score = noise_model_analyzer.compute_clusters(correlations_table=correlations_table,
                                                                            maximal_size=max_cluster_size,
                                                                           method_kwargs=clustering_function_arguments)
                           
            if perform_neighbors_search:
                    
                qprint('find_all_neighborhoods starts')
                                
                neighbors = noise_model_analyzer.find_all_neighborhoods(maximal_size= max_cluster_size+2)

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


def perform_noise_model_reconstruction_routine(results_dictionary:Dict[str,Dict[str,int]],number_of_qubits:int,all_clusters_sets_neighbors_dictionary:Dict,experiment_type:str = 'DDOT',find_neighbors:bool =False):

        
    noise_model_list =[]

    all_clusters_sets_list = list(all_clusters_sets_neighbors_dictionary.keys())
        
    for cluster_list in all_clusters_sets_list:

        if experiment_type =='DDOT':
                
            marginals_analyzer_noise_model = DOTMarginalsAnalyzer(results_dictionary_ddot=results_dictionary)

            new_noise_matrix_dictionary = {}

            cn_arrangement = all_clusters_sets_neighbors_dictionary[tuple(cluster_list)]

            if find_neighbors:
                
                cn_arrangement=change_format_of_cn_dictionary(cn_dictionary=cn_arrangement)



            for subset in cluster_list:

                if cn_arrangement[subset]  == None:

                    new_noise_matrix_dictionary[subset] =  {'averaged':marginals_analyzer_noise_model._compute_noise_matrix_averaged(subset=subset)}


                else:

                    new_noise_matrix_dictionary[subset] =  marginals_analyzer_noise_model._compute_noise_matrix_dependent(qubits_of_interest=subset,neighbors_of_interest=cn_arrangement[subset]) 

                        
                    new_noise_matrix_dictionary[subset]['averaged'] =  marginals_analyzer_noise_model._compute_noise_matrix_averaged(subset=subset)
                        



            marginals_analyzer_noise_model.compute_subset_noise_matrices_averaged(subsets_list=cluster_list,show_progress_bar=True)
                
            noise_matrices_dictionary = marginals_analyzer_noise_model.noise_matrices_dictionary

               
        elif experiment_type =='QDOT':
                
            marginals_analyzer_noise_model  = QDTMarginalsAnalyzer(results_dictionary=results_dictionary,experiment_name='QDT')
                
            marginals_analyzer_noise_model.initialize_labels_interpreter(interpreter='PAULI')

            #calculate reduced POVMs
            marginals_analyzer_noise_model.compute_subsets_POVMs_averaged(subsets_of_qubits=cluster_list, #cluster_substets
                                                            show_progress_bar=True,
                                                            estimation_method='PLS')
                
            POVMs_dictionary = marginals_analyzer_noise_model ._POVM_dictionary

            marginals_analyzer_noise_model .compute_noise_matrices_from_POVMs(subsets_of_qubits=cluster_list,
                                                        show_progress_bar=True)
                
            noise_matrices_dictionary = marginals_analyzer_noise_model.noise_matrices_dictionary

            new_noise_matrix_dictionary = {}

            for key, noise_matrix in noise_matrices_dictionary.items():
                    
                if noise_matrix != {}:
                        
                    new_noise_matrix_dictionary[key] =noise_matrix['averaged']

            #for test purposes

        

        noise_model = CNModelData(number_of_qubits)

        noise_model.set_clusters_tuple(clusters_tuple=tuple(cluster_list))

        noise_model.clusters_neighborhoods = cn_arrangement

        noise_model.set_noise_matrices_dictionary(noise_matrices_dictionary=new_noise_matrix_dictionary)

       

        for cluster, noise_matrix in noise_model.noise_matrices.items():

            noise_model.noise_matrices[cluster] = change_dictionary_format(noise_matrix=noise_matrix)

            
        noise_model_list.append(noise_model)

    return noise_model_list


def perform_POVMs_errors_and_correlation_coefficients_computation(results_dictionary:Dict,marginals_dictionary:Dict,POVMs_dictionary:Dict,distances_types:List = [('worst_case','classical')] ):

    noise_analyzer = InitialNoiseAnalyzer(results_dictionary=results_dictionary,
                                      marginals_dictionary=marginals_dictionary,
                                      POVM_dictionary=POVMs_dictionary
                                        )
    
    number_of_qubits = len(next(iter(results_dictionary.keys()))) 

    qubits_subset =[(qi,) for qi in range(number_of_qubits)]
    
    noise_analyzer.compute_errors_POVMs(qubits_subsets=qubits_subset,distances_types=distances_types)

    distances_types_correlations = [('worst_case', 'classical')]#,('worst_case', 'quantum'),('average case', 'classical'),('average case', 'quantum') ]

       
    
    noise_analyzer.compute_correlations_data_pairs(qubit_indices=range(number_of_qubits),
                                              distances_types=distances_types_correlations)

    correlations_data = noise_analyzer.correlations_data

    return {'POVMs_error_dictionary': noise_analyzer._errors_data_POVMs, 'correlations_data_dictionary' : correlations_data}






    









def execute_characterization_workflow(results_dictionary:Dict[str, Dict[str, int]], marginals_dictionary:Dict[str,Dict[Tuple, np.ndarray ]] = None, experiment_type: str = 'DDOT',   data_directory: str ='',number_of_benchmark_circuits: int = 0  , return_old_mitigation_data: bool = False, perform_clustering: bool =True, ground_states_list = None, coherence_witnesses_list = None,  name_id ='', perform_noise_model_reconstruction = True, find_neighbors = False):

    """
    A function that performs analysis of experimental data and provides various characteristics of readout noise as well as reconstructs a CN redout noise model. It returns a list of reconstructed CNModelData objects with noise models data.
    In addition, it saves data about correlation coefficients and reduced POVMs into files.    

    
    It is used to:

    1) Perform full characterization workflow
    2) As an initial step before running mitigation routine 


    Parameters
    ----------
    results_dictionary: dictionary (key - string encoding experimental setting , value - dictionary with experimental results i.e. key - string encoding outcome bits, values- number of occurrences of a given outcome  )
        Stores results of a characterization experiment. 

    marginals_dictionary: dictionary (key - string encoding experimental setting , value - dictionary with experimental results i.e. key - tuple encoding qubits in a marginal values- marginal probability distribution for a given marginal  ) 
        Optional parameter that allows to input precomputed marginal probability distributions into characterization routine. Stores marginal probability distributions for marginals. 

    experiment_type: string
        String encoding type of characterization experiment. Possible values: Diagonal detector overlapping tomography -  'DDOT' Quantum detector overlapping tomography - 'QDOT' 

    data_directory: string
        A string specifying path where results of characterization procedure are saved. 

    number_of_benchmark_circuits : integer
        The number of experiments from results_dictionary to be used in benchmarks. For number_of_benchmark_circuits > 0 the corresponding cumber of entries from results_dictionary is excluded from characterization procedure.  

    return_old_mitigation_data: bool
        Specifies whether old mitigation data are returned. Probably it should be excluded from the final release. It stays for now for debugging purposes. 
    
    perform_clustering: bool
       To be removed in final release: Specifies whether clustering algorithms to find noise model 
        

    
    Returns
    -------
    characterization_data_dictionary: dictionary
        A dictionary with results of characterization procedure
        
        Structure:
            Key: 'noise_models_list' - a list of noise models that is a result of clustering 
            
            Key: 'benchmarks_results_dictionary' - a dictionary with results to be used for benchmarks, empty unless number_of_benchmark_circuits > 0
            
            Key: 'benchmarks_marginals_dictionary' - a dictionary with marginals to be used for benchmarks, empty unless number_of_benchmark_circuits > 0
            
            Key: 'POVMs_dictionary' - a dictionary with two qubit POVMs
            
            Key: 'correlations_data' - a dictionary with correlation coefficients

            Old mitigation data:
            the rest of the keys is used to perform "old mitigation", it should be probably removed before final release, here it is used for debugging purposes 



    
        a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """    

    #number of qubits is read from results dictionary

    number_of_qubits = len(next(iter(results_dictionary.keys())))
    
 
    

    #this will be used in characterization functions
    qubit_indices = [i for i in range(number_of_qubits)]

    #here subsets of qubits used in initial caracterization are established. By default we perform analysis for single and two qubit substes.
    single_qubits = [(i,) for i in range(number_of_qubits)]
    pairs_of_qubits = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]
    marginals_to_mitigate= single_qubits + pairs_of_qubits


    #this we use to save the results
    name_string = str(date.today()) + " q" + str(number_of_qubits)+name_id

    
   
  


    
    #computation of marginals is performed, if they are not provided
    if marginals_dictionary == None:
        
        marginals_dictionary = compute_marginals_dictionary(results_dictionary=results_dictionary,marginals_list=marginals_to_mitigate,experiment_type=experiment_type,data_directory=data_directory,name_string=name_string)
          
        qprint('MARGINALS COMPUTATION FINISHED')

        dictionary_to_save = {'marginals_dictionary':marginals_dictionary}
        file_name_marginals = experiment_type+ '_marginals_workflow_' + name_string +'.pkl'
        #io.save(dictionary_to_save=dictionary_to_save,
        #                        directory=data_directory,
        #                        custom_filename=file_name_marginals)
        
   

    characterization_data_dictionary = divide_data_into_characterization_benchmark_coherence_witness_data(results_dictionary=results_dictionary,marginals_dictionary=marginals_dictionary,ground_states_list=ground_states_list,coherence_witnesses_list=coherence_witnesses_list)

    characterization_marginals_dictionary = characterization_data_dictionary['characterization_marginals_dictionary']
    
    characterization_results_dictionary = characterization_data_dictionary['characterization_results_dictionary']

    benchmarks_results_dictionary = characterization_data_dictionary['benchmarks_results_dictionary']
    
    benchmarks_marginals_dictionary = characterization_data_dictionary['benchmarks_marginals_dictionary']  

    #noise matrices are computes, new object is created to prevent data to be overwritten 
    all_two_qubits_subsets = qrem_math.get_k_local_subsets(number_of_qubits, 2,True)
 
    characterization_POVMs_dictionary, characterization_matrices_dictionary = compute_reduced_POVMs_and_noise_matrices(results_dictionary=characterization_results_dictionary,marginals_dictionary=characterization_marginals_dictionary,subset_of_qubits=all_two_qubits_subsets,experiment_type='DDOT')
    
    qprint('REDUCED POVMS COMPUTATION FINISHED')    
   
    dictionary_to_save = {'POVMs_dictionary':characterization_POVMs_dictionary}

    file_name_POVMs  = 'DDOT_POVMs_PLS_workflow_' + name_string +'.pkl'

    #io.save(dictionary_to_save=dictionary_to_save,
    #                            directory=data_directory,
    #                            custom_filename=file_name_POVMs)
    
  

    noise_analysis_dictionary =  perform_POVMs_errors_and_correlation_coefficients_computation(results_dictionary=characterization_results_dictionary,marginals_dictionary=characterization_marginals_dictionary,POVMs_dictionary=characterization_POVMs_dictionary) 

    correlations_data = noise_analysis_dictionary['correlations_data_dictionary']

    POVMs_distances_dictionary = noise_analysis_dictionary['POVMs_error_dictionary']


   


   
    if name_id == 'IBM_Cusco':
        settings_list=['2','3','4','5']

    elif name_id == 'Rigetti_Aspen-M-3':
        settings_list=['2','3']
        
    coherence_bound_dictionary = fca.compute_coherence_indicator(marginals_dictionary=characterization_data_dictionary['coherence_witnesses_marginals_dictionary'],subset_list=[(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)],settings_list=settings_list)



 

    
      
    

    #here we compute only worst case classical distance, others can be added
   

    dictionary_to_save = {'correlations_data':correlations_data,
                      
                     }

    file_name_errors  = 'DDOT_correlations_workflow_' + name_string +'.pkl'
    #io.save(dictionary_to_save=dictionary_to_save,
    #                            directory=data_directory,
    #                            custom_filename=file_name_errors)



    qprint('CORRELATIONS COEFFICIENTS COMPUTATION FINISHED')  

    noise_model_analyzer = NoiseModelGenerator(results_dictionary=characterization_results_dictionary,
                                           marginals_dictionary=characterization_marginals_dictionary,
                                           noise_matrices_dictionary=characterization_matrices_dictionary,
                                           correlations_data=correlations_data
                                           )

    all_clusters_sets_dictionary = {tuple([(qi,) for qi in range(number_of_qubits)]):None}

    #clustering starts 

    clustering_functions_parameters = {'sizes_clusters':[2,3,4],'distance_type':'worst_case','correlations_type': 'classical','alpha_hyperparameters': [0] }
    
    all_clusters_sets_neighbors_dictionary = perform_clustering_routine(results_dictionary = characterization_results_dictionary,marginals_dictionary = characterization_marginals_dictionary, noise_matrices_dictionary = characterization_matrices_dictionary ,correlations_data = correlations_data, number_of_qubits=number_of_qubits,clustering_functions_parameters = clustering_functions_parameters)
    
    

    #########################################################################################################
    ###### Characterization routine ends here                                     ###########################  
    ###### What happens next is preparation of data used in mitigation routine    ###########################
    #########################################################################################################    


    
    all_clusters_sets_list = list(all_clusters_sets_neighbors_dictionary.keys())
    
    cluster_subsets = []

    for cluster_list in all_clusters_sets_list:
        for cluster in cluster_list:
            if cluster not in cluster_subsets:
                cluster_subsets.append(cluster)

    #marginals_analyzer.compute_all_marginals(cluster_subsets,
    #                                     show_progress_bar=True,
    #                                     multiprocessing=True)


    
    #all_clusters_sets_list = all_clusters_sets_list[1:]
    
    if perform_noise_model_reconstruction:
        #noise models data are saved in a list of CNModelData 
        
      
        noise_model_list = perform_noise_model_reconstruction_routine(results_dictionary=characterization_results_dictionary,number_of_qubits=number_of_qubits,all_clusters_sets_neighbors_dictionary=all_clusters_sets_neighbors_dictionary)

        characterization_data_dictionary['noise_models_list'] = noise_model_list
    
    characterization_data_dictionary['benchmarks_results_dictionary'] = benchmarks_results_dictionary
    characterization_data_dictionary['benchmarks_marginals_dictionary']= benchmarks_marginals_dictionary
    characterization_data_dictionary['POVMs_dictionary'] = characterization_POVMs_dictionary
    characterization_data_dictionary['correlations_data'] = correlations_data
    

    #old mitigation data can be also computed
    if return_old_mitigation_data:

        if experiment_type =='DDOT':
            marginals_analyzer_noise_model_old = DOTMarginalsAnalyzer(results_dictionary_ddot=characterization_results_dictionary)


            marginals_analyzer_noise_model_old.compute_subset_noise_matrices_averaged(subsets_list=cluster_subsets,show_progress_bar=True)
            noise_matrices_dictionary_old = marginals_analyzer_noise_model_old.noise_matrices_dictionary

        elif experiment_type =='QDOT':
            marginals_analyzer_noise_model_old  = QDTMarginalsAnalyzer(results_dictionary=characterization_results_dictionary,experiment_name='QDT')
            marginals_analyzer_noise_model_old .initialize_labels_interpreter(interpreter='PAULI')

        #calculate reduced POVMs
            marginals_analyzer_noise_model_old .compute_subsets_POVMs_averaged(subsets_of_qubits=cluster_subsets,
                                                           show_progress_bar=True,
                                                           estimation_method='PLS')
            POVMs_dictionary = marginals_analyzer_noise_model_old ._POVM_dictionary

            marginals_analyzer_noise_model_old .compute_noise_matrices_from_POVMs(subsets_of_qubits=cluster_subsets,
                                                     show_progress_bar=True)
            
                 
            new_noise_matrices_dictionary_old = marginals_analyzer_noise_model_old .noise_matrices_dictionary

            noise_matrices_dictionary_old = {}

            for key, noise_matrix in new_noise_matrices_dictionary_old .items():
                if noise_matrix != {}:
                    noise_matrices_dictionary_old[key] =noise_matrix

        
        correction_matrices, correction_indices,mitigation_data_dictionary = fdt.get_multiple_mitigation_strategies_clusters_for_pairs_of_qubits(
       
        pairs_of_qubits=pairs_of_qubits,
       
        clusters_sets=all_clusters_sets_dictionary,
       
        dictionary_results=characterization_results_dictionary,
       
        noise_matrices_dictionary=noise_matrices_dictionary_old,
        show_progress_bar=True)

        characterization_data_dictionary['correction_matrices'] = correction_matrices

        characterization_data_dictionary['correlation_coefficients_dictionary'] = correlations_data
        
        characterization_data_dictionary['POVMs_dictionary']= characterization_POVMs_dictionary
        
        characterization_data_dictionary['POVMs_distances_dictionary']=POVMs_distances_dictionary

        
        characterization_data_dictionary['correction_indices'] = correction_indices
        
        characterization_data_dictionary['mitigation_data_dictionary'] = mitigation_data_dictionary
        
        characterization_data_dictionary['all_clusters_sets_dictionary'] = all_clusters_sets_dictionary
        
        characterization_data_dictionary['noise_matrices_dictionary_old'] = noise_matrices_dictionary_old
        
        characterization_data_dictionary['all_clusters_sets_neighbors_dictionary'] = all_clusters_sets_neighbors_dictionary

        characterization_data_dictionary['coherence_bound_dictionary']  = coherence_bound_dictionary




        
       
        
    file_name_mitigation_data  = 'DDOT_characterization_data_' + name_string +'.pkl'
    #io.save(dictionary_to_save=characterization_data_dictionary,
    #                            directory=data_directory,
    #                            custom_filename=file_name_mitigation_data)

    
    return characterization_data_dictionary
    


    


