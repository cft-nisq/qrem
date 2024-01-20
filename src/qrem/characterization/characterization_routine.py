

import numpy as np

from qrem.common import povmtools

from qrem.common import convert 


from typing import Dict, Tuple, List
from qrem.qtypes.characterization_data import CharacterizationData 
from qrem.characterization import characterization
from  qrem.common import probability, math as qrem_math
from qrem.common.printer import qprint, qprint_array
from qrem.qtypes import CNModelData
from datetime import date
from typing import Tuple, Dict, List, Optional, Type 
from qrem.common.printer import qprint






    


def execute_characterization_workflow(characterization_data_container:Type[CharacterizationData],  name_id ='', perform_noise_model_reconstruction = True, find_neighbors = False):

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

    number_of_qubits = len(next(iter(characterization_data_container.results_dictionary.keys())))

   
    
 
    

    #this will be used in characterization functions
    qubit_indices = [i for i in range(number_of_qubits)]

    #here subsets of qubits used in initial caracterization are established. By default we perform analysis for single and two qubit substes.
    single_qubits = [(i,) for i in range(number_of_qubits)]
    pairs_of_qubits = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]
    marginals_to_mitigate= single_qubits + pairs_of_qubits


    #this we use to save the results
    name_string = str(date.today()) + " q" + str(number_of_qubits)+name_id
   
   
    
    #computation of marginals is performed, if they are not provided
    if characterization_data_container.marginals_dictionary == None:
        
        characterization_data_container.marginals_dictionary = probability.compute_marginals_single(results_dictionary=characterization_data_container.results_dictionary,subsets_list=marginals_to_mitigate,normalization=True)
          
    
        
        qprint('MARGINALS COMPUTATION FINISHED')

    
    characterization_data_container = convert.divide_data_into_characterization_benchmark_coherence_witness_data(characterization_data_container)
    
 

    all_two_qubits_subsets = qrem_math.get_k_local_subsets(number_of_qubits, 2,True)


        
    noise_matrices_dictionary, POVMs_dictionary = characterization.compute_reduced_POVMs_and_noise_matrices(characterization_data=characterization_data_container,subset_of_qubits=all_two_qubits_subsets)
    
    characterization_data_container.POVMs_dictionary = POVMs_dictionary

    characterization_data_container.noise_matrices_dictionary = noise_matrices_dictionary
      
    
    qprint('REDUCED POVMS COMPUTATION FINISHED')    
   
   
  

    

    
    distances_types = [('worst_case','classical')] 
    
    characterization_data_container.POMVs_errors_dictionary = characterization.compute_errors_POVMs(POVMs_dictionary=characterization_data_container.POVMs_dictionary,qubits_subsets=single_qubits,distances_types=distances_types)

    qprint('POVMS DISTANCES COMPUTATION FINISHED') 

    characterization_data_container.correlation_coefficients_dictionary  = characterization.compute_correlations_data_pairs(qubit_indices=range(number_of_qubits),POVMs_dictionary=characterization_data_container.POVMs_dictionary,distances_types=distances_types)

    qprint('PAIRWISE READOUT ERRORS CORRELATION COEFFICIENTS FINISHED')
    



    



   


   
    if name_id == 'IBM_Cusco':
        settings_list=['2','3','4','5']

    elif name_id == 'Rigetti_Aspen-M-3':
        settings_list=['2','3']
    
    if characterization_data_container.coherence_witnesses_list != None:

        
        characterization_data_container.coherence_bound_dictionary = characterization.compute_coherence_indicator(marginals_dictionary=characterization_data_container.coherence_witness_marginals_dictionary,subset_list=[(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)],settings_list=settings_list)

        qprint('COHERENCE BOUND cOMPUTATION FINISHED')  

 

    


    
  


    #clustering starts 

    clustering_functions_parameters = {'sizes_clusters':[2,3,4],'distance_type':'worst_case','correlations_type': 'classical','alpha_hyperparameters': [0] }
    
    
    
    characterization_data_container.clusters_neighbors_sets_dictionary = characterization.perform_clustering_routine(characterization_data = characterization_data_container, number_of_qubits=number_of_qubits,clustering_functions_parameters = clustering_functions_parameters,perform_neighbors_search=find_neighbors)
    
    

    #########################################################################################################
    ###### Characterization routine ends here                                     ###########################  
    ###### What happens next is preparation of data used in mitigation routine    ###########################
    #########################################################################################################    


    

    
    if perform_noise_model_reconstruction:
        #noise models data are saved in a list of CNModelData 
        
      
        
        characterization_data_container.noise_model_list = characterization.perform_noise_model_reconstruction_routine(characterization_data_container=characterization_data_container,find_neighbors=find_neighbors)
       
    
    return characterization_data_container
    


    


