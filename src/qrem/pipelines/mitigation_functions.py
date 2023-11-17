import sys
import os
import random
import pickle 

from tqdm import tqdm
import numpy as np

import qrem.common.providers.ibmutils.data_converters
from qrem.common import io

#directory_QREM = os.environ["QREM"] +'\\src\\qrem\\'
#sys.path.append(os.path.dirname(directory_QREM))
#os.environ["QREM"] = "/home/tuzjan/Documents/QREM_DEVELOPMENT/QREM_SECRET_DEVELOPMENT/src/qrem/"

import qrem.functions_qrem.ancillary_functions  as anf
import qrem.functions_qrem.functions_data_analysis as fdt 

from qrem.noise_characterization.tomography_design.overlapping.DOTMarginalsAnalyzer import DOTMarginalsAnalyzer  
from qrem.noise_characterization.base_classes.OverlappingTomographyBase import OverlappingTomographyBase
from qrem.noise_characterization.tomography_design.overlapping.SeparableCircuitsCreator import SeparableCircuitsCreator
from qrem.noise_characterization.data_analysis.InitialNoiseAnalyzer import InitialNoiseAnalyzer
from typing import Dict, Tuple, List

from qrem.backends_support.qiskit import qiskit_utilities
from qrem.functions_qrem import functions_data_analysis as fdt

from qrem.noise_simulation.CN.noise_implementation import  simulate_noise_results_dictionary
from qrem.noise_model_generation.CN.NoiseModelGenerator import NoiseModelGenerator

import qrem.common.math as qrem_math
from qrem.common.printer import qprint, qprint_array
from qrem.functions_qrem import functions_benchmarks as fun_ben
from qrem.types import CNModelData
from qrem.cn import mitigation as cnmitigation
from qrem.cn import simulation as cnsimulation 
from datetime import date

from tqdm import tqdm


import statistics 

# Function used to divide a dictionary into two parts. Here it is used to create characterization and benchmark data 

def divide_dictionary(dictionary, items_number, start_index=0):
    counter=0
    divided_dictionary_1 ={}
    divided_dictionary_2={}
    for key,entry in  dictionary.items():
        if counter<start_index+items_number-1:
            divided_dictionary_1[key] = entry
        else:
            divided_dictionary_2[key]=entry    
        counter=counter+1
    return divided_dictionary_1, divided_dictionary_2

def compute_mitigation_errors(mitigated_energy_dictionary, hamiltonian_energy_dictionary,number_of_qubits):
    
    noise_models_mitigated_energy_dictionary_error = {}
    
    for key, benchmark_energy_dictionary in mitigated_energy_dictionary.items():
    
        mitigated_energy_dictionary_error ={}
    
        for index, energy_value in benchmark_energy_dictionary.items():
    
            if index in hamiltonian_energy_dictionary.keys():
    
                mitigated_energy_dictionary_error[index] = abs(energy_value- hamiltonian_energy_dictionary[index]['energy_ideal'])/number_of_qubits
     
        noise_models_mitigated_energy_dictionary_error[key] = mitigated_energy_dictionary_error 
    
    return noise_models_mitigated_energy_dictionary_error

"""

"""
def compute_mitigation_error_median_mean(mitigation_errors_dictionary:Dict,print_results:bool = True):
    benchmark_results_mean_median_dictionary = {'median':{}, 'mean':{}}


    for cluster_assigment, benchmark_results_dictionary in mitigation_errors_dictionary .items():
        benchmark_results_list = list(benchmark_results_dictionary.values()) 
        
    
        benchmark_results_mean_median_dictionary['median'][cluster_assigment] = statistics.median(benchmark_results_list)
        benchmark_results_mean_median_dictionary['mean'][cluster_assigment] = statistics.mean(benchmark_results_list)
    
        if print_results:    
            print(cluster_assigment)
            print('Mitigation error mean' ,benchmark_results_mean_median_dictionary['mean'][cluster_assigment] )
        
            print('Mitigation error median', benchmark_results_mean_median_dictionary['median'][cluster_assigment])

    return benchmark_results_mean_median_dictionary


def estimate_mitigated_energy_for_hamiltonian(results_dictionary:dict[str,dict[str,int]], hamiltonian_dictionary:dict[str,dict], noise_model:type[CNModelData], ensure_proper_probability_distribution:bool = True, product_mitigation:bool =False, return_marginals:bool = False):
    
    """
    Function performs error mitigation based on CN noise model for a particular Hamiltonain and measurement statistics. It is used in usecases and benchmarks. It calls routines from cn/mitigation module.

    Parameters
    ----------
    results_dictionary: dictionary (key - string, value - dictionary)
        Dictionary of experimental results to be mitigated 

     hamiltonian_dictionary: dictionary 
        Dictionary storing hamiltonian data, in this function only weights_dictionary is used 
    
    noise_model: instance of CNModelData class
        Stores information about noise model used in mitigation

    ensure_proper_probability_distribution: bool (optional, by default set to True)
        Controls whether in mitigation one uses proper probability distributions or possibly pseudo probability distributions 
    
    product_mitigation: bool (optional)
        Used for debugging purposes, probably should be removed for final release

    return marginals: bool
         Used for debugging purposes, probably should be removed for final release   
      
        

    """
    
    
    weights_list=list(hamiltonian_dictionary['weights_dictionary'].keys())
    #mitigation is performed 
    
    if not product_mitigation:
        mitigated_marginals = cnmitigation.mitigate_marginals(marginals_list=weights_list,results_dictionary=results_dictionary,noise_model=noise_model, ensure_proper_probability_distribution=ensure_proper_probability_distribution,state_independent_mitigation=False)
    
    else:
        mitigated_marginals = cnmitigation.mitigate_marginals_product(marginals_list=weights_list,results_dictionary=results_dictionary,noise_model=noise_model, ensure_proper_probability_distribution=ensure_proper_probability_distribution)


    
    energy_corrected = \
                    fdt.estimate_energy_from_marginals(weights_dictionary=hamiltonian_dictionary['weights_dictionary'],
                                                       marginals_dictionary= mitigated_marginals[next(iter(mitigated_marginals))])
    
    return {'corrected_energy':energy_corrected} if not return_marginals else {'corrected_energy':energy_corrected, 'mitigated_marginals':mitigated_marginals}

def estimate_mitigated_energy_for_hamiltonians_set(results_dictionary:dict[str,dict[str,int]], hamiltonians_dictionary:dict[str,dict] ,noise_model:type[CNModelData] ,ensure_proper_probability_distribution:bool = True,product_mitigation:bool=False, return_marginals:bool = False):

    """
    Function performs error mitigation based on CN noise model for a set Hamiltonains and set of measurement results. It is used in usecases and benchmarks. 
    Due to QREM convention mitigated measurement statistics from one input state are used to estimate energy of one hamiltonian.
    Function calls estimate_mitigated_energy_for_hamiltonian routine.

    Parameters
    ----------
    results_dictionary: dictionary (key - string, value - dictionary)
        Dictionary of experimental results to be mitigated 

     hamiltonians_dictionary: dictionary 
        Dictionary storing hamiltonians datad 
    
    noise_model: instance of CNModelData class
        Stores information about noise model used in mitigation

    ensure_proper_probability_distribution: bool (optional, by default set to True)
        Controls whether in mitigation one uses proper probability distributions or possibly pseudo probability distributions 
    
    product_mitigation: bool (optional)
        Used for debugging purposes, probably should be removed for final release

    return marginals: bool
         Used for debugging purposes, probably should be removed for final release   
      
        

    """

    mitigation_results_dictionary ={}
    temporary_mitigation_results_dictionary = {}

    mitigation_results_dictionary[ 'corrected_energy']= {hamiltonian_index : 0 for hamiltonian_index in  hamiltonians_dictionary.keys()} 
    if return_marginals:
        mitigation_results_dictionary['mitigated_marginals'] = {hamiltonian_index : 0 for hamiltonian_index in  hamiltonians_dictionary.keys()} 
  

    for state, hamiltoanian_index in tqdm(zip(results_dictionary.keys(), hamiltonians_dictionary.keys()),total=len(list(results_dictionary.keys()))):
        
       
        temporary_mitigation_results_dictionary = estimate_mitigated_energy_for_hamiltonian(results_dictionary={state:results_dictionary[state]},hamiltonian_dictionary= hamiltonians_dictionary[hamiltoanian_index],noise_model=noise_model,ensure_proper_probability_distribution=ensure_proper_probability_distribution,product_mitigation=product_mitigation, return_marginals=return_marginals ) 
        
        mitigation_results_dictionary['corrected_energy'][hamiltoanian_index]  = temporary_mitigation_results_dictionary['corrected_energy']
        
        if return_marginals:
            mitigation_results_dictionary['mitigated_marginals'][hamiltoanian_index]  = temporary_mitigation_results_dictionary['mitigated_marginals']

   
       
    return mitigation_results_dictionary 


def estimate_mitigated_energy_over_noise_models(results_dictionary,hamiltonians_dictionary,noise_models_list,ensure_proper_probability_distribution = True,product_mitigation=False, return_marginals = False ):

    noise_models_mitigation_results_dictionary = {}

    for noise_model in noise_models_list:

        noise_models_mitigation_results_dictionary[noise_model.clusters_tuple] = estimate_mitigated_energy_for_hamiltonians_set(results_dictionary=results_dictionary,hamiltonians_dictionary=hamiltonians_dictionary,noise_model=noise_model, ensure_proper_probability_distribution=ensure_proper_probability_distribution,product_mitigation=product_mitigation, return_marginals= return_marginals)
        
        
         
        #if not return_marginals:
        #    noise_models_mitigated_energy_dictionary[noise_model.clusters_tuple] = estimate_mitigated_energy_for_hamiltonians_set(results_dictionary=results_dictionary,hamiltonians_dictionary=hamiltonians_dictionary,noise_model=noise_model, ensure_proper_probability_distribution=ensure_proper_probability_distribution,product_mitigation=product_mitigation)
            
        #else:
        #  noise_models_mitigated_energy_dictionary[noise_model.clusters_tuple], noise_models_mitigated_marginals_dictionary[noise_model.clusters_tuple] = estimate_mitigated_energy_for_hamiltonians_set(results_dictionary=results_dictionary,hamiltonians_dictionary=hamiltonians_dictionary,noise_model=noise_model, ensure_proper_probability_distribution=ensure_proper_probability_distribution,product_mitigation=product_mitigation,return_marginals=True)
                      
    #structure of the results dictionary is changed 
    noise_models_mitigated_energy_dictionary ={}
    noise_models_marginals_dictionary = {}
    results_dictionary = {}
    
    for key, entry in noise_models_mitigation_results_dictionary.items():
        noise_models_mitigated_energy_dictionary[key] = entry['corrected_energy']
        if return_marginals:
            noise_models_marginals_dictionary[key] = entry['mitigated_marginals']
    
    results_dictionary['corrected_energy'] = noise_models_mitigated_energy_dictionary
    if return_marginals:
        results_dictionary['mitigated_marginals'] = noise_models_marginals_dictionary


    
    return results_dictionary


def compute_noisy_energy_over_noise_models(results_dictionary,hamiltonians_dictionary,noise_models_list):

    noise_models_prediction_results_dictionary = {}

    for noise_model in noise_models_list:

        noise_models_prediction_results_dictionary[noise_model.clusters_tuple] = compute_noisy_energy_for_hamiltonians_set(results_dictionary=results_dictionary,hamiltonians_dictionary=hamiltonians_dictionary,noise_model=noise_model)
        
        
         
   
    noise_models_predicted_energy_dictionary ={}
   
    results_dictionary = {}
    
    for key, entry in noise_models_prediction_results_dictionary.items():
        noise_models_predicted_energy_dictionary[key] = entry['predicted_energy']
    
    
    results_dictionary['predicted_energy'] = noise_models_predicted_energy_dictionary
 


    
    return results_dictionary


def compute_noisy_energy_for_hamiltonians_set(results_dictionary:dict[str,dict[str,int]], hamiltonians_dictionary:dict[str,dict] ,noise_model:type[CNModelData] ):

  

   

    prediction_results_dictionary = {}
  
    prediction_results_dictionary[ 'predicted_energy']= {hamiltonian_index : 0 for hamiltonian_index in  hamiltonians_dictionary.keys()} 
    for state, hamiltoanian_index in tqdm(zip(results_dictionary.keys(), hamiltonians_dictionary.keys()),total=len(list(results_dictionary.keys()))):
        
       
       
        prediction_results_dictionary['predicted_energy'][hamiltoanian_index]  =  compute_noise_energy_for_hamiltonian(hamiltonian_dictionary= hamiltonians_dictionary[hamiltoanian_index],noise_model=noise_model,input_state=state ) 
        
    

   
       
    return prediction_results_dictionary 



def compute_noise_energy_for_hamiltonian( hamiltonian_dictionary:dict[str,dict], noise_model:type[CNModelData],input_state):
    
  
    
    
    weights_dictionary=hamiltonian_dictionary['weights_dictionary']
    #mitigation is performed 
    
    predicted_energy = compute_noisy_energy_prediction(noise_model=noise_model,weights_dictionary=weights_dictionary,input_state=input_state)
    
    return {'predicted_energy':predicted_energy} 



def compute_noisy_energy_prediction(noise_model: type[CNModelData],weights_dictionary:Dict,input_state):
        
       

        needed_pairs = [x for x in weights_dictionary.keys() if len(x) == 2]
        energies_dictionary_now={}
        errors_dictionary_now = {}
       

        noise_matrices_dictionary_model = {cluster: noise_model.noise_matrices[cluster]['averaged']
                                                   for cluster in list(noise_model.clusters_neighborhoods.keys())}

        predicted_energy = fdt.get_noisy_energy_product_noise_model(input_state=input_state,
                                                                            noise_matrices_dictionary=noise_matrices_dictionary_model,
                                                                            needed_pairs=needed_pairs,
                                                                            weights_dictionary_tuples=weights_dictionary)

        
        
        return predicted_energy


def simulate_noisy_experiment( noise_model: type[CNModelData], number_of_circuits: int, number_of_shots: int, number_of_benchmark_circuits:int, data_directory: str ,name_id: str = '',save_data: bool = False,return_ideal_experiment_data: bool = False):

    number_of_qubits = noise_model.number_of_qubits  
    qubit_indices = [i for i in range(number_of_qubits)]

 

    name_string = str(date.today()) + " q" + str(number_of_qubits)+name_id 
    
    OT_creator = OverlappingTomographyBase(number_of_qubits=number_of_qubits, experiment_name='DDOT') 
                                       

    circuits_QDOT = OT_creator.get_random_circuits_list(number_of_circuits=number_of_circuits)
        

    

    circuits_creator = SeparableCircuitsCreator(SDK_name='qiskit', 
                                            qubit_indices=qubit_indices,
                                            descriptions_of_circuits=circuits_QDOT,
                                            experiment_name='DDOT')

    OT_circuits_list = circuits_creator.get_circuits()

    batches = anf.create_batches(circuits_list=OT_circuits_list, circuits_per_job = 300)

    jobs_list = qiskit_utilities.run_batches(backend_name='qasm_simulator',batches=batches, shots=number_of_shots)

    unprocessed_results = qrem.common.providers.ibmutils.data_converters.get_counts_from_qiskit_jobs(jobs_list=jobs_list)

    processed_results = fdt.convert_counts_overlapping_tomography(counts_dictionary=unprocessed_results, experiment_name='DDOT')


    file_name_results  = 'DDOT_simulation_workflow_results_' + name_string +'.pkl'
    io.save(dictionary_to_save=processed_results,
                                directory=data_directory,
                                custom_filename=file_name_results)




    results_dictionary =  {}
    if return_ideal_experiment_data:
        
        marginals_analyzer_ideal = DOTMarginalsAnalyzer(results_dictionary_ddot=processed_results)

        single_qubits = [(i,) for i in range(number_of_qubits)]
        pairs_of_qubits = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]

        marginals_to_mitigate=list(noise_model.clusters_tuple)  + single_qubits + pairs_of_qubits

        marginals_analyzer_ideal.compute_all_marginals(subsets_dictionary=marginals_to_mitigate,show_progress_bar=True,multiprocessing=True)
        ideal_experiments_dic, ideal_benchmarks_dic = divide_dictionary(marginals_analyzer_ideal.marginals_dictionary,len(marginals_analyzer_ideal.marginals_dictionary.keys())-number_of_benchmark_circuits)
        results_dictionary['ideal_experiments_dictionary'] = ideal_experiments_dic
        results_dictionary['ideal_benchmarks_dictionary'] = ideal_benchmarks_dic



    noisy_results_dictionary=cnsimulation.simulate_noise_results_dictionary(processed_results,noise_model)

    results_dictionary['noisy_results_dictionary'] = noisy_results_dictionary



    if save_data:
        file_name_noisy_results  = 'DDOT_noisy_simulation_results_' + name_string +'.pkl'

        io.save(dictionary_to_save=results_dictionary,
                                directory=data_directory,
                                custom_filename=file_name_noisy_results)
    return results_dictionary

    
     

    print('Simulation of noise is finished')

def create_data_to_test_mitigation(number_of_qubits, noise_model, number_of_circuits, number_of_shots,number_of_benchmark_circuits,data_directory,return_old_mitigation_data = False):
    

    number_of_benchmark_circuits = number_of_benchmark_circuits -1


    


    qubit_indices = [i for i in range(number_of_qubits)]

    single_qubits = [(i,) for i in range(number_of_qubits)]
    pairs_of_qubits = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]

    marginals_to_mitigate=list(noise_model.clusters_tuple)  + single_qubits + pairs_of_qubits

    name_string = str(date.today()) + " q" + str(number_of_qubits) 
    
    OT_creator = OverlappingTomographyBase(number_of_qubits=number_of_qubits, experiment_name='DDOT') 
                                       

    circuits_QDOT = OT_creator.get_random_circuits_list(number_of_circuits=number_of_circuits)
        

    

    circuits_creator = SeparableCircuitsCreator(SDK_name='qiskit', 
                                            qubit_indices=qubit_indices,
                                            descriptions_of_circuits=circuits_QDOT,
                                            experiment_name='DDOT')

    OT_circuits_list = circuits_creator.get_circuits()

    batches = anf.create_batches(circuits_list=OT_circuits_list, circuits_per_job = 300)

    jobs_list = qiskit_utilities.run_batches(backend_name='qasm_simulator',batches=batches, shots=number_of_shots)

    unprocessed_results = qrem.common.providers.ibmutils.data_converters.get_counts_from_qiskit_jobs(jobs_list=jobs_list)

    processed_results = fdt.convert_counts_overlapping_tomography(counts_dictionary=unprocessed_results, experiment_name='DDOT')


    file_name_results  = 'DDOT_simulation_workflow_results_' + name_string +'.pkl'
    io.save(dictionary_to_save=processed_results,
                                directory=data_directory,
                                custom_filename=file_name_results)


    marginals_analyzer_ideal = DOTMarginalsAnalyzer(results_dictionary_ddot=processed_results)

    marginals_analyzer_ideal.compute_all_marginals(subsets_dictionary=marginals_to_mitigate,show_progress_bar=True,multiprocessing=False)

    ideal_experiments_dic, ideal_benchmarks_dic = divide_dictionary(marginals_analyzer_ideal.marginals_dictionary,len(marginals_analyzer_ideal.marginals_dictionary.keys())-number_of_benchmark_circuits)



    noisy_results_dictionary=cnsimulation.simulate_noise_results_dictionary(processed_results,noise_model)


    file_name_noisy_results  = 'DDOT_noisy_simulation_results_' + name_string +'.pkl'

    io.save(dictionary_to_save=noisy_results_dictionary,
                                directory=data_directory,
                                custom_filename=file_name_noisy_results)

    


    io.save(dictionary_to_save=noisy_results_dictionary,
                                directory=data_directory,
                                custom_filename=file_name_noisy_results)

    print('Simulation of noise is finished')

    locality=2
    subset_of_qubits = qrem_math.get_k_local_subsets(number_of_qubits, 2,True)
    #subset_of_qubits = pairs_of_qubits
    #subset_of_qubits +=  [(i,) for i in range(number_of_qubits)]





    print('Marginals calculation starts')

    marginals_analyzer = DOTMarginalsAnalyzer(results_dictionary_ddot=noisy_results_dictionary)


    marginals_analyzer.compute_all_marginals(subset_of_qubits,show_progress_bar=True,multiprocessing=False)

    marginals_dictionary=marginals_analyzer.marginals_dictionary

    dictionary_to_save = {'marginals_dictionary':marginals_dictionary}

    characterization_results_dictionary, benchmarks_results_dictionary=divide_dictionary(noisy_results_dictionary,len(noisy_results_dictionary.keys())-number_of_benchmark_circuits)
    characterization_marginals_dictionary, benchmarks_marginals_dictionary = divide_dictionary(marginals_dictionary,len(noisy_results_dictionary.keys())-number_of_benchmark_circuits)


    file_name_marginals = 'DDOT_marginals_workflow_' + name_string +'.pkl'

    io.save(dictionary_to_save=dictionary_to_save,
                                directory=data_directory,
                                custom_filename=file_name_marginals)


    io.save(dictionary_to_save=dictionary_to_save,
                                directory=data_directory,
                                custom_filename=file_name_marginals)

    marginals_analyzer = DOTMarginalsAnalyzer(results_dictionary_ddot=characterization_results_dictionary,marginals_dictionary=characterization_marginals_dictionary)

    marginals_analyzer.compute_subset_noise_matrices_averaged(subset_of_qubits)


    POVMs_dictionary_diag= marginals_analyzer._noise_matrices_dictionary




    file_name_POVMs  = 'DDOT_POVMs_PLS_workflow_' + name_string +'.pkl'


    POVMs_dictionary = {}
    for key, entry in POVMs_dictionary_diag.items():
        list_of_matrices=[]
        for matrix in POVMs_dictionary_diag[key]['averaged']:
            list_of_matrices.append(np.diag(matrix))

        POVMs_dictionary[key]=list_of_matrices

    dictionary_to_save = {'POVMs_dictionary':POVMs_dictionary}


    characterization_POVMs_dictionary=POVMs_dictionary

    noise_analyzer = InitialNoiseAnalyzer(results_dictionary=characterization_results_dictionary,
                                      marginals_dictionary=characterization_marginals_dictionary,
                                      POVM_dictionary=characterization_POVMs_dictionary
                                              )

    io.save(dictionary_to_save=dictionary_to_save,
                                directory=data_directory,
                                custom_filename=file_name_POVMs)


    qubit_indices = list(range(number_of_qubits))

    distances_types_correlations = [('worst_case', 'classical')]
        
    noise_analyzer.compute_correlations_data_pairs(qubit_indices=range(number_of_qubits),
                                              distances_types=distances_types_correlations)

    correlations_data = noise_analyzer.correlations_data

    dictionary_to_save = {'correlations_data':correlations_data,
                      
                     }

    file_name_errors  = 'DDOT_correlations_workflow_' + name_string +'.pkl'
    io.save(dictionary_to_save=dictionary_to_save,
                                directory=data_directory,
                                custom_filename=file_name_errors)



    characterization_noise_matrix_dictionary = fdt.get_noise_matrices_from_POVMs_dictionary(characterization_POVMs_dictionary)


    noise_model_analyzer = NoiseModelGenerator(results_dictionary=characterization_results_dictionary,
                                           marginals_dictionary=characterization_marginals_dictionary,
                                           noise_matrices_dictionary=characterization_noise_matrix_dictionary ,
                                           correlations_data=correlations_data
                                           )



    sizes_clusters = [2,3,4]

    distance_type = 'worst_case'
    correlations_type = 'classical'
    correlations_table = correlations_data[distance_type][correlations_type]

    alpha_hyperparameters = np.linspace(0.0, 3.0, 16)
    alpha_hyperparameters = [np.round(alpha, 3) for alpha in alpha_hyperparameters]
    alpha_hyperparameters=[1]

    all_clusters_sets_dictionary = {tuple([(qi,) for qi in range(number_of_qubits)]):None}
    #all_clusters_sets_dictionary= {}

    for max_cluster_size in sizes_clusters:
        print("\nCurrent max cluster size:", max_cluster_size, 'red')

        for alpha_multiplier in alpha_hyperparameters:
            clustering_function_arguments = {'alpha_multiplier': alpha_multiplier}

            clusters_tuples_now, score = noise_model_analyzer.compute_clusters(correlations_table=correlations_table,
                                                                           maximal_size=max_cluster_size,
                                                                           method_kwargs=clustering_function_arguments)
           # neighborhoods = noise_model_analyzer.find_all_neighborhoods(maximal_size=2)
            if clusters_tuples_now in all_clusters_sets_dictionary.keys():
                all_clusters_sets_dictionary[clusters_tuples_now].append(
                    (max_cluster_size, clustering_function_arguments['alpha_multiplier'], score))#, neighborhoods))
            else:
                all_clusters_sets_dictionary[clusters_tuples_now] = [
                    (max_cluster_size, clustering_function_arguments['alpha_multiplier'], score)]#, neighborhoods)]












    all_clusters_sets_list = list(all_clusters_sets_dictionary.keys())

    cluster_subsets = []

    for cluster_list in all_clusters_sets_list:
        for cluster in cluster_list:
            if cluster not in cluster_subsets:
                cluster_subsets.append(cluster)

    marginals_analyzer.compute_all_marginals(cluster_subsets,
                                         show_progress_bar=True,
                                         multiprocessing=False)

    marginals_dictionary_clusters = marginals_analyzer.marginals_dictionary


    noise_model_list =[]
    noise_models_noise_matrices ={}
    for cluster_list in all_clusters_sets_list:
        #marginals_analyzer.noise_matrices_dictionary = {}
        marginals_analyzer_noise_model = DOTMarginalsAnalyzer(results_dictionary_ddot=   characterization_results_dictionary)
        marginals_analyzer_noise_model.compute_subset_noise_matrices_averaged(subsets_list=cluster_list,show_progress_bar=True)
        #for test purposes
        noise_models_noise_matrices[cluster_list] = marginals_analyzer.noise_matrices_dictionary 
        #marginals_analyzer.compute_subset_noise_matrices_averaged(subsets_list=cluster_list,show_progress_bar=True)
        ##addidtion starts###
        #POVMs_dictionary_diag= marginals_analyzer._noise_matrices_dictionary
        #POVMs_dictionary = {}
        #for key, entry in POVMs_dictionary_diag.items():
        #    list_of_matrices=[]
        #    for matrix in POVMs_dictionary_diag[key]['averaged']:
        #        list_of_matrices.append(np.diag(matrix))

        #    POVMs_dictionary[key]=list_of_matrices

        #noise_matrices_dictionary = fdt.get_noise_matrices_from_POVMs_dictionary(characterization_POVMs_dictionary)
        ###addition ends###
        #new_noise_matrix_dictionary = noise_matrices_dictionary
        noise_matrices_dictionary = marginals_analyzer_noise_model.noise_matrices_dictionary
        new_noise_matrix_dictionary = {}
        for key, noise_matrix in noise_matrices_dictionary.items():
            new_noise_matrix_dictionary[key] =noise_matrix['averaged']
        noise_model = CNModelData(number_of_qubits)
        noise_model.set_clusters_tuple(clusters_tuple=tuple(cluster_list))
        noise_model.set_noise_matrices_dictionary(noise_matrices_dictionary=new_noise_matrix_dictionary)
        noise_model_list.append(noise_model)






    if return_old_mitigation_data:

        marginals_analyzer_noise_model_old = DOTMarginalsAnalyzer(results_dictionary_ddot=   characterization_results_dictionary)
        marginals_analyzer_noise_model_old.compute_subset_noise_matrices_averaged(subsets_list=cluster_subsets,show_progress_bar=True)

        #marginals_analyzer.compute_subset_noise_matrices_averaged(subsets_list=chosen_clustering,show_progress_bar=True)



        #marginals_analyzer.compute_subset_noise_matrices_averaged(chosen_clustering,
        #                                 show_progress_bar=True)
        noise_matrices_dictionary_old = marginals_analyzer_noise_model_old.noise_matrices_dictionary 
        
        correction_matrices, correction_indices,mitigation_data_dictionary = fdt.get_multiple_mitigation_strategies_clusters_for_pairs_of_qubits(
        pairs_of_qubits=pairs_of_qubits,
        clusters_sets=all_clusters_sets_dictionary,
        dictionary_results=characterization_results_dictionary,
        #marginals_dictionary={},
        noise_matrices_dictionary=noise_matrices_dictionary_old,
        show_progress_bar=True)

        marginals_analyzer.noise_matrices_dictionary  = {}
        marginals_analyzer.compute_subset_noise_matrices_averaged(subsets_list= cluster_subsets,show_progress_bar=True)

        return {'new_mitigation_data':[noise_model_list,benchmarks_results_dictionary,benchmarks_marginals_dictionary],'old_mitigation_data': [correction_matrices, correction_indices,mitigation_data_dictionary,    all_clusters_sets_dictionary, noise_matrices_dictionary_old ], 'ideal_benchmarks_dictionary': ideal_benchmarks_dic}
        
    else:
        return noise_model_list,benchmarks_results_dictionary,benchmarks_marginals_dictionary
    


    


