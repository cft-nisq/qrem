"""
Collection of functions used in benchmarking CN noise models. Used to compute results of error mitigation for a set of benchmark Hamiltonians and reconstructed CN noise models.
"""

from tqdm import tqdm
from typing import Dict, Tuple, List, Type
from qrem.common.printer import qprint, qprint_array
from qrem.qtypes import CNModelData
from qrem.cn import mitigation as cnmitigation
from qrem.benchmarks import hamiltonians, benchmarks_functions 
from qrem.qtypes.mitigation_data import MitigationData
from qrem.qtypes.characterization_data import CharacterizationData

from tqdm import tqdm


import statistics 

def compute_mitigation_errors(mitigation_data:MitigationData, hamiltonian_energy_dictionary,number_of_qubits) -> Dict:

    """
        Function computes error of readout error mitigation benchmark for a set results for different CN noise models.
        Used in error mitigation benchmarks.

        Parameters
        ----------
        mitigated_energy_dictionary: dictionary (key - tuple (cluster structure), value - dictionary)
            Dictionary storing values of energies after error mitigation for different noise models.  

        hamiltonian_dictionary: dictionary 
            Dictionary storing estimation of ideal energy eigenvalue of a eigen/ground state for a Hamiltonian used in benchmarks.  
    
        number of qubits: int
            Number of qubits of a device
        
        Returns
        ----------

        noise_models_mitigated_energy_dictionary_error: Dictionary (key - Tuple (clusters structure), Value - Dictionary (key Hamiltonian index, value - energy mitigation error))
            A dictionary storing errors of error mitigation for Hamiltonians used in benchmarks and different CN noise models
            

    """

    noise_models_mitigated_energy_dictionary_error = {}
    
    for key, benchmark_energy_dictionary in mitigation_data.noise_models_mitigation_results_dictionary['corrected_energy'].items():
    
        mitigated_energy_dictionary_error ={}
    
        for index, energy_value in benchmark_energy_dictionary.items():
    
            if index in hamiltonian_energy_dictionary.keys():
    
                mitigated_energy_dictionary_error[index] = abs(energy_value- hamiltonian_energy_dictionary[index]['energy_ideal'])/number_of_qubits
     
        noise_models_mitigated_energy_dictionary_error[key] = mitigated_energy_dictionary_error

    
    return  noise_models_mitigated_energy_dictionary_error

def compute_mitigation_error_median_mean(mitigation_data:type[MitigationData],print_results:bool = False):
    
    """
        Function computes median and mean for a dictionary storing errors of error mitigation for Hamiltonians used in benchmarks and different CN noise models
        Used in analysis of error mitigation performance.

        Parameters
        ----------
        mitigation_errors_dictionary: Dictionary (key - Tuple (clusters structure), Value - Dictionary (key Hamiltonian index, value - energy mitigation error))
            A dictionary storing errors of error mitigation for Hamiltonians used in benchmarks and different CN noise models


        print_results: bool (default False)
            Optional parameter indicating if results are printed
        
        Returns
        ----------
        benchmark_results_mean_median_dictionary - Dictionary
            Dictionary storing results of the analysis: median and mean of error mitigation over a Hamiltonian set for different noise models

            

    """

    
    
    noise_models_mitigated_energy_dictionary_error_statistics = {'median':{}, 'mean':{}}


    for cluster_assigment, benchmark_results_dictionary in mitigation_data.noise_models_mitigated_energy_dictionary_error.items():
        benchmark_results_list = list(benchmark_results_dictionary.values()) 
        
    
        noise_models_mitigated_energy_dictionary_error_statistics['median'][cluster_assigment] = statistics.median(benchmark_results_list)
        noise_models_mitigated_energy_dictionary_error_statistics['mean'][cluster_assigment] = statistics.mean(benchmark_results_list)
    
        if print_results:    
            qprint(cluster_assigment)
            qprint('Mitigation error mean' ,noise_models_mitigated_energy_dictionary_error_statistics['mean'][cluster_assigment] )
        
            qprint('Mitigation error median', noise_models_mitigated_energy_dictionary_error_statistics['median'][cluster_assigment])

    return noise_models_mitigated_energy_dictionary_error_statistics


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
        mitigated_marginals = cnmitigation.mitigate_marginals(marginals_list=weights_list,results_dictionary=results_dictionary,noise_model=noise_model, ensure_proper_probability_distribution=ensure_proper_probability_distribution,state_independent_mitigation=True)
    
    else:
        mitigated_marginals = cnmitigation.mitigate_marginals_product(marginals_list=weights_list,results_dictionary=results_dictionary,noise_model=noise_model, ensure_proper_probability_distribution=ensure_proper_probability_distribution)


    
    energy_corrected = \
                    hamiltonians.estimate_energy_from_marginals(weights_dictionary=hamiltonian_dictionary['weights_dictionary'],
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


def estimate_mitigated_energy_over_noise_models(characterization_data:type[CharacterizationData],hamiltonians_dictionary,ensure_proper_probability_distribution = True, return_marginals = False ):

    mitigation_data =MitigationData()

    noise_models_mitigation_results_dictionary = {}

    for noise_model in characterization_data.noise_model_list:

        noise_models_mitigation_results_dictionary[noise_model.clusters_tuple] = estimate_mitigated_energy_for_hamiltonians_set(results_dictionary=characterization_data.benchmark_results_dictionary,hamiltonians_dictionary=hamiltonians_dictionary,noise_model=noise_model, ensure_proper_probability_distribution=ensure_proper_probability_distribution, return_marginals= return_marginals)
        
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


    
    return mitigation_data

def compute_noisy_energy_over_noise_models(characterization_data:type[CharacterizationData],hamiltonians_dictionary,):

    noise_models_prediction_results_dictionary = {}

    for noise_model in characterization_data.noise_model_list:

        noise_models_prediction_results_dictionary[noise_model.clusters_tuple] = compute_noisy_energy_for_hamiltonians_set(results_dictionary=characterization_data.benchmark_results_dictionary,hamiltonians_dictionary=hamiltonians_dictionary,noise_model=noise_model)
        
        
         
   
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
        
       
       
        prediction_results_dictionary['predicted_energy'][hamiltoanian_index]  =  compute_noisy_energy_for_hamiltonian(hamiltonian_dictionary= hamiltonians_dictionary[hamiltoanian_index],noise_model=noise_model,input_state=state ) 
        
    

   
       
    return prediction_results_dictionary 



def compute_noisy_energy_for_hamiltonian( hamiltonian_dictionary:dict[str,dict], noise_model:type[CNModelData],input_state):
    
  
    
    
    weights_dictionary=hamiltonian_dictionary['weights_dictionary']
    #mitigation is performed 
    
    predicted_energy = compute_noisy_energy_prediction(noise_model=noise_model,weights_dictionary=weights_dictionary,input_state=input_state)
    
    return {'predicted_energy':predicted_energy} 



def compute_noisy_energy_prediction(noise_model: type[CNModelData],weights_dictionary:Dict,input_state):
        
       

        needed_pairs = [x for x in weights_dictionary.keys() if len(x) == 2]
     

        noise_matrices_dictionary_model = {cluster: noise_model.noise_matrices[cluster]['averaged']
                                                   for cluster in list(noise_model.clusters_neighborhoods.keys())}

        predicted_energy = benchmarks_functions.get_noisy_energy_product_noise_model(input_state=input_state,
                                                                            noise_matrices_dictionary=noise_matrices_dictionary_model,
                                                                            needed_pairs=needed_pairs,
                                                                            weights_dictionary_tuples=weights_dictionary)

        
        
        return predicted_energy