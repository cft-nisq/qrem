import operator
from functools import reduce
from qrem.types import CNNoiseModel
from typing import Dict, Tuple, List
import numpy as np
from qrem.common import probability, math
from qrem.functions_qrem import povmtools 
from qrem.functions_qrem import functions_data_analysis as fda





def mitigate_marginal(marginal: Tuple, results_dictionary: Dict[str, Dict[str, int]], noise_model:type[CNNoiseModel],ensure_proper_probability_distribution:bool = False ) ->np.array:
        
    """
    Function performs mitigation of a single marginal

    Parameters
    ----------
    marginal : tuple
        A tuple of qubits indices specifying the marginal e.g. (0,1)   
    result_dictionary:
        A nested dictionary with experimental results to be mitigated. Key corresponds to bitstring encoding input state, values to a dictionary with results bitstrings and counts  
        
    noise_model : object of CNNoiseModel class
        an object of CNNoiseModel class

    ensure_proper_probability_distribution: bool
        a boolean specifying whether mitigated pseudo probability distribution should be projected onto proper probability distribution, by default set to False 


    Returns
    -------
    mitigated_marginal_probability_distribution
        a numpy array with marginal (pseudo) marginal probability distribution 


    """

    #JT: a list of clusters that are involved in marginal is created
    clusters_in_marginal_list = get_clusters_in_marginal_list(marginal=marginal, noise_model=noise_model)
    
    #creates a list of qubit indices involved in a marginal
    unordered_qubits_in_marginal_list = list(reduce(operator.concat, clusters_in_marginal_list))
    
    #creates an inverse noise matrix for the marginal 
    clusters_inverse_noise_matrix = get_marginal_inverse_noise_matrix(noise_model=noise_model, clusters_in_marginal_list=clusters_in_marginal_list)

    #qubits list is sorted

    unordered_qubits_in_marginal_list.sort()

    #computes extended marginal
    clusters_marginal_counts=probability.compute_marginals(results_dictionary=results_dictionary,subsets_list=[tuple(unordered_qubits_in_marginal_list)])

    #we are interested in the first key only hence the construction nex(iter()) TODO: this does not work, start from here 
    mitigated_clusters_marginal_probability_distribution = fda.convert_subset_counts_dictionary_to_probability_distribution(clusters_marginal_counts)
    
    #performs mitigation on an extended marginal  
    mitigated_clusters_marginal_probability_distribution = clusters_inverse_noise_matrix.dot(mitigated_clusters_marginal_probability_distribution)

    if ensure_proper_probability_distribution:
        if not probability.is_valid_probability_vector(mitigated_clusters_marginal_probability_distribution):
           mitigated_clusters_marginal_probability_distribution = povmtools.find_closest_prob_vector_l2(mitigated_clusters_marginal_probability_distribution)

        

    #returns marginalization of an  extended marginal

    mitigated_marginal_probability_distribution = probability.compute_marginal_of_probability_distribution(mitigated_clusters_marginal_probability_distribution,[unordered_qubits_in_marginal_list.index(qubit) for qubit in marginal])

    return mitigated_marginal_probability_distribution.flatten()

def mitigate_marginals(marginals_list: List[Tuple], results_dictionary: Dict[str, Dict[str, int]], noise_model: type[CNNoiseModel] ,ensure_proper_probability_distribution = False )->Dict[str, Dict[tuple[int], np.array]]:
    """Function performs mitigation for marginals list

    Parameters
    ----------
    marginals_list : List[tuple]
        A list of tuples encoding marginals e.g. [(0,1),(4,6)]   

    result_dictionary:
        A nested dictionary with experimental results to be mitigated. Key corresponds to bitstring encoding input state, values to a dictionary with results bitstrings and counts  
        
    noise_model : object of CNNoiseModel class
        an object of CNNoiseModel class

    ensure_proper_probability_distribution: bool
        a boolean specifying whether mitigated pseudo probability distribution should be projected onto proper probability distribution, by default set to False 


    Returns
    -------

        a nested dictionary {input bitstring : {(subset_tuple): marginal_probability_distributution}}, where input bitstring encodes input setting, subset_tuple encodes qubits in the marginal,
        and marginal_probability_distributution is a (pseudo) probability distribution 
    """
    #JT this dictionary 
    mitigated_marginals_dictionary ={}
   
    for marginal in marginals_list:
        mitigated_marginals_dictionary[marginal] = mitigate_marginal(marginal=marginal,results_dictionary=results_dictionary,noise_model=noise_model,ensure_proper_probability_distribution=ensure_proper_probability_distribution)
   
    return {next(iter(results_dictionary)): mitigated_marginals_dictionary}
 
def get_marginal_inverse_noise_matrix(noise_model : type[CNNoiseModel], clusters_in_marginal_list:List[Tuple])-> np.array:
    """Function computes an inverse noise matrix for specified clusters  

    Parameters
    ----------
    noise_model : object of CNNoiseModel class
        An object of CNNoiseModel class 

    clusters_in_marginal_list:
        A list of tuples with clusters involved in the marginal   
        

    Returns
    -------

        An inverse noise matrix for qubits specified in clusters_in_marginal_list, qubits are sorted in ascending order 
        (e.g. for a clusters_in_marginal_list =[(0,4),(1,8)], indices of the inverse noise matrix indices correspond to qubits in the order (0,1,4,8) )
    
    """   
    
    unordered_qubits_in_marginal_list = list(reduce(operator.concat, clusters_in_marginal_list))

    
    marginal_inverse_noise_matrix = np.array([1])
    
    for cluster in clusters_in_marginal_list:
        marginal_inverse_noise_matrix = np.kron(marginal_inverse_noise_matrix, noise_model.inverse_noise_matrices[cluster])
  
    return math.permute_composite_matrix(qubits_list=unordered_qubits_in_marginal_list,noise_matrix=marginal_inverse_noise_matrix)

def get_clusters_in_marginal_list(marginal : Tuple[int], noise_model : type[CNNoiseModel]) -> List[Tuple]:
    """Function creates a list of clusters that are involved in a marginal

    Parameters
    ----------
    marginal : tuple
        A tuple specifying marginal 

    noise_model : object of CNNoiseModel class
        An object of CNNoiseModel class  
        

    Returns
    -------

    clusters_in_marginal_list
        A list of tuples involved in the input marginal 
        
    
    """  
    clusters_in_marginal_list=[]
    
    
    for qubit in marginal:
        cluster = noise_model.qubit_in_cluster[(qubit,)]
        if cluster not in clusters_in_marginal_list:
            clusters_in_marginal_list.append(noise_model.qubit_in_cluster[(qubit,)])
    return clusters_in_marginal_list



##############################################
######For tests purposes######################
#############################################

def mitigate_marginal_product(marginal: Tuple, results_dictionary: Dict[str, Dict[str, int]], noise_model:type[CNNoiseModel],ensure_proper_probability_distribution:bool = False ) ->np.array:
    
    #JT: a list of clusters that are involved in marginal is created
    clusters_in_marginal_list = get_clusters_in_marginal_list(marginal=marginal, noise_model=noise_model)

    final_marginal = np.array([1])

    for cluster in clusters_in_marginal_list:

        clusters_inverse_noise_matrix = get_marginal_inverse_noise_matrix(noise_model=noise_model, clusters_in_marginal_list=[cluster])

        clusters_marginal_counts=probability.compute_marginals(results_dictionary=results_dictionary,subsets_list=[cluster])
        
        mitigated_clusters_marginal_probability_distribution = fda.convert_subset_counts_dictionary_to_probability_distribution(clusters_marginal_counts)

        mitigated_clusters_marginal_probability_distribution = clusters_inverse_noise_matrix.dot(mitigated_clusters_marginal_probability_distribution)

        final_marginal = np.kron(final_marginal,mitigated_clusters_marginal_probability_distribution)
    



    
    #creates a list of qubit indices involved in a marginal
    unordered_qubits_in_marginal_list = list(reduce(operator.concat, clusters_in_marginal_list))
    
    #creates an inverse noise matrix for the marginal 
    mitigated_clusters_marginal_probability_distribution = math.permute_composite_vector(unordered_qubits_in_marginal_list, final_marginal)

    #qubits list is sorted

    unordered_qubits_in_marginal_list.sort()

    if ensure_proper_probability_distribution:
        if not probability.is_valid_probability_vector(mitigated_clusters_marginal_probability_distribution):
           mitigated_clusters_marginal_probability_distribution = povmtools.find_closest_prob_vector_l2(mitigated_clusters_marginal_probability_distribution)

    mitigated_marginal_probability_distribution = probability.compute_marginal_of_probability_distribution(mitigated_clusters_marginal_probability_distribution ,[unordered_qubits_in_marginal_list.index(qubit) for qubit in marginal])


 

        

    #returns marginalization of an  extended marginal


    return mitigated_marginal_probability_distribution.flatten()

def mitigate_marginals_product(marginals_list: List[Tuple], results_dictionary: Dict[str, Dict[str, int]], noise_model: type[CNNoiseModel] ,ensure_proper_probability_distribution = False )->Dict[str, Dict[tuple[int], np.array]]:
    
    #JT this dictionary 
    mitigated_marginals_dictionary ={}
   
    for marginal in marginals_list:
        mitigated_marginals_dictionary[marginal] = mitigate_marginal_product(marginal=marginal,results_dictionary=results_dictionary,noise_model=noise_model,ensure_proper_probability_distribution=ensure_proper_probability_distribution)
   
    return {next(iter(results_dictionary)): mitigated_marginals_dictionary}
