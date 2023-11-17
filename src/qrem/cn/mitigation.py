import operator
from functools import reduce
from qrem.types import CNModelData
from typing import Dict, Tuple, List
import numpy as np
from qrem.common import probability
from qrem.common import math
from qrem.functions_qrem import povmtools 
from qrem.functions_qrem import functions_data_analysis as fda





#TODO mitigate_single_marginal

def mitigate_marginal(marginal: Tuple, results_dictionary: Dict[str, Dict[str, int]], noise_model:type[CNModelData],threshold:float,ensure_proper_probability_distribution:bool = False, check_inverse_noise_matrix_norm:bool = True ) ->np.array:
    
    """
    A basic function performing CN noise model based mitigation of a single marginal probability distribution
    by acting on it with an appropriate inverse noise matrix. Used in mitigation routines.

    Parameters
    ----------
    marginal : tuple
        A tuple of qubits indices specifying the marginal e.g. (0,1)   
    
    result_dictionary:
        A nested dictionary with experimental results to be mitigated. Key corresponds to bitstring encoding input state, values to a dictionary with results bitstrings and counts  
        
    noise_model : object of CNModelData class
        an object of CNModelData class storing CN noise model data

    ensure_proper_probability_distribution: bool
        a boolean specifying whether mitigated pseudo probability distribution should be projected onto proper probability distribution, by default set to False 
    
    check_inverse_noise_matrix_norm: bool
        a boolean specifying whether norm of inverse noise matrices should be checked during mitigation (the norm is connected with bound on error mitigation), by default set to True

    threshold: float
        sets threshold for acceptance of inverse noise matrix norm (see above), by default set to 3   


    Returns
    -------
    mitigated_marginal_probability_distribution
        a numpy array with marginal (pseudo) marginal probability distribution 


"""
    
    
    #JT: a list of clusters that are involved in marginal is created
    clusters_in_marginal_list = noise_model.get_clusters_in_marginal_list(marginal=marginal)
    
    #old
    #get_clusters_in_marginal_list(marginal=marginal, noise_model=noise_model)
    
    #creates a list of qubit indices involved in a marginal
    unordered_qubits_in_marginal_list = list(reduce(operator.concat, clusters_in_marginal_list))

    #qubits list is sorted

    unordered_qubits_in_marginal_list.sort()
    
    #creates an inverse noise matrix for the marginal 
    if tuple(unordered_qubits_in_marginal_list) in noise_model.inverse_noise_matrices.keys():
        clusters_inverse_noise_matrix = noise_model.inverse_noise_matrices[tuple(unordered_qubits_in_marginal_list)]
    else:
        clusters_inverse_noise_matrix = noise_model.compute_extended_inverse_noise_matrices(clusters_in_marginal_list=clusters_in_marginal_list)



    #old
    #clusters_inverse_noise_matrix = get_marginal_inverse_noise_matrix(noise_model=noise_model, clusters_in_marginal_list=clusters_in_marginal_list)

   

    #computes extended marginal probability distribution on which is 
    clusters_marginal_counts=probability.compute_marginals(results_dictionary=results_dictionary,subsets_list=[tuple(unordered_qubits_in_marginal_list)])

    #we are interested in the first key only hence the construction nex(iter()) TODO: this does not work, start from here 
    mitigated_clusters_marginal_probability_distribution = fda.convert_subset_counts_dictionary_to_probability_distribution(clusters_marginal_counts)
    
    #reshape probability vector
    mitigated_clusters_marginal_probability_distribution = np.array(mitigated_clusters_marginal_probability_distribution).reshape(mitigated_clusters_marginal_probability_distribution.shape[0], 1)

    #performs mitigation on an extended marginal 

    #if the norm does not exceed the threshold, the mitigation is performed 
    if check_inverse_noise_matrix_norm:
        if np.linalg.norm(clusters_inverse_noise_matrix, ord=1) < threshold:

            mitigated_clusters_marginal_probability_distribution = clusters_inverse_noise_matrix.dot(mitigated_clusters_marginal_probability_distribution)

    #if there is no verification of the norm condition, mitigation is performed 
    else:

        mitigated_clusters_marginal_probability_distribution = clusters_inverse_noise_matrix.dot(mitigated_clusters_marginal_probability_distribution)

     


   
        

    #returns marginalization of an  extended marginal

    mitigated_marginal_probability_distribution = probability.compute_marginal_of_probability_distribution(mitigated_clusters_marginal_probability_distribution,[unordered_qubits_in_marginal_list.index(qubit) for qubit in marginal])

    if ensure_proper_probability_distribution:
        if not probability.is_valid_probability_vector(mitigated_marginal_probability_distribution):
           mitigated_marginal_probability_distribution = povmtools.find_closest_prob_vector_l2(mitigated_marginal_probability_distribution).flatten()

    
    return mitigated_marginal_probability_distribution


def mitigate_marginal_state_dependent(marginal: Tuple, results_dictionary: Dict[str, Dict[str, int]], noise_model:type[CNModelData],threshold:float,ensure_proper_probability_distribution:bool = False, check_inverse_noise_matrix_norm:bool = True ) ->np.array:
    
    """
    A basic function performing CN noise model based mitigation of a single marginal probability distribution
    by acting on it with an appropriate inverse noise matrix. Used in mitigation routines.

    Parameters
    ----------
    marginal : tuple
        A tuple of qubits indices specifying the marginal e.g. (0,1)   
    
    result_dictionary:
        A nested dictionary with experimental results to be mitigated. Key corresponds to bitstring encoding input state, values to a dictionary with results bitstrings and counts  
        
    noise_model : object of CNModelData class
        an object of CNModelData class storing CN noise model data

    ensure_proper_probability_distribution: bool
        a boolean specifying whether mitigated pseudo probability distribution should be projected onto proper probability distribution, by default set to False 
    
    check_inverse_noise_matrix_norm: bool
        a boolean specifying whether norm of inverse noise matrices should be checked during mitigation (the norm is connected with bound on error mitigation), by default set to True

    threshold: float
        sets threshold for acceptance of inverse noise matrix norm (see above), by default set to 3   


    Returns
    -------
    mitigated_marginal_probability_distribution
        a numpy array with marginal (pseudo) marginal probability distribution 


"""
    
    
    #JT: a list of clusters that are involved in marginal is created
    clusters_and_neighbors_in_marginal_dictionary = noise_model.get_clusters_and_neighborhoods_in_marginal_dictionary(marginal=marginal)
    
    #old
    #get_clusters_in_marginal_list(marginal=marginal, noise_model=noise_model)
    
    #creates a list of qubit indices involved in a marginal
    unordered_qubits_in_marginal_list = list(reduce(operator.concat, list(clusters_and_neighbors_in_marginal_dictionary.keys())))

    
    #neighbors_list = reduce(operator.concat(list(clusters_and_neighbors_in_marginal_dictionary.values())))

    ###
    #get indices of noise matrices needed for mitigation  
    ###
    

    ###
    #compute or retrieve noise matrix form memory
    ###    

    #qubits list is sorted

    #we establish which noise matrices should be accounted for in mitigation 
    #noise_matrices_indicies = noise_model.get_noise_matrices_indexes(marginal=marginal,state=results_dictionary.keys())

    unordered_qubits_in_marginal_list.sort()
    
    #creates an inverse noise matrix for the marginal 
    #if tuple(unordered_qubits_in_marginal_list) in noise_model.inverse_noise_matrices.keys():
    #    clusters_inverse_noise_matrix = noise_model.inverse_noise_matrices[tuple(unordered_qubits_in_marginal_list)]
    #else:
    clusters_inverse_noise_matrix = noise_model.compute_extended_inverse_noise_matrices_state_dependent(clusters_in_marginal_list=list(clusters_and_neighbors_in_marginal_dictionary.keys()),marginal=marginal,state = next(iter(results_dictionary.keys())))



    #old
    #clusters_inverse_noise_matrix = get_marginal_inverse_noise_matrix(noise_model=noise_model, clusters_in_marginal_list=clusters_in_marginal_list)

   

    #computes extended marginal probability distribution on which is 
    clusters_marginal_counts=probability.compute_marginals(results_dictionary=results_dictionary,subsets_list=[tuple(unordered_qubits_in_marginal_list)])

    #we are interested in the first key only hence the construction nex(iter()) TODO: this does not work, start from here 
    mitigated_clusters_marginal_probability_distribution = fda.convert_subset_counts_dictionary_to_probability_distribution(clusters_marginal_counts)
    
    #reshape probability vector
    mitigated_clusters_marginal_probability_distribution = np.array(mitigated_clusters_marginal_probability_distribution).reshape(mitigated_clusters_marginal_probability_distribution.shape[0], 1)

    #performs mitigation on an extended marginal 

    #if the norm does not exceed the threshold, the mitigation is performed 
    if check_inverse_noise_matrix_norm:
        if np.linalg.norm(clusters_inverse_noise_matrix, ord=1) < threshold:

            mitigated_clusters_marginal_probability_distribution = clusters_inverse_noise_matrix.dot(mitigated_clusters_marginal_probability_distribution)

    #if there is no verification of the norm condition, mitigation is performed 
    else:

        mitigated_clusters_marginal_probability_distribution = clusters_inverse_noise_matrix.dot(mitigated_clusters_marginal_probability_distribution)

     


   
        

    #returns marginalization of an  extended marginal

    mitigated_marginal_probability_distribution = probability.compute_marginal_of_probability_distribution(mitigated_clusters_marginal_probability_distribution,[unordered_qubits_in_marginal_list.index(qubit) for qubit in marginal])

    if ensure_proper_probability_distribution:
        if not probability.is_valid_probability_vector(mitigated_marginal_probability_distribution):
           mitigated_marginal_probability_distribution = povmtools.find_closest_prob_vector_l2(mitigated_marginal_probability_distribution).flatten()

    
    return mitigated_marginal_probability_distribution



#correct this to allow for a results dictionary that has more than one setting

def mitigate_marginals(marginals_list: List[Tuple], results_dictionary: Dict[str, Dict[str, int]], noise_model: type[CNModelData] ,ensure_proper_probability_distribution = True, check_inverse_noise_matrix_norm:bool = False, threshold:float = 100, state_independent_mitigation:bool = False )->Dict[str, Dict[tuple[int], np.array]]:

    """
    A  function performing CN noise model based mitigation for marginal probability distributions specified by a marginals list.
    Used in mitigation routines. Mitigation is performed by multiplying marginal probability distributions by inverse noise matrices.

    Parameters
    ----------
    marginals_list : List[tuple]
        A list of tuples encoding marginals e.g. [(0,1),(4,6)]   

    result_dictionary:
        A nested dictionary with experimental results to be mitigated. Key corresponds to bitstring encoding input state, values to a dictionary with results bitstrings and counts  
        
    noise_model : object of CNModelData class
        an object of CNModelData class

    ensure_proper_probability_distribution: bool
        a boolean specifying whether mitigated pseudo probability distribution should be projected onto proper probability distribution, by default set to False 

    check_inverse_noise_matrix_norm: bool
        a boolean specifying whether norm of inverse noise matrices should be checked during mitigation (the norm is connected with bound on error mitigation), by default set to True

    threshold: float
        sets threshold for acceptance of inverse noise matrix norm (see above), by default set to 3   

    Returns
    -------

        a nested dictionary {input bitstring : {(subset_tuple): marginal_probability_distributution}}, where input bitstring encodes input setting, subset_tuple encodes qubits in the marginal,
        and marginal_probability_distributution is a (pseudo) probability distribution 
    """

    #This dictionary holds mitigated marginals   
    
   
    #A loop over experimental results dictionary and marginals specified by marginals_list, for each marginal mitigation is performed by mitigate_marginal   
    
    total_mitigated_marginals_dictionary = {}

    for result_setting in results_dictionary.keys():
        mitigated_marginals_dictionary ={}
        for marginal in marginals_list:
            
            if state_independent_mitigation:
                mitigated_marginals_dictionary[marginal] = mitigate_marginal(marginal=marginal,results_dictionary={result_setting: results_dictionary[result_setting]},noise_model=noise_model,ensure_proper_probability_distribution=ensure_proper_probability_distribution, check_inverse_noise_matrix_norm=check_inverse_noise_matrix_norm,threshold=threshold)
            
            else:
                mitigated_marginals_dictionary[marginal] = mitigate_marginal_state_dependent(marginal=marginal,results_dictionary={result_setting: results_dictionary[result_setting]},noise_model=noise_model,ensure_proper_probability_distribution=ensure_proper_probability_distribution, check_inverse_noise_matrix_norm=check_inverse_noise_matrix_norm,threshold=threshold)


        total_mitigated_marginals_dictionary[result_setting] = mitigated_marginals_dictionary
    # results are returned as a nested dictionary with key corresponding to the experimental state, and value
    # corresponds to mitigated marginals encoded in a dictionary 
    return total_mitigated_marginals_dictionary





  
    
#MOVE TO CNModelData class   

def get_marginal_inverse_noise_matrix(noise_model : type[CNModelData], clusters_in_marginal_list:List[Tuple])-> np.array:

    """
    Function determines inverse noise matrix for a given marginal. It is used as a step in mitigation_marginal function.
    The matrix is constructed as a tensor product on inverse noise matrices involved in a marginal, when it is necessary
    the matrix is permuted to ensure proper ordering of qubits indices. E.g. for a cluster_in_marginal_list 
    [(0,2),(1,3)], the returned matrix corresponds to qubits ordered as (0,1,2,3).   


    Parameters
    ----------
    noise_model : object of CNModelData class
        An object of CNModelData class 

    clusters_in_marginal_list:
        A list of tuples with clusters involved in the marginal   
        

    Returns
    -------

        An inverse noise matrix for qubits specified in clusters_in_marginal_list, qubits are sorted in ascending order 
        (e.g. for a clusters_in_marginal_list =[(0,4),(1,8)], indices of the inverse noise matrix indices correspond to qubits in the order (0,1,4,8) )
    
    """ 

    # a list of qubits is established
    unordered_qubits_in_marginal_list = list(reduce(operator.concat, clusters_in_marginal_list))

    #the inverse noise matrix is initialized as a unit numpy array 
    marginal_inverse_noise_matrix = np.array([1])
    
    # a loop over clusters in clusters_in_marginal_list
    for cluster in clusters_in_marginal_list:
    
        # total noise matrix is updated by taking tensor product with a inverse noise matrix of the currect cluster
        marginal_inverse_noise_matrix = np.kron(marginal_inverse_noise_matrix, noise_model.inverse_noise_matrices[cluster])
  
    #final noise matrix is returned, permuted to preserve ascending order of qubit indices if necessary 
    return math.permute_composite_matrix(qubits_list=unordered_qubits_in_marginal_list,noise_matrix=marginal_inverse_noise_matrix)




#MOVE TO CNModelData class   

def get_clusters_in_marginal_list(marginal : Tuple[int], noise_model : type[CNModelData]) -> List[Tuple]:
    
    """
    Function creates a list of clusters that are involved in a marginal. Used in mitigation routines.
    For a given marginal inspects provided noise model and checks clusters membership of qubits form
    marginal.
    
    Parameters
    ----------
    marginal : tuple
        A tuple specifying marginal 

    noise_model : object of CNModelData class
        An object of CNModelData class  
        

    Returns
    -------

    clusters_in_marginal_list
        A list of tuples involved in the input marginal 
        
    
    """  

    #list storing clusters involved in the marginal, empty for start
    clusters_in_marginal_list=[]
    
    #a loop over qubits in the marginal
    for qubit in marginal:
        
        #a cluster to which the qubit belongs is determined 
        cluster = noise_model.qubit_in_cluster_membership[(qubit,)]
        
        #this cluster is appended to clusters_in_marginal_list if it is not there 
        if cluster not in clusters_in_marginal_list:
            clusters_in_marginal_list.append(noise_model.qubit_in_cluster_membership[(qubit,)])
    
    #results are returned
    return clusters_in_marginal_list



##############################################
######For tests purposes######################
#############################################

def mitigate_marginal_product(marginal: Tuple, results_dictionary: Dict[str, Dict[str, int]], noise_model:type[CNModelData],ensure_proper_probability_distribution:bool = False ) ->np.array:
    
    #JT: a list of clusters that are involved in marginal is created
    clusters_in_marginal_list = get_clusters_in_marginal_list(marginal=marginal, noise_model=noise_model)

    final_marginal = np.array([1])

    for cluster in clusters_in_marginal_list:

        clusters_inverse_noise_matrix = get_marginal_inverse_noise_matrix(noise_model=noise_model, clusters_in_marginal_list=[cluster])

        clusters_marginal_counts=probability.compute_marginals(results_dictionary=results_dictionary,subsets_list=[cluster])
        
        mitigated_clusters_marginal_probability_distribution = fda.convert_subset_counts_dictionary_to_probability_distribution(clusters_marginal_counts)

        mitigated_clusters_marginal_probability_distribution =np.array(mitigated_clusters_marginal_probability_distribution).reshape(mitigated_clusters_marginal_probability_distribution.shape[0], 1)

        mitigated_clusters_marginal_probability_distribution = clusters_inverse_noise_matrix.dot(mitigated_clusters_marginal_probability_distribution)

        final_marginal = np.kron(final_marginal,mitigated_clusters_marginal_probability_distribution)
        final_marginal = final_marginal.flatten()
        if ensure_proper_probability_distribution:
            if not probability.is_valid_probability_vector(final_marginal):
                final_marginal = povmtools.find_closest_prob_vector_l2(final_marginal)
                final_marginal = final_marginal.flatten()




    
    #creates a list of qubit indices involved in a marginal
    unordered_qubits_in_marginal_list = list(reduce(operator.concat, clusters_in_marginal_list))
    
    #creates an inverse noise matrix for the marginal 
    mitigated_clusters_marginal_probability_distribution = math.permute_composite_vector(unordered_qubits_in_marginal_list, final_marginal)

    #qubits list is sorted

    unordered_qubits_in_marginal_list.sort()

   
    mitigated_marginal_probability_distribution = probability.compute_marginal_of_probability_distribution(mitigated_clusters_marginal_probability_distribution ,[unordered_qubits_in_marginal_list.index(qubit) for qubit in marginal])


 

        

    #returns marginalization of an  extended marginal


    return mitigated_marginal_probability_distribution.flatten()




def mitigate_marginals_product(marginals_list: List[Tuple], results_dictionary: Dict[str, Dict[str, int]], noise_model: type[CNModelData] ,ensure_proper_probability_distribution = False )->Dict[str, Dict[tuple[int], np.array]]:
    
    #JT this dictionary 
    mitigated_marginals_dictionary ={}
   
    for marginal in marginals_list:
        mitigated_marginals_dictionary[marginal] = mitigate_marginal_product(marginal=marginal,results_dictionary=results_dictionary,noise_model=noise_model,ensure_proper_probability_distribution=ensure_proper_probability_distribution)
   
    return {next(iter(results_dictionary)): mitigated_marginals_dictionary}
