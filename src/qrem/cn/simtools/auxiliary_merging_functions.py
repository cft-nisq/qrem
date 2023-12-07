import random
import numpy as np
from typing import List


def divide_qubits_in_the_clusters(qubits_indices:List, clusters_specification:List[List])->List[tuple]:
    '''A function used to divide qubits into clusters 

    Parameters
    ----------

    qubits_indices: list
        a list of qubits indices defined upon ..... 

    clusters_specification: list
        - a nested list containing description of clusters to be created:

            structure of the list [[element_1, element_2], ...]:
                element_1 - integer encoding locality of a cluster
                element_2 - number of clusters of given locality
            
            e.g. [[3,1],[2,2]] encodes 1 3 qubits cluster and two 2 qubits clusters

    Returns
    -------        
    cluster_list : 
        list with qubits assigned to clusters

              

    '''
    qubit_indices_now=qubits_indices
    cluster_list = []
    cluster_now =[]
    # a loop over 
    for el in clusters_specification:
        
        #a sublist of tot
        np.random.seed()
        cluster_now=random.sample(list(qubit_indices_now), el[0]*el[1] )
        qubit_indices_now = set(qubit_indices_now) - set(cluster_now)

        cluster_now=[cluster_now[i:i+el[0]] for i in range(0,el[0]*el[1],el[0])]
        for element in cluster_now:
            element.sort()
            cluster_list.append(tuple(element))
    return tuple(cluster_list)


# Deprecated

"""
A function creating a string of outcomes from measurement statistics

input parameters:

marginal_results_list - a list with marginal measurement outcomes results are encoded in integers, eg [0,3,0,1,1,2] is a list of 6 results for a two qubit marginals 

"""

def transform_marginal_statistics_to_list_of_results(marginal_results):
    
    results_number= len(marginal_results)
    
    marginal_results_list=[]
    
    for i in range(results_number):
        for j in range(marginal_results[i]):
            marginal_results_list.append(i) 
    random.shuffle(marginal_results_list)

    return marginal_results_list





"""
A function that creates a joint result list by gluing outcomes from two marginals lists 

input parameters:  

marginal_list_1 - a list containing marginal results, results are encoded in integers, eg [0,3,0,1,1,2] is a list of 6 results for a two qubit marginals 

marginal_list_2 - as above 

output parameters:

results_list - a join list of results



"""

'''
A creating a joint sample out of samples created according to product probability distributions

input parameters:  


marginal_results_list - a list of list corresponding to product samples


output parameters:

joint_sample - a list corresponding to the joint sample


'''

def merge_samples_list(marginal_list_1, dim1, marginal_list_2, dim2):
    


    results_list = [0 for i in range(dim1*dim2)]

    for r1, r2 in zip(marginal_list_1,marginal_list_2):
        results_list[r1*dim2+r2]+=1

    return results_list



def merge_results_list(results_1, results_2):

    """

    Function joins marginal results lists. Used in simulation of CN noise model
   
   
   Parameters
    ----------

    results_1: list 
        list with marginal results of an experiment

    results_2: list
        list with marginal results of an experiment
    
    Returns
    ----------
    merged_results
        list with merged results of experiment


    """

    dim_1 = len(results_1)
    dim_2 = len(results_2)
    merged_results = [0 for i in range(dim_1*dim_2)]

    for index_1 in range(dim_1):
        
        for index_2 in range(dim_2):
        
            if results_1[index_1] >= results_2[index_2]:
                element = results_2[index_2]
                merged_results[index_1*dim_1+index_2] = element
                results_1[index_1] = results_1[index_1] - element
                results_2[index_2] = 0
        
            else:
                element = results_1[index_1]
                merged_results[index_1*dim_1+index_2] = element
                results_1[index_1] = 0
                results_2[index_2] = results_2[index_2] - element
                break
    
    return merged_results


def t1():
    list_2 = [80,100]
    list_1 = [60,120]
    print(merge_results_list(list_1,list_2))

if __name__ == "__main__":
    t1()
