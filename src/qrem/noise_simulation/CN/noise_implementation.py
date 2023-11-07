
import numpy as np

from typing import Tuple, Dict, List, Optional






#This fuction takes a dictionary consisting of DOT experiment setting (string describing unitaries) and result and transforms it accordingly to a given noise model
#The noise model is specified accordingly to QREM convection, ditionary where keay are tuples of integers encoding qubits belonging to a cluster and assosiated array of noise matrices on clusters (key avaraged indicates absence of neighborhoods )




import random as rd



"""
A function creating a string of outcomes from measurement statistics

input parameters:

marginal_results_list - a list with marginal measurement outcomes results are encoded in intgeres, eg [0,3,0,1,1,2] is a list of 6 results for a two qubit marginals 

"""

def transform_marginal_statistics_to_list_of_results(marginal_results):
    
    results_number= len(marginal_results)
    
    marginal_results_list=[]
    
    for i in range(results_number):
        for j in range(marginal_results[i]):
            marginal_results_list.append(i) 
    rd.shuffle(marginal_results_list)

    return marginal_results_list





"""
A function that creates a joint result list by gluing outcomes from two marginals lists 

input parameters:  

marginal_list_1 - a list contaning marinal results, results are encoded in intgeres, eg [0,3,0,1,1,2] is a list of 6 results for a two qubit marginals 

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









#
def create_joint_measurement_statisctic_from_marginals(marginal_results_list):
    
    joint_samplec=[1]

    for el in marginal_results_list:
        joint_sample = np.kron(joint_sample,el)
    
    return joint_sample 


'''
A function simulating adding noise to noiseless results of simulations

input parameters:  


results_dictionary - dictionary with results of a perfect DOT experiment

noise_model_description - a dictionary:
        keys - tuples corresponding to cluster assigments
        values - a dictionary with keys - 'averaged' - averaged noise matrices for clusters 
        
        neighborhoods_tuples - this is not used now, but can be used in the future when neighbors are added   

'''


def simulate_noise_results_dictionary(results_dictionary: Dict[str, Dict[str, int]],
                            noise_model_description: Dict[Tuple[int], Dict[str, np.ndarray]],
                            neighborhoods_tuples: Optional[Dict[Tuple[int], List[int]]]={}
                            ):

    #_noisy_results_total - a dictionary that holds results of noisy experiments, at the beginning empty

    noisy_results_total={}
    # an outer loop over settings and results from results_dictionary, a dictionary storing results of ideal (noisless) experiment
    for settings, results_for_settings in results_dictionary.items():

        #an array for new noise samples, noise is added sample by sample, the qubit ordering according to the qubit cluster list
        
        new_results=np.array([ 0 for _ in range(2**len(list(results_for_settings.keys())[0]))])

        # for each input state a probability distribution encoding noise is created

        for input_state, number_of_samples in results_for_settings.items():

            

            # this loop goes over clusters and respective noise matrices, as for now we support only the simplified model without nieghbours

            prob_dist_now =[]
            marginal_samples =[]
            first=True
            dim_total=1
            for cluster_qubits, noise_matrices_on_cluster in noise_model_description.items():
                
                

                dim_now=2**len(cluster_qubits)

               

            
                
                #a state of qubits belonging to the current cluster is established

                input_on_cluster = ''.join([input_state[x] for x in cluster_qubits])

                #the code below is prepared for clusters extension, which is not implemented now
                #TODO: FINISH this
                
                #cluster noise matrix is specified
                noise_matrix = noise_matrices_on_cluster['averaged'] 
                        

                # a transfer probability is chosen for a given state of the results

                prob_dist_now=noise_matrix[:, int(input_on_cluster, 2)]

                
         

                #samples are created according to respective product probabiity distributions
                # the if statment is needed to handle the case when a first sample is crated (there are no other samples to join)
                # the general idea of joining different samples is the following
                # 1) a list of counts is crated according to a probability distribution e.g. for two qubits it is of a formm [4,8,1,7] 
                # for an experiment with 20 shots
                # 2) The list is tranformed to a randomized result list e.g [2,0,1,3,...]
                # 3) This list is joined with a previously generated measurement statistic
                                  
                if first:
                    marginal_samples = np.random.multinomial(n=number_of_samples,pvals=prob_dist_now)
                    marginal_samples=transform_marginal_statistics_to_list_of_results(marginal_samples)
                    
                    dim_total=dim_total*dim_now
                    
                    first=False
                else:

                      

                    marginal_samples_temp=transform_marginal_statistics_to_list_of_results(np.random.multinomial(n=number_of_samples,pvals=prob_dist_now))

                   
                    marginal_results=merge_samples_list(marginal_samples,dim_total,marginal_samples_temp,dim_now)

                    marginal_samples = transform_marginal_statistics_to_list_of_results(marginal_results)

                    dim_total=dim_total*dim_now

            #When the loop over clusters is finished new noisy result is updated 

            
            new_results =new_results + marginal_results
   




           



            

        #In the above, the orders of qubits was that of clusters. Here the initial order of qubits is restored.

        noisy_results={}

        # this is a loop over all possible bitstrings (results) of qubits
        #the order of qubits is changed
        #key on clusters is a string corresponding to results

        key = ['0' for _ in  range(len(input_state))]

        for i in range(len(new_results)):

            #the bit string corresponding to results is created

            key_on_clusters = "{0:b}".format(2**len(input_state)+i)
            key_on_clusters = key_on_clusters[1:]

            #counter is a variable that goes over physical qubits

            counter=0

            #the first for loop is over different cluster assigments

            for qubits_in_clusters in noise_model_description.keys():

                #the second for loop is over qubits belonging to that cluster

                for qubit in qubits_in_clusters:
                    #order of qubits is corrected: in the noisy results qubit on counter place, in put on the place according to the qubit ordering
                    key[qubit] = key_on_clusters[counter]
                    counter+=1

            #now it is checked wheter a given result appers in the noise results, if yess, it is assigned to a dictionary

            if new_results[i] != 0:

                noisy_results.update({''.join(key):new_results[i]})

        noisy_results_total.update({settings:noisy_results})

    return noisy_results_total



