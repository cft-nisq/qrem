import random
from qrem.cn.simtools import auxiliary_merging_functions as m_f

from qrem.qtypes import CNModelData as CNModelData 
from qrem.common import probability

import numpy as np

from typing import Tuple, Dict, List, Optional
import time


#As for now neighbors functionality is not developed  

def create_random_noise_model(number_of_qubits:int,clusters_specification:List[List],neighbors:List[Tuple]=None) -> type[CNModelData]:
    '''
    Creates CNModelData with random noise. It used to generate data for tutorials. 
    Qubits are randomly assigned to clusters, number of clusters and their locality is specified by clusters_specification.
    Noise matrices assigned for clusters are random stochastic matrices. 

    

    Parameters
    ----------
    number_of_qubits - a number of qubits for which model is created 

    clusters_specification - a nested list containing description of clusters to be created:

            structure of the list [[element_1, element_2], ...]:
                element_1 - integer encoding locality of a cluster
                element_2 - number of clusters of given locality
            
            e.g. [[3,1],[2,2]] encodes 1 3 qubit cluster and two 2 qubit clusters

    neighbors - an optional parameter for neighbors specification, by default set to None 

    Returns
    -------
        noise_model: CNModelData object
             CNModelData object storing noise model data

    '''
    #[1] Create empty holder for model data CNModelData
    noise_model = CNModelData(number_of_qubits=number_of_qubits)

    #[2] Now qubits are randomly assigned to clusters of a provided structure
    clusters_tuple=m_f.divide_qubits_in_the_clusters([i for i in range(number_of_qubits)],clusters_specification=clusters_specification)
    noise_model.set_clusters_neighborhoods(clusters_neighborhoods=clusters_tuple)

    
    #[3] Random stochastic matrix is generated per each cluster, data is stored in a dictionary
    noise_model_dictionary ={}    

    if neighbors==None:
        #[3.1] TODO: now we don't suport neighbours, so it should be always true
        for element in clusters_tuple:
            noise_model_dictionary[element] = {'averaged': probability.random_stochastic_matrix(2**len(element))}
    noise_model.set_noise_matrices_dictionary(noise_matrices_dictionary=noise_model_dictionary)

    #[4] We should be set with simulated CNModelData
    return noise_model


def create_custom_noise_model (number_of_qubits:int, noise_matrices_dictionary:dict) -> type[CNModelData]:
    
    '''
    Creates CNModelData with data provided by the user. It used to debug mitigation routines. 
     

    

    Parameters
    ----------
    number_of_qubits: integer 
        A number of qubits for which model is created
    
    noise_matrices_dictionary: dictionary
        Dictionary specifying clusters and noise matrices. Key correspond to tupels of qubit indices belonging to the same cluster, values to corresponding nosie matrices.
        E.g . (0,2) : 4 by 4 stochastic matrix    


   
    Returns
    -------
        noise_model: CNModelData object
             CNModelData object storing noise model data
    '''
    
    
    #[1] Create empty holder for model data CNModelData
    noise_model = CNModelData(number_of_qubits=number_of_qubits)
    
    #[2] Object is filled with data provided by 
    noise_model.set_noise_model(noise_matrices_dictionary=noise_matrices_dictionary)

    return noise_model






#This function takes a dictionary consisting of DOT experiment setting (string describing unitaries) and result and transforms it accordingly to a given noise model
#The noise model is specified accordingly to QREM convection, dictionary where keys are tuples of integers encoding qubits belonging to a cluster and associated array of noise matrices on clusters (key averaged indicates absence of neighborhoods )


def simulate_noise_results_dictionary(results_dictionary: Dict[str, Dict[str, int]],
                            noise_model:type[CNModelData],
                            neighborhoods_tuples: Optional[Dict[Tuple[int], List[int]]]={}
                            ) -> Dict[str ,Dict]:
    '''
    Function simulates readout noise according to a noise model specified CNModelData on experimental data.
    It is used in tutorials. Function loops over provided (noiseless) results dictionary and transforms the
    counts according to a product probability distribution specified by a CN noise model.  

    Parameters
    ----------

    results_dictionary: dictionary 
        dictionary with results of an experiment

    noise_model: CNModelData object
        CNModelData object storing noise model data
    
    Returns
    ----------
    noisy_results_total
            dictionary with results of a an experiment with added readout noise


    '''
    noisy_results_dictionary={}

    for input_state, noiseless_results in results_dictionary.items():
        
        noisy_results_list = []
        cluster_samples_dict = {}
        for single_result, number_of_samples in noiseless_results.items():
            cluster_samples_dict[single_result]={}
            t0=time.time()
            for cluster_qubits, noise_matrix_on_cluster in (noise_model.noise_matrices).items():

                state_on_cluster = ''.join([single_result[x] for x in cluster_qubits])
                
                if noise_model.clusters_neighborhoods == {}:
                    noise_matrix = noise_matrix_on_cluster
                else:
                    noise_matrix = noise_matrix_on_cluster['averaged']

                prob_dist = noise_matrix[:,int(state_on_cluster,2)]

                cluster_dist = np.random.multinomial(n=number_of_samples, pvals=prob_dist)
                cluster_content = []
                for i in range(len(cluster_dist)):
                    for j in range(cluster_dist[i]):
                        cluster_content.append(i)

                cluster_sample = np.random.permutation(cluster_content)
                cluster_samples_dict[single_result][cluster_qubits]=cluster_sample
            

            for m in range(number_of_samples):
                
                state_string = list(single_result)

                for cluster_qubits in (noise_model.noise_matrices).keys():
                    k = len(cluster_qubits)
                    substring = f'{cluster_samples_dict[single_result][cluster_qubits][m]:0{k}b}'
                    
                    for i in range(k):
                        state_string[cluster_qubits[i]]=substring[i]
                    
                
                noisy_results_list.append("".join(state_string))
                
            
            
        
        noisy_results_dictionary[input_state]={}
        single_input_dict = noisy_results_dictionary[input_state]

        for result in noisy_results_list:
            if result in single_input_dict.keys():
                single_input_dict[result]+=1
            else:
                single_input_dict[result]=1
        
    
    return noisy_results_dictionary







def simulate_noise_results_dictionary_old(results_dictionary: Dict[str, Dict[str, int]],
                            noise_model:type[CNModelData],
                            neighborhoods_tuples: Optional[Dict[Tuple[int], List[int]]]={}
                            ) -> Dict[str ,Dict]:
    
    '''
    DEPRECATED: This function is replaced by the optimized version simulate_noise_results_dictionary() 
    '''
    
    


    #_noisy_results_total - a dictionary that holds results of noisy experiments, at the beginning empty

    noisy_results_total={}
    # an outer loop over settings and results from results_dictionary, a dictionary storing results of ideal (noiseless) experiment
    for settings, results_for_settings in results_dictionary.items():

        #an array for new noise samples, noise is added sample by sample, the qubit ordering according to the qubit cluster list
        
        new_results = np.array([ 0 for _ in range(2**len(list(results_for_settings.keys())[0]))])

        # for each input state a probability distribution encoding noise is created
        t0 = time.time()
        for input_state, number_of_samples in results_for_settings.items():

            

            # this loop goes over clusters and respective noise matrices, as for now we support only the simplified model without neighbors

            prob_dist_now =[]
            marginal_samples =[]
            first=True
            dim_total=1
            print(f"    {input_state},{number_of_samples}")
            for cluster_qubits, noise_matrices_on_cluster in (noise_model.noise_matrices).items() :
                
                

                dim_now=2**len(cluster_qubits)

               

            
                
                #a state of qubits belonging to the current cluster is established

                input_on_cluster = ''.join([input_state[x] for x in cluster_qubits])

                #the code below is prepared for clusters extension, which is not implemented now
                #TODO: FINISH this
                
                #cluster noise matrix is specified
                if neighborhoods_tuples == {}:
                    noise_matrix = noise_matrices_on_cluster
                else:
                    noise_matrix = noise_matrices_on_cluster['averaged']

                        

                # a transfer probability is chosen for a given state of the results

                prob_dist_now=noise_matrix[:, int(input_on_cluster, 2)]

                
         

                #samples are created according to respective product probabity distributions
                # the if statement is needed to handle the case when a first sample is crated (there are no other samples to join)
                # the general idea of joining different samples is the following
                # 1) a list of counts is crated according to a probability distribution e.g. for two qubits it is of a form [4,8,1,7] 
                # for an experiment with 20 shots
                # 2) The list is transformed to a randomized result list e.g [2,0,1,3,...]
                # 3) This list is joined with a previously generated measurement statistic
                                  
                if first:
                    marginal_samples = np.random.multinomial(n=number_of_samples,pvals=prob_dist_now)
                    marginal_samples=m_f.transform_marginal_statistics_to_list_of_results(marginal_samples)
                    
                    dim_total=dim_total*dim_now
                    
                    first=False
                else:

                      

                    marginal_samples_temp=m_f.transform_marginal_statistics_to_list_of_results(np.random.multinomial(n=number_of_samples,pvals=prob_dist_now))

                   
                    marginal_results=m_f.merge_samples_list(marginal_samples,dim_total,marginal_samples_temp,dim_now)

                    marginal_samples = m_f.transform_marginal_statistics_to_list_of_results(marginal_results)

                    dim_total=dim_total*dim_now
                
            #When the loop over clusters is finished new noisy result is updated 

            
            new_results = new_results + marginal_results
        t1=time.time()
        print(f"random sampling time {t1-t0} seconds")




           



            

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

            #the first for loop is over different cluster assigment

            for qubits_in_clusters in noise_model.clusters_tuple:

                #the second for loop is over qubits belonging to that cluster

                for qubit in qubits_in_clusters:
                    #order of qubits is corrected: in the noisy results qubit on counter place, in put on the place according to the qubit ordering
                    key[qubit] = key_on_clusters[counter]
                    counter+=1

            #now it is checked whether a given result appears in the noise results, if yes, it is assigned to a dictionary

            if new_results[i] != 0:

                noisy_results.update({''.join(key):new_results[i]})

        noisy_results_total.update({settings:noisy_results})
        t2=time.time()
        print(f"formatting time: {t2-t1} seconds")
    return noisy_results_total



#possibly faster version, test needed
#KKM: tried it ad hoc, it has some bugs
def simulate_noise_results_dictionary__new_version(results_dictionary: Dict[str, Dict[str, int]],
                            noise_model:type[CNModelData],
                            neighborhoods_tuples: Optional[Dict[Tuple[int], List[int]]]={}
                            ) -> Dict[str ,Dict]:
    
    '''
    DEPRECATED: This function is replaced by the optimized version simulate_noise_results_dictionary() 

    Function simulate readout noise according to a noise model specified CNModelData on experimental data.
    It is used in tutorials. Function loops over provided (noiseless) results dictionary and transforms the
    counts according to a product probability distribution specified by a CN noise model.  

    Parameters
    ----------

    results_dictionary: dictionary 
        dictionary with results of an experiment

    noise_model: CNModelData object
        CNModelData object storing noise model data
    
    Returns
    ----------
    noisy_results_total
            dictionary with results of a an experiment with added readout noise


    '''
    
    


    #_noisy_results_total - a dictionary that holds results of noisy experiments, at the beginning empty

    noisy_results_total={}
    # an outer loop over settings and results from results_dictionary, a dictionary storing results of ideal (noiseless) experiment
    
    for settings, results_for_settings in results_dictionary.items():
        #an array for new noise samples, noise is added sample by sample, the qubit ordering according to the qubit cluster list
        
        new_results = np.array([ 0 for _ in range(2**len(list(results_for_settings.keys())[0]))])

        # for each input state a probability distribution encoding noise is created
        t0 = time.time()
        for input_state, number_of_samples in results_for_settings.items():

            

            # this loop goes over clusters and respective noise matrices, as for now we support only the simplified model without neighbors

            prob_dist_now =[]
            marginal_samples =[]
            first=True
            dim_total=1
            for cluster_qubits, noise_matrices_on_cluster in (noise_model.noise_matrices).items() :
                
                

                dim_now=2**len(cluster_qubits)

               

            
                
                #a state of qubits belonging to the current cluster is established

                input_on_cluster = ''.join([input_state[x] for x in cluster_qubits])

                #the code below is prepared for clusters extension, which is not implemented now
                #TODO: FINISH this
                
                #cluster noise matrix is specified
                if neighborhoods_tuples == {}:
                    noise_matrix = noise_matrices_on_cluster
                else:
                    noise_matrix = noise_matrices_on_cluster['averaged']

                        

                # a transfer probability is chosen for a given state of the results

                prob_dist_now=noise_matrix[:, int(input_on_cluster, 2)]

                
         

                #samples are created according to respective product probabity distributions
                # the if statement is needed to handle the case when a first sample is crated (there are no other samples to join)
                # the general idea of joining different samples is the following
                # 1) a list of counts is crated according to a probability distribution e.g. for two qubits it is of a form [4,8,1,7] 
                # for an experiment with 20 shots
                # 2) The list is transformed to a randomized result list e.g [2,0,1,3,...]
                # 3) This list is joined with a previously generated measurement statistic
                                  
                if first:
                    marginal_samples = np.random.multinomial(n=number_of_samples,pvals=prob_dist_now)
                    #marginal_samples=m_f.transform_marginal_statistics_to_list_of_results(marginal_samples)
                    
                    #dim_total=dim_total*dim_now
                    
                    first=False
                else:

                      
                    marginal_samples_temp=np.random.multinomial(n=number_of_samples,pvals=prob_dist_now)    
                    #marginal_samples_temp=m_f.transform_marginal_statistics_to_list_of_results(np.random.multinomial(n=number_of_samples,pvals=prob_dist_now))

                   
                    #marginal_results=m_f.merge_samples_list(marginal_samples,dim_total,marginal_samples_temp,dim_now)

                    marginal_samples = m_f.merge_results_list(marginal_samples,marginal_samples_temp)

                    #dim_total=dim_total*dim_now

            #When the loop over clusters is finished new noisy result is updated 

            
            new_results = new_results + marginal_samples
        t1=time.time()
        print(f"random sampling time {t1-t0} seconds")
        




           



            

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

            #the first for loop is over different cluster assigment

            for qubits_in_clusters in noise_model.clusters_tuple:

                #the second for loop is over qubits belonging to that cluster

                for qubit in qubits_in_clusters:
                    #order of qubits is corrected: in the noisy results qubit on counter place, in put on the place according to the qubit ordering
                    key[qubit] = key_on_clusters[counter]
                    counter+=1

            #now it is checked whether a given result appears in the noise results, if yes, it is assigned to a dictionary

            if new_results[i] != 0:

                noisy_results.update({''.join(key):new_results[i]})

        noisy_results_total.update({settings:noisy_results})
        t2=time.time()
        print(f"formatting time: {t2-t1} seconds")
    return noisy_results_total





            

