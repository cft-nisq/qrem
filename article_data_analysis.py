import pickle
from qrem.pipelines.characterization_routine import execute_characterization_workflow
from  qrem.pipelines.mitigation_functions import estimate_mitigated_energy_over_noise_models
from  qrem.pipelines.mitigation_functions import compute_mitigation_errors 

import numpy as np
from qrem.functions_qrem import functions_benchmarks as fun_ben
from qrem.functions_qrem import ancillary_functions as anf
from qrem.common.printer import qprint




import sys
import os

import numpy as np
import pickle



from datetime import date
from typing import Dict



def change_dictionary_format(noise_matrix:Dict) -> Dict:

    new_format ={}
    
    for key, matrix in noise_matrix.items():

        if key != 'averaged':
            
        
                   
            new_index = [int(character) for character in key]
        
            new_format[tuple(new_index)] = matrix
        
        else:

            new_format[key] = matrix

    return new_format

################################################################################
############################### Data are loaded here ###########################
################################################################################
  
 
#set path to data directory
DATA_DIRECTORY= '/media/tuzjan/T7/work_tuzjan/experiments_ibm/ibm_cusco_2023-10-01/'

#set name of file where bare counts are stored 
FILE_NAME_RESULTS =  'ibm_cusco_results.pkl'

#set name of file with ground states list, this is the states used in benchmarks
FILE_NME_GROUND_STATES_LIST = '/media/tuzjan/T7/work_tuzjan/ground_state_hamiltonians/ibm_ground_states_list.pkl'

#set name of file with states list used to calculate coherence strength 
COHERENCE_STRENGTH_CIRCUITS_PATH = '/media/tuzjan/T7/work_tuzjan/coherence witness circuits/coherence_witness_circuits_all_combinations_q127.pkl'

#

DIRECTORY_HAMILTONIANS= '/media/tuzjan/T7/work_tuzjan/ground_state_hamiltonians/'

FILE_NAME_HAMILTONIANS = 'hamiltonians_dictionary_600_hamiltonian_127_qubits.pkl'


#Optional: set name of file where marginal probability distributions are stored, if they were calculated previously
FILE_NAME_MARGINALS ='DDOT_marginals_workflow_2023-10-04 q127.pkl'





#experimental results are loaded
with open(DATA_DIRECTORY + FILE_NAME_RESULTS, 'rb') as filein:
    results_dictionary = pickle.load(filein)

#ground states list us loaded
with open(FILE_NME_GROUND_STATES_LIST, 'rb') as filein:
    circuits_ground_states_preparation_collection= pickle.load( filein)

#coherence strength circuits are loaded 
with open(COHERENCE_STRENGTH_CIRCUITS_PATH , 'rb') as filein:
    coherence_witness_circuits =  pickle.load(filein)

#Optional: marginals are loaded
with open(DIRECTORY_HAMILTONIANS + FILE_NAME_HAMILTONIANS, 'rb') as filein:
    hamiltonians_dictionary = pickle.load(filein)

#Optional: marginals are loaded
with open(DATA_DIRECTORY + FILE_NAME_MARGINALS, 'rb') as filein:
    marginals_dictionary = pickle.load(filein)

marginals_dictionary =marginals_dictionary['marginals_dictionary']
        
       
   



qprint("DATA LOADED")

if __name__ == "__main__":
    characterization_routine_results_dictionary = execute_characterization_workflow(results_dictionary=results_dictionary,marginals_dictionary=marginals_dictionary,number_of_benchmark_circuits=300,ground_states_list=circuits_ground_states_preparation_collection,return_old_mitigation_data=True,data_directory=DATA_DIRECTORY,coherence_witnesses_list=list(coherence_witness_circuits),perform_noise_model_reconstruction=True)
 
    #below data and data structures used in new mitigation routine are extracted from characterization routine 
    noise_model_list = characterization_routine_results_dictionary['noise_models_list']

    benchmarks_results_dictionary = characterization_routine_results_dictionary['benchmarks_results_dictionary']

    benchmarks_marginals_dictionary = characterization_routine_results_dictionary['benchmarks_marginals_dictionary']

    #below data and data structures used in old mitigation routine are extracted from characterization routine 

    correction_matrices = characterization_routine_results_dictionary['correction_matrices']  
    
    correction_indices = characterization_routine_results_dictionary['correction_indices']   
    
    mitigation_data_dictionary = characterization_routine_results_dictionary['mitigation_data_dictionary']    
    
    all_clusters_sets_dictionary = characterization_routine_results_dictionary['all_clusters_sets_dictionary']  
    
    noise_matrices_dictionary = characterization_routine_results_dictionary['noise_matrices_dictionary_old'] 


    for noise_model in noise_model_list:

        for cluster, noise_matrix in noise_model.noise_matrices.items():

            noise_model.noise_matrices[cluster] = change_dictionary_format(noise_matrix=noise_matrix)

        #noise_model.clusters_neighborhoods[cluster] = None

    

        






    noise_model_list = [noise_model_list[0]]#,noise_model_list[1]]#,noise_model_list[1]]










    energy_dictionary= fun_ben.eigenstate_energy_calculation_and_estimation(benchmarks_results_dictionary,benchmarks_marginals_dictionary,hamiltonians_dictionary)

    qprint("MITIGATION STARTS")

#################################################################################
#### Mitigation routine starts                                 ##################
#################################################################################



#benchmarks_results,benchmarks_results_analysis=fun_ben.run_benchmarks(number_of_qubits,benchmarks_results_dictionary, benchmarks_marginals_dictionary, energy_dictionary, hamiltonians_dictionary,all_clusters_sets_dictionary,correction_matrices, correction_indices,mitigation_data_dictionary,noise_matrices_dictionary)

#benchmarks_results=fun_ben.run_benchmarks(number_of_qubits,benchmarks_results_dictionary, benchmarks_marginals_dictionary, energy_dictionary, hamiltonians_dictionary,all_clusters_sets_dictionary,correction_matrices, correction_indices,mitigation_data_dictionary,noise_matrices_dictionary)



#raw error
#raw_error_list = []
#for key,result in benchmarks_results.items():
#    raw_error_list.append(result['errors']['error_raw'])
#print('Raw error median')
#print(statistics.median(raw_error_list))


#print('Old mitigation routine')

#for key,results in benchmarks_results_analysis.items():
#    print(key, 'Mitigation error mean', results['mean'], 'Mitigation error median', results['median'],)




          
    noise_models_mitigated_energy_dictionary=    estimate_mitigated_energy_over_noise_models(results_dictionary=benchmarks_results_dictionary,hamiltonians_dictionary=hamiltonians_dictionary,noise_models_list=noise_model_list,return_marginals=False)


#################################################################################
#### Mitigation error is computed                                 ###############
#################################################################################

    number_of_qubits=127
    noise_models_mitigated_energy_dictionary_error = compute_mitigation_errors(mitigated_energy_dictionary=noise_models_mitigated_energy_dictionary,hamiltonian_energy_dictionary=energy_dictionary,number_of_qubits=number_of_qubits)

    qprint("MITIGATION ENDS")



####################################################################################### 
#### Mean of mitigation error for different noise models is computed and displayed  ###
#######################################################################################


#mitigation_routines.compute_mitigation_error_median_mean(noise_models_mitigated_energy_dictionary_error)
    """
    JT:
    Here NEW MITIGATION routine is run, however it is implemented differently (marginals for clusters are mitigated, and then necessary tensor products are computed). This is added for tests purposes.

    It returns:

    noise_models_mitigated_energy_dictionary_product - dictionary with mitigated energy for the generated sets of Hamiltonians and input states 

    noise_models_marginals_dictionary_product - dictionary with mitigated marginals distribution 

    """


#noise_models_mitigated_energy_dictionary_product = mitigation_routines.estimate_mitigated_energy_over_noise_models(results_dictionary=benchmarks_results_dictionary,hamiltonians_dictionary=hamiltonians_dictionary,noise_models_list=noise_model_list,product_mitigation=True,return_marginals=False)



    dictionary_to_save ={'mitigated_energy_dictionary' : noise_models_mitigated_energy_dictionary ,
                        #'noise_models_mitigated_energy_dictionary_product' : noise_models_mitigated_energy_dictionary_product,
                        'hamiltonians_energy_data' : energy_dictionary,
                        #'ideal_benchmarks_marginals_dictionary': ideal_benchmarks_dictionary,
                        #'noise_models_marginals_dictionary_old':noise_models_marginals_dictionary_old,
                        #'noise_models_marginals_dictionary': noise_models_marginals_dictionary,
                        #'noise_models_marginals_dictionary_product': noise_models_marginals_dictionary_product,
                        #'benchmarks_results_analysis_old' : benchmarks_results_analysis,
                        'benchmarks_results_dictionary':benchmarks_results_dictionary,
                        #'benchmarks_execution_results': benchmarks_results,
                        #'benchmarks_marginals_dictionary': benchmarks_marginals_dictionary,
                        #'hamiltonians_dictionary':hamiltonians_dictionary,

                        #'benchmark_results_old':benchmarks_results,
                        'all_clusters_set_dictionary':all_clusters_sets_dictionary,
                        'noise_models_mitigated_energy_dictionary_error' : noise_models_mitigated_energy_dictionary_error

                        
                        }
    #name_string = str(date.today()) + " q" + str(number_of_qubits)

        

    


#noise_models_mitigated_energy_product_dictionary_error = mitigation_routines.compute_mitigation_errors(mitigated_energy_dictionary=noise_models_mitigated_energy_dictionary_product,hamiltonian_energy_dictionary=energy_dictionary,number_of_qubits=number_of_qubits)

#mitigation_routines.compute_mitigation_error_mean(noise_models_mitigated_energy_product_dictionary_error,'Mitigation error mean product ')

#noise_models_mitigated_energy_dictionary_error = mitigation_routines.compute_mitigation_errors(mitigated_energy_dictionary=noise_models_mitigated_energy_dictionary,hamiltonian_energy_dictionary=energy_dictionary,number_of_qubits=number_of_qubits)

#mitigation_routines.compute_mitigation_error_mean(noise_models_mitigated_energy_dictionary_error,'Mitigation error mean ')




#################################################################################
#### Mitigation error is computed                                 ###############
#################################################################################


#noise_models_mitigated_energy_dictionary_error = mitigation_routines.compute_mitigation_errors(mitigated_energy_dictionary=noise_models_mitigated_energy_dictionary,hamiltonian_energy_dictionary=energy_dictionary,number_of_qubits=number_of_qubits)

#noise_models_mitigated_energy_product_dictionary_error = mitigation_routines.compute_mitigation_errors(mitigated_energy_dictionary=noise_models_mitigated_energy_dictionary_product,hamiltonian_energy_dictionary=energy_dictionary,number_of_qubits=number_of_qubits)


####################################################################################### 
#### Mean of mitigation error for different noise models is computed and displayed  ###
#######################################################################################


#mitigation_routines.compute_mitigation_error_mean(noise_models_mitigated_energy_dictionary_error,'Mitigation error mean ')

#mitigation_routines.compute_mitigation_error_mean(noise_models_mitigated_energy_product_dictionary_error,'Mitigation error mean product ')




####################################################################################### 
#### Result of mitigation is saved                                                  ###
#######################################################################################

