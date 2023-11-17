import pickle
from qrem.pipelines.characterization_routine import execute_characterization_workflow
from  qrem.pipelines.mitigation_functions import estimate_mitigated_energy_over_noise_models
from qrem.pipelines.mitigation_functions import compute_noisy_energy_over_noise_models
from qrem.visualisation import article_plots_functions as apf
import numpy as np
from qrem.functions_qrem import functions_benchmarks as fun_ben
from qrem.functions_qrem import ancillary_functions as anf
from qrem.common.printer import qprint


from matplotlib import pyplot as plt

from qrem.common import io




import sys
import os

import numpy as np
import pickle



from datetime import date
from typing import Dict





################################################################################
############################### Data are loaded here ###########################
################################################################################
  
 
#set path to data directory
DATA_DIRECTORY= '/media/tuzjan/T7/work_tuzjan/experiments_ibm/ibm_cusco_2023-10-01/'

#set name of file where bare counts are stored 
FILE_NAME_RESULTS_IBM =  'ibm_cusco_results.pkl'

#set name of file with ground states list, this is the states used in benchmarks
FILE_NME_GROUND_STATES_LIST_IBM = '/media/tuzjan/T7/work_tuzjan/ground_state_hamiltonians/ibm_ground_states_list.pkl'

#set name of file with states list used to calculate coherence strength 
COHERENCE_STRENGTH_CIRCUITS_PATH_IBM = '/media/tuzjan/T7/work_tuzjan/coherence witness circuits/coherence_witness_circuits_all_combinations_q127.pkl'

#

DIRECTORY_HAMILTONIANS= '/media/tuzjan/T7/work_tuzjan/ground_state_hamiltonians/'

FILE_NAME_HAMILTONIANS_IBM = 'hamiltonians_dictionary_600_hamiltonian_127_qubits.pkl'


#Optional: set name of file where marginal probability distributions are stored, if they were calculated previously
FILE_NAME_MARGINALS_IBM ='DDOT_marginals_workflow_2023-10-04 q127.pkl'





#experimental results are loaded
with open(DATA_DIRECTORY + FILE_NAME_RESULTS_IBM, 'rb') as filein:
    results_dictionary_ibm = pickle.load(filein)

#ground states list us loaded
with open(FILE_NME_GROUND_STATES_LIST_IBM, 'rb') as filein:
    circuits_ground_states_preparation_collection_ibm= pickle.load( filein)

#coherence strength circuits are loaded 
with open(COHERENCE_STRENGTH_CIRCUITS_PATH_IBM , 'rb') as filein:
    coherence_witness_circuits_ibm =  pickle.load(filein)

#Optional: marginals are loaded
with open(DIRECTORY_HAMILTONIANS + FILE_NAME_HAMILTONIANS_IBM, 'rb') as filein:
    hamiltonians_dictionary_ibm = pickle.load(filein)

#Optional: marginals are loaded
with open(DATA_DIRECTORY + FILE_NAME_MARGINALS_IBM, 'rb') as filein:
    marginals_dictionary_ibm = pickle.load(filein)

marginals_dictionary_ibm =marginals_dictionary_ibm['marginals_dictionary']
        

DATA_DIRECTORY_RIG = '/media/tuzjan/T7/work_tuzjan/experiments_rigetti/'

FILE_NAME_RESULTS_RIG =  'rigetti_results.pkl'

FILE_NAME_MARGINALS_RIG = 'DDOT_marginals_workflow_2023-09-29 q79Rigetti_4.pkl'

COHERENCE_WITNESS_CIRCUITS_PATH_RIG = '/media/tuzjan/T7/work_tuzjan/coherence witness circuits/coherence_witness_circuits_q79.pkl'


FILE_NAME_HAMILTONIANS_RIG = 'hamiltonians_dictionary_100_hamiltonian_79_qubits.pkl'

FILE_NAME_GROUND_STATES_LIST_RIG = '/media/tuzjan/T7/work_tuzjan/ground_state_hamiltonians/hamiltonians_dictionary_50_gorund_states_79_qubits.pkl'

GROUND_STATES_NUMBER = 50






#experimental results are loaded
with open(DATA_DIRECTORY_RIG + FILE_NAME_RESULTS_RIG, 'rb') as filein:
    results_dictionary_rig = pickle.load(filein)

#ground states list us loaded
with open(FILE_NAME_GROUND_STATES_LIST_RIG, 'rb') as filein:
    circuits_ground_states_preparation_collection_rig= pickle.load( filein)

#coherence strength circuits are loaded 
with open(COHERENCE_WITNESS_CIRCUITS_PATH_RIG  , 'rb') as filein:
    coherence_witness_circuits_rig =  pickle.load(filein)

#Optional: marginals are loaded
with open(DIRECTORY_HAMILTONIANS + FILE_NAME_HAMILTONIANS_RIG, 'rb') as filein:
    hamiltonians_dictionary_rig = pickle.load(filein)

#Optional: marginals are loaded
with open(DATA_DIRECTORY_RIG + FILE_NAME_MARGINALS_RIG, 'rb') as filein:
    marginals_dictionary_rig = pickle.load(filein)








qprint("DATA LOADED")

if __name__ == "__main__":
    
    qprint("CHARACTERIZATION STARTS")
    
    characterization_routine_results_dictionary_ibm = execute_characterization_workflow(results_dictionary=results_dictionary_ibm,marginals_dictionary=marginals_dictionary_ibm,number_of_benchmark_circuits=300,ground_states_list=circuits_ground_states_preparation_collection_ibm,return_old_mitigation_data=True,data_directory=DATA_DIRECTORY,coherence_witnesses_list=list(coherence_witness_circuits_ibm),perform_noise_model_reconstruction=True,name_id='IBM_Cusco')
 
    characterization_routine_results_dictionary_rig = execute_characterization_workflow(results_dictionary=results_dictionary_rig,marginals_dictionary=marginals_dictionary_rig['marginals_dictionary'],number_of_benchmark_circuits=50,ground_states_list=circuits_ground_states_preparation_collection_rig,return_old_mitigation_data=True,data_directory=DATA_DIRECTORY,coherence_witnesses_list=list(coherence_witness_circuits_rig),perform_noise_model_reconstruction=True,name_id = 'Rigetti_Aspen-M-3')


    qprint("CHARACTERIZATION ENDS")

   

    #################################################################################
    #### Mitigation routine starts                                 ##################
    #################################################################################

    qprint("MITIGATION STARTS")

          
    noise_models_mitigated_energy_dictionary_ibm = estimate_mitigated_energy_over_noise_models(results_dictionary=characterization_routine_results_dictionary_ibm['benchmarks_results_dictionary'],hamiltonians_dictionary=hamiltonians_dictionary_ibm,noise_models_list=characterization_routine_results_dictionary_ibm['noise_models_list'],return_marginals=False)

    noise_models_mitigated_energy_dictionary_rig = estimate_mitigated_energy_over_noise_models(results_dictionary=characterization_routine_results_dictionary_rig['benchmarks_results_dictionary'],hamiltonians_dictionary=hamiltonians_dictionary_rig,noise_models_list=characterization_routine_results_dictionary_rig['noise_models_list'],return_marginals=False)

    qprint("MITIGATION ENDS")



    #data_dictionary = io.load('/media/tuzjan/T7/work_tuzjan/tests/2023-11-16 qibm_rig.pkl')

    #characterization_routine_results_dictionary_ibm= data_dictionary['characterization_dictionary_ibm' ]

    #characterization_routine_results_dictionary_rig= data_dictionary['characterization_dictionary_rig' ]

    #noise_models_mitigated_energy_dictionary_rig=data_dictionary['noise_models_mitigated_energy_dictionary_rig']

    #noise_models_mitigated_energy_dictionary_ibm=data_dictionary['noise_models_mitigated_energy_dictionary_ibm']


    noise_models_predicted_energy_dictionary_ibm = compute_noisy_energy_over_noise_models(results_dictionary=characterization_routine_results_dictionary_ibm['benchmarks_results_dictionary'],hamiltonians_dictionary=hamiltonians_dictionary_ibm,noise_models_list=characterization_routine_results_dictionary_ibm['noise_models_list'])

    noise_models_predicted_energy_dictionary_rig = compute_noisy_energy_over_noise_models(results_dictionary=characterization_routine_results_dictionary_rig['benchmarks_results_dictionary'],hamiltonians_dictionary=hamiltonians_dictionary_rig,noise_models_list=characterization_routine_results_dictionary_rig['noise_models_list'])






    ####################################################################################### 
    #### Result of mitigation is saved                                                  ###
    #######################################################################################

    dictionary_to_save ={'characterization_dictionary_ibm' : characterization_routine_results_dictionary_ibm ,
                        'characterization_dictionary_rig' : characterization_routine_results_dictionary_rig,
                        'noise_models_mitigated_energy_dictionary_ibm' : noise_models_mitigated_energy_dictionary_ibm,
                        'noise_models_mitigated_energy_dictionary_rig' : noise_models_mitigated_energy_dictionary_rig,
                        'noise_models_predicted_energy_dictionary_rig': noise_models_predicted_energy_dictionary_rig,
                        'noise_models_predicted_energy_dictionary_ibm':noise_models_predicted_energy_dictionary_ibm

                       }
    
    io.save(dictionary_to_save=dictionary_to_save,custom_filename="ARTICLE_ANALYSIS",directory='/media/tuzjan/T7/work_tuzjan/tests')



    ######################################################################### 
    #### Fig 3  Distance to projective measurements                       ###
    #########################################################################

    apf.create_POVMs_distance_histogram(POVMs_errors_ibm=characterization_routine_results_dictionary_ibm['POVMs_distances_dictionary'],POVMs_errors_rig=characterization_routine_results_dictionary_rig['POVMs_distances_dictionary'],path_to_save='/media/tuzjan/T7/work_tuzjan/tests/POVMs_distances_histogram')

    ######################################################################### 
    #### Fig 3 histogram correlations coefficients                        ###
    #########################################################################

    apf.create_correlations_distance_histogram(correlations_coefficients_matrix_ibm=characterization_routine_results_dictionary_ibm['correlations_data']['worst_case']['classical'],correlations_coefficients_matrix_rig=characterization_routine_results_dictionary_rig['correlations_data']['worst_case']['classical'],path_to_save='/media/tuzjan/T7/work_tuzjan/tests/correlations_coefficients_histogram')

    ######################################################################### 
    #### Fig 4 (Only Windows ) cCorrelation coefficients                  ###
    #########################################################################

    

    ######################################################################### 
    #### Fig 5 Coherence Histogram                                        ###
    #########################################################################

    apf.create_coherence_bound_histogram(coherence_bound_dictionary_ibm=characterization_routine_results_dictionary_ibm['coherence_bound_dictionary'],coherence_bound_dictionary_rigetti=characterization_routine_results_dictionary_rig['coherence_bound_dictionary'],path_to_save='/media/tuzjan/T7/work_tuzjan/tests/coherence_bound_histogram')

    ######################################################################### 
    #### Fig 6 Error mitigation an Error prediction histogram             ###
    #########################################################################


    energy_dictionary_ibm= fun_ben.eigenstate_energy_calculation_and_estimation(results_dictionary=characterization_routine_results_dictionary_ibm['benchmarks_results_dictionary'],marginals_dictionary=characterization_routine_results_dictionary_ibm['benchmarks_marginals_dictionary'],hamiltonians_data=hamiltonians_dictionary_ibm)

    energy_dictionary_rig= fun_ben.eigenstate_energy_calculation_and_estimation(results_dictionary=characterization_routine_results_dictionary_rig['benchmarks_results_dictionary'],marginals_dictionary=characterization_routine_results_dictionary_rig['benchmarks_marginals_dictionary'],hamiltonians_data=hamiltonians_dictionary_rig)




    apf.create_error_mitigation_prediction_histogram(noise_models_mitigated_energy_dictionary=noise_models_mitigated_energy_dictionary_ibm,noise_models_predicted_energy_dictionary=noise_models_predicted_energy_dictionary_ibm,energy_dictionary=energy_dictionary_ibm,number_of_qubits=127,path_to_save = '/media/tuzjan/T7/work_tuzjan/tests/ibm')

    apf.create_error_mitigation_prediction_histogram(noise_models_mitigated_energy_dictionary=noise_models_mitigated_energy_dictionary_rig,noise_models_predicted_energy_dictionary=noise_models_predicted_energy_dictionary_rig,energy_dictionary=energy_dictionary_rig,number_of_qubits=79,path_to_save = '/media/tuzjan/T7/work_tuzjan/tests/rig')



  
                                            
    qprint("PLOTS GENERATED")