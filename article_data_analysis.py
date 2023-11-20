import pickle
from qrem.pipelines.characterization_routine import execute_characterization_workflow
from  qrem.pipelines.mitigation_functions import estimate_mitigated_energy_over_noise_models
from qrem.pipelines.mitigation_functions import compute_noisy_energy_over_noise_models
from qrem.visualisation import article_plots_functions as apf
from qrem.functions_qrem import functions_benchmarks as fun_ben
from qrem.common.printer import qprint
from qrem.common import io
import pickle





################################################################################
############################### Data are loaded here ###########################
################################################################################
  
 
#set path to data directory
DATA_DIRECTORY= ''

#set name of file where bare counts are stored 
FILE_NAME_RESULTS_IBM =  'RESULTS_IBM_CUSCO.pkl'

#set name of file with ground states list, this is the states used in benchmarks
FILE_NAME_GROUND_STATES_LIST_IBM = 'IBM_CUSCO_GROUND_STATES_LIST.pkl '

#set name of file with states list used to calculate coherence strength 
COHERENCE_WITNESS_CIRCUITS_PATH_IBM = 'IBM_CUSCO_COHERENCE_WITNESS_CIRCUITS.pkl'


FILE_NAME_HAMILTONIANS_IBM = 'IBM_CUSCO_HAMILTONIANS_DICTIONARY.pkl'


#Optional: set name of file where marginal probability distributions are stored, if they were calculated previously
FILE_NAME_MARGINALS_IBM ='IBM_CUSCO_MARGINALS_DICTIONARY.pkl'





#experimental results are loaded
with open(DATA_DIRECTORY + FILE_NAME_RESULTS_IBM, 'rb') as filein:
    results_dictionary_ibm = pickle.load(filein)

#ground states list us loaded
with open(DATA_DIRECTORY+FILE_NAME_GROUND_STATES_LIST_IBM, 'rb') as filein:
    circuits_ground_states_preparation_collection_ibm= pickle.load( filein)

#coherence strength circuits are loaded 
with open(DATA_DIRECTORY+COHERENCE_WITNESS_CIRCUITS_PATH_IBM , 'rb') as filein:
    coherence_witness_circuits_ibm =  pickle.load(filein)

#Optional: marginals are loaded
with open(DATA_DIRECTORY+ FILE_NAME_HAMILTONIANS_IBM, 'rb') as filein:
    hamiltonians_dictionary_ibm = pickle.load(filein)

#Optional: marginals are loaded
with open(DATA_DIRECTORY + FILE_NAME_MARGINALS_IBM, 'rb') as filein:
    marginals_dictionary_ibm = pickle.load(filein)

marginals_dictionary_ibm =marginals_dictionary_ibm['marginals_dictionary']
        



FILE_NAME_RESULTS_RIG =  'RESULTS_RIGETTI_ASPEN-M-3.pkl'

FILE_NAME_GROUND_STATES_LIST_RIG = 'RIGETTI-ASPEN-M-3_GROUND_STATES_LIST.pkl'

COHERENCE_WITNESS_CIRCUITS_PATH_RIG = 'RIGETTI-ASPEN-M-3_COHERENCE_WITNESS_CIRCUITS.pkl'

FILE_NAME_HAMILTONIANS_RIG = 'RIGETTI-ASPEN-M-3_HAMILTONIANS_DICTIONARY.PKL'

FILE_NAME_MARGINALS_RIG = 'RIGETTI-ASPEN-M-3_MARGINALS_DICTIONARY.pkl'







#experimental results are loaded
with open(DATA_DIRECTORY + FILE_NAME_RESULTS_RIG, 'rb') as filein:
    results_dictionary_rig = pickle.load(filein)

#ground states list us loaded
with open(DATA_DIRECTORY+FILE_NAME_GROUND_STATES_LIST_RIG, 'rb') as filein:
    circuits_ground_states_preparation_collection_rig= pickle.load( filein)

#coherence strength circuits are loaded 
with open(DATA_DIRECTORY+COHERENCE_WITNESS_CIRCUITS_PATH_RIG  , 'rb') as filein:
    coherence_witness_circuits_rig =  pickle.load(filein)

#Optional: marginals are loaded
with open(DATA_DIRECTORY + FILE_NAME_HAMILTONIANS_RIG, 'rb') as filein:
    hamiltonians_dictionary_rig = pickle.load(filein)

#Optional: marginals are loaded
with open(DATA_DIRECTORY + FILE_NAME_MARGINALS_RIG, 'rb') as filein:
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
    
    io.save(dictionary_to_save=dictionary_to_save,custom_filename="ARTICLE_ANALYSIS",directory=DATA_DIRECTORY + '/tests')



    ######################################################################### 
    #### Fig 3  Distance to projective measurements                       ###
    #########################################################################

    apf.create_POVMs_distance_histogram(POVMs_errors_ibm=characterization_routine_results_dictionary_ibm['POVMs_distances_dictionary'],POVMs_errors_rig=characterization_routine_results_dictionary_rig['POVMs_distances_dictionary'],path_to_save=DATA_DIRECTORY + '/tests/POVMs_distances_histogram')

    ######################################################################### 
    #### Fig 3 histogram correlations coefficients                        ###
    #########################################################################

    apf.create_correlations_distance_histogram(correlations_coefficients_matrix_ibm=characterization_routine_results_dictionary_ibm['correlations_data']['worst_case']['classical'],correlations_coefficients_matrix_rig=characterization_routine_results_dictionary_rig['correlations_data']['worst_case']['classical'],path_to_save= DATA_DIRECTORY + '/tests/correlations_coefficients_histogram')

    ######################################################################### 
    #### Fig 4 (Only Windows ) cCorrelation coefficients                  ###
    #########################################################################

    

    ######################################################################### 
    #### Fig 5 Coherence Histogram                                        ###
    #########################################################################

    apf.create_coherence_bound_histogram(coherence_bound_dictionary_ibm=characterization_routine_results_dictionary_ibm['coherence_bound_dictionary'],coherence_bound_dictionary_rigetti=characterization_routine_results_dictionary_rig['coherence_bound_dictionary'],path_to_save=DATA_DIRECTORY + '/tests/coherence_bound_histogram')

    ######################################################################### 
    #### Fig 6 Error mitigation an Error prediction histogram             ###
    #########################################################################


    energy_dictionary_ibm= fun_ben.eigenstate_energy_calculation_and_estimation(results_dictionary=characterization_routine_results_dictionary_ibm['benchmarks_results_dictionary'],marginals_dictionary=characterization_routine_results_dictionary_ibm['benchmarks_marginals_dictionary'],hamiltonians_data=hamiltonians_dictionary_ibm)

    energy_dictionary_rig= fun_ben.eigenstate_energy_calculation_and_estimation(results_dictionary=characterization_routine_results_dictionary_rig['benchmarks_results_dictionary'],marginals_dictionary=characterization_routine_results_dictionary_rig['benchmarks_marginals_dictionary'],hamiltonians_data=hamiltonians_dictionary_rig)




    apf.create_error_mitigation_prediction_histogram(noise_models_mitigated_energy_dictionary=noise_models_mitigated_energy_dictionary_ibm,noise_models_predicted_energy_dictionary=noise_models_predicted_energy_dictionary_ibm,energy_dictionary=energy_dictionary_ibm,number_of_qubits=127,path_to_save = DATA_DIRECTORY + 'tests/ibm')

    apf.create_error_mitigation_prediction_histogram(noise_models_mitigated_energy_dictionary=noise_models_mitigated_energy_dictionary_rig,noise_models_predicted_energy_dictionary=noise_models_predicted_energy_dictionary_rig,energy_dictionary=energy_dictionary_rig,number_of_qubits=79,path_to_save = DATA_DIRECTORY + 'tests/rig')



  
                                            
    qprint("PLOTS GENERATED")