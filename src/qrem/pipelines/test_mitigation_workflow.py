



from qrem.cn import simulation as cnsimulation 
from qrem.mitigation import mitigation_routines

from qrem.benchmarks import hamiltonians
from qrem.characterization.characterization_routine import execute_characterization_workflow
from datetime import date
from qrem.pipelines import simulate_experiment
from qrem.qtypes.characterization_data import CharacterizationData







        



if __name__ == '__main__':
 




    #specify n#
    number_of_qubits = 10


    #################################################################################
    # noise model specification #
    # 1. Provide number of qubits #
    #2. There are two options now#
    #2.1 Enter noise matrices and cluster structure by hand (see option 2.1 below)
    #2.2 Create random noise model by specifying number of clusters for each locality (see option 2.2 below )
    #2. #
    #################################################################################


    ################################################################################
    ######### Option 2.1                                                   #########  
    ######### clusters: (0,7), (1,9), (5,6), (3,8), (2,) (4,)              #########
    ################################################################################

    """
    model_noise_matrices_dictionary = {}

    model_noise_matrices_dictionary[(0,7)] =  np.array([[0.3,0.1,0,0.1],
                                                [0,0.3,0.2,0],
                                                [0.7,0.3,0.6,0.1],
                                                [0,0.3,0.1,0.8]]
                                                )
    model_noise_matrices_dictionary[(1,9)] =  np.array([[0.7, 0.1, 0,   0.],
                                                [0,   0.6, 0., 0.1],
                                                [0.3, 0.2, 0.8, 0.2],
                                                [0,   0.1, 0.2, 0.7]]
                                                )
    model_noise_matrices_dictionary[(5,6)] =  np.array([[0.3,0.1,0,0.1],
                                                [0,0.3,0.2,0],
                                                [0.7,0.3,0.6,0.1],
                                                [0,0.3,0.1,0.8]]
                                                )
    model_noise_matrices_dictionary[(3,8)] =  np.array([[0.7, 0.1, 0,   0.],
                                                [0,   0.6, 0., 0.1],
                                                [0.3, 0.2, 0.8, 0.2],
                                                [0,   0.1, 0.2, 0.7]]
                                                )

    model_noise_matrices_dictionary[(2,)] =  np.array([  [0.9, 0.3],
                                                [0.1, 0.7]]
                                                
                                                )

    model_noise_matrices_dictionary[(4,)] =  np.array([  [0.9, 0.3],
                                                [0.1, 0.7]]
                                                
                                                )

    noise_model_simulation=cnsimulation.create_custom_noise_model(number_of_qubits=number_of_qubits,noise_matrices_dictionary=model_noise_matrices_dictionary)

    """

    ################################################################################
    ######### Option 2.2                                                   #########  
    ######### clusters:                                         `          #########
    ######### qubits 4: 1                                                  #########
    ######### qubits 3: 1                                                  #########
    ######### qubits 2: 1                                                  #########
    ######### qubits 1: 1                                                  #########
    ################################################################################



    number_of_4_qubit_clusters = 0

    number_of_3_qubit_clusters = 0

    number_of_2_qubit_clusters = 5

    number_of_1_qubit_clusters = 0

    clusters_specification = [[4,number_of_4_qubit_clusters], [3,number_of_3_qubit_clusters], [2, number_of_2_qubit_clusters], [1, number_of_1_qubit_clusters]]


    noise_model_simulation=cnsimulation.create_random_noise_model(number_of_qubits=number_of_qubits,clusters_specification=clusters_specification)


    marginals_to_mitigate=list(noise_model_simulation.clusters_tuple) 




    ######################################################################################
    #### mitigation data creation                                       ##################
    ####specify number of circuits, shots and directory where data is to be saved#########
    #### number_of_benchmark_circuits : number of circuits used to perform mitigation  ###
    ######################################################################################
    data_directory = 'C:\\CFT Chmura\\Theory of Quantum Computation\\QREM_Data\\ibm\\test_data\\'

    number_of_circuits = 200

    number_of_shots = 10**4

    number_of_benchmark_circuits = 50

    #characterization procedure is run, noise model data are computed
    noisy_experiment_dictionary = simulate_experiment.simulate_noisy_experiment(noise_model=noise_model_simulation,number_of_circuits=number_of_circuits,number_of_shots=number_of_shots,number_of_benchmark_circuits=number_of_benchmark_circuits,data_directory=data_directory,return_ideal_experiment_data=False,save_data=True)


    
    
    noisy_results_dictionary = noisy_experiment_dictionary['noisy_results_dictionary']
    
    
    
    benchmark_state_list = list(noisy_results_dictionary.keys())[-number_of_benchmark_circuits:]

    characterization_data_container = CharacterizationData()

    characterization_data_container= execute_characterization_workflow(results_dictionary=noisy_results_dictionary,return_old_mitigation_data=False,data_directory="C:\\CFT Chmura\\Theory of Quantum Computation\\QREM_Data\\tests",ground_states_list=benchmark_state_list)

    #current characterization data

   
     
    
  

    
   





    #################################################################################
    ###  Optional step                                                          #####
    ###  Uncomment to include created noise model in mitigation                 #####
    #################################################################################

    characterization_data_container.noise_model_list.append(noise_model_simulation) 





    #################################################################################
    #### Hamiltonians are generated                                ##################
    #################################################################################

    hamiltonians_dictionary =hamiltonians.create_hamiltonians_for_benchmarks(number_of_qubits=number_of_qubits, number_of_hamiltonians=number_of_benchmark_circuits, clause_density=4.0)



    energy_dictionary= hamiltonians.eigenstate_energy_calculation_and_estimation(characterization_data_container.benchmark_results_dictionary ,  characterization_data_container.benchmark_marginals_dictionary ,hamiltonians_dictionary)

    #################################################################################
    #### Mitigation routine starts                                 ##################
    #################################################################################


    """
    JT:
    Here NEW MITIGATION routine is run. It returns:

    noise_models_mitigated_energy_dictionary - dictionary with mitigated energy for the generated sets of Hamiltonians and input states 

    noise_models_marginals_dictionary - dictionary with mitigated marginals distribution 

    """
            
    noise_models_mitigation_results_dictionary = mitigation_routines.estimate_mitigated_energy_over_noise_models(results_dictionary=characterization_data_container.benchmark_results_dictionary ,hamiltonians_dictionary=hamiltonians_dictionary,noise_models_list=characterization_data_container.noise_model_list,return_marginals=True)

    noise_models_mitigated_energy_dictionary =noise_models_mitigation_results_dictionary['corrected_energy']
    noise_models_marginals_dictionary = noise_models_mitigation_results_dictionary['mitigated_marginals']

        


    """
    JT:
    Here NEW MITIGATION routine is run, however it is implemented differently (marginals for clusters are mitigated, and then necessary tensor products are computed). This is added for tests purposes.

    It returns:

    noise_models_mitigated_energy_dictionary_product - dictionary with mitigated energy for the generated sets of Hamiltonians and input states 

    noise_models_marginals_dictionary_product - dictionary with mitigated marginals distribution 

    """




    """
    JT:
    Here OLD MITIGATION routine is run.

    It returns:

    benchmarks_results- dictionary with mitigated energy for the generated sets of Hamiltonians and input states 

    benchmarks_results_analysis - data used to print median of mitigation 

    noise_models_marginals_dictionary_old - dictionary with mitigated marginals distribution 

    """

    #benchmarks_results,benchmarks_results_analysis=fun_ben.run_benchmarks(number_of_qubits,benchmarks_results_dictionary, benchmarks_marginals_dictionary, energy_dictionary, hamiltonians_dictionary,all_clusters_sets_dictionary,correction_matrices, correction_indices,mitigation_data_dictionary,noise_matrices_dictionary)


    #dictionary_to_save ={'mitigated_energy_dictionary' : noise_models_mitigated_energy_dictionary ,
    #                    'hamiltonians_energy_data' : energy_dictionary,
    #                    #'noise_models_marginals_dictionary_old':noise_models_marginals_dictionary_old,
    #                    'noise_models_marginals_dictionary': noise_models_marginals_dictionary,
    #                    'benchmarks_results_analysis_old' : benchmarks_results_analysis,
    #                    'benchmarks_results_dictionary':benchmarks_results_dictionary,
    #                    'benchmarks_marginals_dictionary': benchmarks_marginals_dictionary,
    #                    'hamiltonians_dictionary':hamiltonians_dictionary,
    #                    'all_clusters_sets_dictionary': all_clusters_sets_dictionary

                        
    #                    }


    



    #################################################################################
    #### Mitigation error is computed                                 ###############
    #################################################################################


    noise_models_mitigated_energy_dictionary_error = mitigation_routines.compute_mitigation_errors(mitigated_energy_dictionary=noise_models_mitigated_energy_dictionary,hamiltonian_energy_dictionary=energy_dictionary,number_of_qubits=number_of_qubits)



    ####################################################################################### 
    #### Mean of mitigation error for different noise models is computed and displayed  ###
    #######################################################################################


    mitigation_routines.compute_mitigation_error_median_mean(noise_models_mitigated_energy_dictionary_error, print_results=True)



    """
    print('Old mitigation routine')

    for key,results in benchmarks_results_analysis.items():
        print(key)
        print('Mitigation error mean', results['mean'])
        print('Mitigation error median', results['median'])


    #raw error
    raw_error_list = []
    for key,result in benchmarks_results.items():
        raw_error_list.append(result['errors']['error_raw'])
    print('Raw error mean')
    print(statistics.mean(raw_error_list))
    print('Raw error median')
    print(statistics.median(raw_error_list))
    """


    ####################################################################################### 
    #### Result of mitigation is saved                                                  ###
    #######################################################################################



    name_string = str(date.today()) + " q" + str(number_of_qubits) 

        

    file_name_mitigation_results  = 'DDOT_mitigation_results_workflow_' + name_string +'.pkl'

    #io.save(dictionary_to_save=dictionary_to_save,
    #                                directory= data_directory,
    #                                custom_filename=file_name_mitigation_results)





    """
    JT:
    Main task is to analyze how new and old mitigation routines perform in mitigation of marginal probability distributions. 

    This is especially interesting in cases when there is a large difference im median of energy mitigation between old and new mitigation routines.

    It may be useful to compute Total Variational distance between mitigated marginal probability distributions and ideal marginal probability distribution

    This would require, for each noise model and state used in energy estimation, to compute TVD between each entry of respective marginals dictionary and  ideal marginal probability distribution

    More precisely: for each noise model, and for each state

        A) compute TVD for each entry of noise_models_marginals_dictionary and ideal_benchmarks_dictionary 

        B) Repeat the same for  noise_models_marginals_dictionary_old  and ideal_benchmarks_dictionary

        C) (Optional) Repeat the same for  noise_models_marginals_dictionary_product  and ideal_benchmarks_dictionary 

    Notes:

    Unfortunately noise_models_marginals_dictionary_old  and noise_models_marginals_dictionary are indexed differently



    """

