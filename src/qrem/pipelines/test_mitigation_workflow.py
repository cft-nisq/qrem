from qrem.cn import simulation as cnsimulation 
from qrem.mitigation import mitigation_routines
from qrem.benchmarks import hamiltonians
from qrem.characterization import characterization_routine
from datetime import date
from qrem.pipelines import simulate_experiment
from qrem.qtypes.characterization_data import CharacterizationData
from qrem.qtypes.mitigation_data import MitigationData


       



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
    ######### Option 2.2                                                   #########  
    ######### clusters:                                         `          #########
    ######### qubits 4: 1                                                  #########
    ######### qubits 3: 1                                                  #########
    ######### qubits 2: 1                                                  #########
    ######### qubits 1: 1                                                  #########
    ################################################################################



    number_of_4_qubit_clusters = 0

    number_of_3_qubit_clusters = 1

    number_of_2_qubit_clusters = 3

    number_of_1_qubit_clusters = 1

    clusters_specification = [[4,number_of_4_qubit_clusters], [3,number_of_3_qubit_clusters], [2, number_of_2_qubit_clusters], [1, number_of_1_qubit_clusters]]


    noise_model_simulation=cnsimulation.create_random_noise_model(number_of_qubits=number_of_qubits,clusters_specification=clusters_specification)


    marginals_to_mitigate=list(noise_model_simulation.clusters_tuple) 




    ######################################################################################
    #### mitigation data creation                                       ##################
    ####specify number of circuits, shots and directory where data is to be saved#########
    #### number_of_benchmark_circuits : number of circuits used to perform mitigation  ###
    ######################################################################################
    data_directory = 'C:\\CFT Chmura\\Theory of Quantum Computation\\QREM_Data\\ibm\\test_data\\'

    number_of_circuits = 400

    number_of_shots = 10**4

    include_benchmark_circuits = True
    
    number_of_benchmark_circuits = 20

    #characterization procedure is run, noise model data are computed

    characterization_data_container = CharacterizationData()

    characterization_data_container.experiment_type = 'DDOT'

   
    if include_benchmark_circuits:
        
        hamiltonians_dictionary, circuits_ground_states = hamiltonians.create_hamiltonians_and_ground_states(number_of_qubits=number_of_qubits,number_of_benchmark_circuits=number_of_benchmark_circuits)
    
    characterization_data_container.results_dictionary = simulate_experiment.simulate_noisy_experiment(noise_model=noise_model_simulation,number_of_circuits=number_of_circuits,number_of_shots=number_of_shots,data_directory=data_directory,return_ideal_experiment_data=False,save_data=True,new_data_format=True,ground_states_circuits=circuits_ground_states)['new_data_format'] 
   
    characterization_data_container.ground_states_list  = list(characterization_data_container.results_dictionary)[-number_of_benchmark_circuits:]
   
    characterization_data_container= characterization_routine.execute_characterization_workflow(characterization_data_container=characterization_data_container,find_neighbors=True)

    #################################################################################
    ###  Optional step                                                          #####
    ###  Uncomment to include created noise model in mitigation                 #####
    #################################################################################

    characterization_data_container.noise_model_list.append(noise_model_simulation) 

    #################################################################################
    #### Hamiltonians are generated                                ##################
    #################################################################################

    energy_dictionary= hamiltonians.eigenstate_energy_calculation_and_estimation(characterization_data_container.benchmark_results_dictionary ,  characterization_data_container.benchmark_marginals_dictionary ,hamiltonians_dictionary)

    #################################################################################
    #### Mitigation routine starts                                 ##################
    #################################################################################

    mitigation_data = MitigationData()
            
    noise_models_mitigation_results_dictionary = mitigation_routines.estimate_mitigated_energy_over_noise_models(characterization_data=characterization_data_container ,hamiltonians_dictionary=hamiltonians_dictionary,return_marginals=True)

     
    mitigation_data.noise_models_mitigation_results_dictionary = noise_models_mitigation_results_dictionary    
  



    #################################################################################
    #### Mitigation error is computed                                 ###############
    #################################################################################


    noise_models_mitigated_energy_dictionary_error = mitigation_routines.compute_mitigation_errors(mitigation_data=mitigation_data,hamiltonian_energy_dictionary=energy_dictionary,number_of_qubits=number_of_qubits)

    mitigation_data.noise_models_mitigated_energy_dictionary_error = noise_models_mitigated_energy_dictionary_error 

    ####################################################################################### 
    #### Mean of mitigation error for different noise models is computed and displayed  ###
    #######################################################################################


    noise_models_mitigated_energy_dictionary_error_statistics= mitigation_routines.compute_mitigation_error_median_mean(mitigation_data=mitigation_data, print_results=True)

    mitigation_data.noise_models_mitigated_energy_dictionary_error_statistics = noise_models_mitigated_energy_dictionary_error_statistics



    ####################################################################################### 
    #### Result of mitigation is saved                                                  ###
    #######################################################################################



    name_string = str(date.today()) + " q" + str(number_of_qubits) 

        

    file_name_mitigation_results  = 'DDOT_mitigation_results_workflow_' + name_string +'.pkl'






