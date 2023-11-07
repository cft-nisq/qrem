#MOcomm - do we need and use this file??
#ORGANIZE - delate/ move?
import pickle

from tqdm import tqdm
import numpy as np

from qrem.functions_qrem import ancillary_functions as anf
from qrem.functions_qrem import functions_data_analysis as fda
from qrem.functions_qrem import functions_standarized_directories as dirs

from qrem.noise_simulation.CN import sampling_improvements as si

from qrem.common import io
from qrem.common.printer import qprint

# specify data used for testing
backend_name = 'ASPEN-9'
date = '2021-09-27'
number_of_qubits = 20
bitstrings_right_to_left = False


backend_name = 'ibm_washington'
date = '2022-04-10'
number_of_qubits = 109



# Specify whether save calculated data
saving = True

running_estimation = False


noise_model_name = "uncorrelated"

noise_model_name = "only_clusters"

alpha_multiplier = 1.0

if noise_model_name.upper()=='ONLY_CLUSTERS':
    no_neighbors = True
    noise_model_name = f"{noise_model_name}_alpha-{alpha_multiplier}"



directory_merged_noise_models_SCR= dirs.get_directory_processed_experimental_results(backend_name=backend_name,
                                                                                                  date=date,
                                                                                                  number_of_qubits=number_of_qubits,
                                                                                                  experiment_name='noise_mitigation',
                                                                                                  additional_path="noise_models_data/merged_models/SCR/")

folder_name_hamiltonians = 'ground_states_implementation/2SAT/'


directory_stored_hamiltonian = dirs.get_directory_stored_hamiltonians(number_of_qubits=number_of_qubits,
                                                                      hamiltonian_name='2SAT')
with open(directory_stored_hamiltonian + "full_information_2SAT_N20.pkl", 'rb') as filein:
    hamiltonians_data = pickle.load(filein)

hamiltonian_weights_all = {input_state:
                           ham_data['']['weights_dictionary']
                           for input_state, ham_data in hamiltonians_data.items()}


directory_noise_mitigation= dirs.get_directory_processed_experimental_results(backend_name=backend_name,
                                                                                      date=date,
                                                                                      number_of_qubits=number_of_qubits,
                                                                                      experiment_name='noise_mitigation')



directory_noise_characterization = dirs.get_directory_processed_experimental_results(backend_name=backend_name,
                                                                                        date=date,
                                                                                        number_of_qubits=number_of_qubits,
                                                                                        experiment_name='noise_characterization')

directory_energy_estimation_results = dirs.get_directory_processed_experimental_results(backend_name=backend_name,
                                                                                        date=date,
                                                                                        number_of_qubits=number_of_qubits,
                                                                                        experiment_name='ground_states_implementation/2SAT')
with open(directory_energy_estimation_results + "energy_estimations_counts.pkl", 'rb') as filein:
    experiments_results = pickle.load(filein)

results_dictionary = experiments_results['results_dictionary']

experiment_keys = list(results_dictionary.keys())

with open(directory_energy_estimation_results + 'estimated_energies_results.pkl', 'rb') as filein:
    estimated_energies = pickle.load(filein)

with open(directory_noise_mitigation + f'correction_data_merged_{noise_model_name}.pkl', 'rb') as filein:
    dictionary_mitigation_data = pickle.load(filein)


dictionary_results_noise_matrices = dictionary_mitigation_data['joint_noise_matrices_dictionary']



joint_correction_matrices_dictionary = dictionary_mitigation_data[
    'joint_correction_matrices_dictionary']
# noise_models_correction_indices = dictionary_mitigation_data['correction_indices']
joint_noise_matrices_dictionary = dictionary_mitigation_data['joint_noise_matrices_dictionary']

# raise KeyboardInterrupt
do_sanity_checks = False

test_cl = [(x,) for x in range(number_of_qubits)]

all_pairs = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]

with open(directory_merged_noise_models_SCR + f'noise_models_merged_data_SCR_{noise_model_name}.pkl', 'rb') as filein:
    dictionary_data_noise_models = pickle.load(filein)


number_of_samples = 10**6

#TODO FBM: fix those indices, something seems to be wrong.

with open(directory_noise_characterization + 'noise_model_identifiers2.pkl', 'rb') as filein:
    noise_model_identifiers = pickle.load(filein)


run_index = 0
qprint("\nRUN INDEX:",run_index,'red')
range_run= slice(run_index*2,(run_index+1)*2)
print(range_run)





# print(len(list(noise_model_identifiers.keys())))
# print(len(list(dictionary_data_noise_models.items())))
#
# print(list(noise_model_identifiers.keys())[0])
for clusters_key, noise_model_data in list(dictionary_data_noise_models.items())[range_run]:
    clusters_list = list(clusters_key)
    local_noise_matrices = noise_model_data['local_noise_matrices']

    noise_parameters = noise_model_data['noise_model_parameters']

    noise_parameters = sorted(noise_parameters, key=lambda x:x[0]+x[2])
    noise_parameters = tuple(noise_parameters)
    # print(noise_parameters)
    #
    # raise KeyboardInterrupt

    noise_model_identifier_now = noise_model_identifiers[noise_parameters]


    qprint("\nCurrent noise model", clusters_list)
    print(noise_model_data.keys())

    # SCR_dict_now = noise_model_data['SCR_dictionary']

    symmetrizers_dictionary = noise_model_data['symmetrizers_dictionary']
    error_probabilities_dictionary = noise_model_data['error_probabilitities_dictionary']

    print(error_probabilities_dictionary)

    qubits_to_clusters_map = {}
    for qubit in range(number_of_qubits):
        for cluster in clusters_list:
            if qubit in cluster:
                qubits_to_clusters_map[qubit] = cluster



    multipliers_mitigation = {}
    for i in range(number_of_qubits):
        for j in range(i+1,number_of_qubits):
            multiplier = 1
            for qi in (i,j):
                multiplier/=(1 - 2 * error_probabilities_dictionary[(qi,)])
            multipliers_mitigation[(i,j)] = multiplier
    for i in range(number_of_qubits):
        multipliers_mitigation[(i,)] = 1/(1 - 2 * error_probabilities_dictionary[(i,)])


    results_dictionary_models = {}
    figures_of_merit_list = []
    for input_state in tqdm(experiment_keys):
        energy_estimated = estimated_energies[input_state]
        hamiltonian_dict = hamiltonians_data[input_state]['']
        weights_dictionary = hamiltonian_dict['weights_dictionary']
        needed_marginals = list(weights_dictionary.keys())
        energy_ideal = hamiltonian_dict['ground_state_energy']


        results_now_dict = results_dictionary[input_state]
        unique_counts = list(results_now_dict.keys())
        probability_distro = np.array(list(results_now_dict.values()),dtype=float)
        numer_of_shots_in_experiments = sum(probability_distro)
        probability_distro/=numer_of_shots_in_experiments
        if number_of_samples==sum(list(results_now_dict.values())):
            resampled_outcomes_dictionary = results_now_dict
        else:
            resampled_outcomes = np.random.multinomial(pvals=probability_distro,
                                                       n=number_of_samples)


            resampled_outcomes_dictionary = {unique_counts[outcome_index]:resampled_outcomes[outcome_index]
                                             for outcome_index in range(len(resampled_outcomes))}


        equalized_samples = si.sample_from_multiple_input_states_alternative(input_states_dictionary=resampled_outcomes_dictionary,
                                                                             local_stochastic_matrices=symmetrizers_dictionary,
                                                                             print_progres_bar=False,
                                                                             # old_sampling=True
                                                                 )

        # energy_corrected = fda.estimate_energy_from_counts_dictionary(counts_dictionary=equalized_samples,
        #                                                               weights_dictionary=weights_dictionary,
        #                                                               additional_multipliers=multipliers_mitigation)
        # print(equalized_samples)
        energy_corrected = fda.estimate_energy_from_counts_dictionary_alternative(counts_dictionary=equalized_samples,
                                                                      weights_dictionary=weights_dictionary,
                                                                      additional_multipliers=multipliers_mitigation,
                                                                                  )

        # t3 = time.time()

        # qprint("TOTAL ENERGY ESTIMATION TIME:",t3-t2)

        error_corrected = abs(energy_corrected-energy_ideal)/number_of_qubits
        error_estimated = abs(energy_estimated-energy_ideal)/number_of_qubits
        figure_of_merit = error_corrected/error_estimated
        figures_of_merit_list.append(figure_of_merit)
        # print(energy_corrected==energy_corrected1)
        # print(energy_corrected-energy_corrected2)

        #
        # qprint("Ratio newmethod/oldmethod:",(dt0/dt1))
        # qprint("number of samples",number_of_samples)



        # raise KeyboardInterrupt

        # qprint("Energy ideal:",energy_ideal)
        qprint("\nEnergy estimated error per qubit:",np.round(error_estimated,3))
        # qprint("Energy corrected:", energy_corrected)
        qprint("Energy corrected error per qubit:", np.round(error_corrected,3))
        qprint("Figure of merit", np.round(figure_of_merit,3))
        print()

        # print(energy_ideal,
        #       energy_estimated,
        #       energy_corrected,
        #       abs(energy_corrected - energy_ideal) / number_of_qubits
        #       # en_noisy
        #       )

        # raise KeyboardInterrupt


        results_dictionary_models[input_state] = {
            'error_corrected': error_corrected,
            'error_estimated': error_estimated,
            'energy_corrected': energy_corrected,
            'energy_estimated': energy_estimated,
            'energy_ideal': energy_ideal,

        }

    qprint("Mean figure of merit:", np.mean(figures_of_merit_list))
    io.save(dictionary_to_save=results_dictionary_models,
                            directory=directory_energy_estimation_results + f'/noise_mitigated/SCR/particular_models/s_{number_of_samples}/',
                            custom_filename="id_" + str(noise_model_identifier_now))


