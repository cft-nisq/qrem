
#MOcomm This looks like outdated file - do we need it?
#ORGANIZE - probably this file should be delated

import pickle

from tqdm import tqdm
import numpy as np

from qrem.functions_qrem import ancillary_functions as anf
from qrem.functions_qrem import functions_data_analysis as fda
from qrem.noise_characterization.base_classes.marginals_analyzer_base import MarginalsAnalyzerBase
from qrem.noise_simulation.CN import sampling_improvements as si

from qrem.common.printer import qprint

# specify data used for testing
backend_name = 'ASPEN-9'
date = '2021-09-27'
number_of_qubits = 20
bitstrings_right_to_left = False

# Specify whether save calculated data
saving = True

running_estimation = False

module_directory = anf.get_local_storage_directory()
tests_directory_main = module_directory + '/saved_data/characterization2021/'
tests_directory_low = f"ground_states_implementation/2SAT/{date}/N{number_of_qubits}/"
directory_to_open = tests_directory_main + 'processed_data/' + tests_directory_low

with open(directory_to_open + "00_raw_results.pkl", 'rb') as filein:
    experiments_results = pickle.load(filein)

results_dictionary = experiments_results['results_dictionary']

with open(directory_to_open + "full_information_2SAT_N20.pkl", 'rb') as filein:
    hamiltonians_data = pickle.load(filein)

# results_dictionary = experiments_results['results_dictionary']


tests_directory_low = f"{date}/{backend_name}/N{number_of_qubits}/"
directory_to_open = tests_directory_main + 'processed_data/' + tests_directory_low
with open(directory_to_open + 'correction_data_tests.pkl', 'rb') as filein:
    dictionary_data_noise_models = pickle.load(filein)

joint_correction_matrices_dictionary = dictionary_data_noise_models[
    'joint_correction_matrices_dictionary']
noise_models_correction_indices = dictionary_data_noise_models['correction_indices']
joint_noise_matrices_dictionary = dictionary_data_noise_models['joint_noise_matrices_dictionary']

# print(noise_models_correction_indices[0])
total_number = 0
new_noise_model_correction_indices = []
# already_done_models_heh = []
all_models = {tup[0]: tup for tup in noise_models_correction_indices}
merged_noise_models = {}
already_done_models_heh = []
mapping_for_oskar = {}
for model_parameters, stuff in all_models.items():
    if model_parameters in already_done_models_heh:
        continue

    list_now = [model_parameters]
    already_done_models_heh.append(model_parameters)
    for model_parameters2, stuff2 in all_models.items():
        if model_parameters2 in already_done_models_heh:
            continue

        if model_parameters != model_parameters2:
            if stuff2[1] == stuff[1]:
                list_now.append(model_parameters2)
                already_done_models_heh.append(model_parameters2)

    merged_noise_models[tuple(list_now)] = (stuff[1], stuff[2])

noise_models_correction_indices = [(key, value[0], value[1]) for key, value in
                                   merged_noise_models.items()]

# anf.save_results_pickle(dictionary_to_save=noise_models_correction_indices,
#                         directory=directory_to_open,
#                         custom_name='sorted_clusters_data')


results_big_dictionary = {model_parameters: {} for model_parameters, _, _ in
                          noise_models_correction_indices}

all_pairs = [(i, j) for i in range(number_of_qubits) for j in
             range(i + 1, number_of_qubits)]

all_pairs = all_pairs + [(i,) for i in range(number_of_qubits)]

if running_estimation:

    # estimated_results = {}
    estimated_energies = {}
    estimated_marginals = {}
    for input_state in tqdm(list(results_dictionary.keys())):
        hamiltonian_dict = hamiltonians_data[input_state]['']
        weights_dictionary = hamiltonian_dict['weights_dictionary']
        needed_marginals = list(weights_dictionary.keys())
        # energy_ideal = hamiltonian_dict['ground_state_energy']

        results_now = results_dictionary[input_state]
        marginals_calculator = MarginalsAnalyzerBase(
            results_dictionary={input_state: results_now},
            bitstrings_right_to_left=bitstrings_right_to_left
        )

        marginals_calculator.compute_unnormalized_marginals(experiment_keys=[input_state],
                                                            subsets_dictionary=all_pairs)

        marginals_dictionary = marginals_calculator.marginals_dictionary[input_state]
        energy_estimated = fda.estimate_energy_from_marginals(
            weights_dictionary=weights_dictionary,
            marginals_dictionary=marginals_dictionary)

        estimated_energies[input_state] = energy_estimated
        estimated_marginals[input_state] = marginals_dictionary

    anf.save_results_pickle(dictionary_to_save=estimated_energies,
                            directory=directory_to_open,
                            custom_name='estimation_energy_results_all')

    anf.save_results_pickle(dictionary_to_save=estimated_marginals,
                            directory=directory_to_open,
                            custom_name='estimation_marginals_results_all')

else:
    with open(directory_to_open + 'estimation_energy_results_all.pkl', 'rb') as filein:
        estimated_energies = pickle.load(filein)

    with open(directory_to_open + 'estimation_marginals_results_all.pkl', 'rb') as filein:
        estimated_marginals = pickle.load(filein)

# raise KeyboardInterrupt
do_sanity_checks = False

test_cl = [(x,) for x in range(number_of_qubits)]

# print(register)
#
all_pairs = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]


# all_pairs = all_pairs + [(i,) for i in range(number_of_qubits)]


def bitstring_swap_wise(bistring, substring, bits):
    # t000=time.time()
    count_bad = [c for c in bistring]
    for b in range(len(substring)):
        count_bad[bits[b]] = substring[b]

    count_bad = count_bad

    returning_this = ''.join([c for c in count_bad])
    # qprint('swap wise took:',time.time()-t000)

    return returning_this


def create_marginal_bitstring(bitstring, bits_of_interest):
    # TODO: make sure about it
    # print(bitstring)
    reversed_order = bitstring
    return ''.join([reversed_order[b] for b in bits_of_interest])


def stochastic_noise_to_count(count,
                              bits,
                              mask,
                              error_matrix):
    random_mask = np.random.choice(mask, p=
    error_matrix[:, int(
        create_marginal_bitstring(count, bits), 2)])

    return bitstring_swap_wise(count, random_mask, bits)


def multiple_stochastic_noise_to_count(
        count,
        set_of_bits,
        masks,
        error_matrices):
    # assuming lack of correlations between error matrices
    new_string = count
    for i in range(len(set_of_bits)):
        new_string = stochastic_noise_to_count(new_string,
                                               set_of_bits[i],
                                               masks[i],
                                               error_matrices[i])

    return new_string


def get_multiple_marginal_bitstrings(count, set_of_bits):
    return [create_marginal_bitstring(count, bs) for bs in set_of_bits]


def create_masks(bits, size):
    return anf.register_names_qubits(bits, size, False)


def make_memory_noisy(memory, bits_to_flip, error_matrices):
    # N=len(memory)
    masks = [create_masks(range(len(bs)), len(bs)) for bs in bits_to_flip]

    return [multiple_stochastic_noise_to_count(count,
                                               bits_to_flip,
                                               masks,
                                               error_matrices)
            for count in memory]


from qrem.noise_mitigation.expected_values.CRS import SCR_functions as scr_fun

run_symmetrization_finders = False

if run_symmetrization_finders:
    symmetrizers_all = {}
    for model_parameters, noise_model_dictionary, _ in tqdm(noise_models_correction_indices[0:]):

        clusters_list = noise_model_dictionary['clusters_list']
        local_noise_matrices = {cluster: joint_noise_matrices_dictionary[cluster]['averaged'] for cluster
                                in clusters_list}

        qprint("\nNoise model now:",clusters_list)

        error_probabilitities_dictionary = {}
        symmetrizers_dictionary = {}

        for cluster in clusters_list:
            local_map = local_noise_matrices[cluster]
            if len(cluster)==1:
                equalizer = scr_fun.find_general_symmetrizer(stochastic_matrix=local_map
                                                             )
                equalized_matrix = equalizer@local_map
                error_probability = equalized_matrix[0,1]
                error_probabilitities_dictionary[cluster] = error_probability
            else:
                equalizer, errors_probabilities = scr_fun.find_optimal_decoupler(stochastic_matrix=local_map,
                                                                                 printing=False)
                for index_qubit in range(len(errors_probabilities)):
                    error_probabilitities_dictionary[(cluster[index_qubit],)] = errors_probabilities[index_qubit]

            symmetrizers_dictionary[cluster] = equalizer
        # numpy.set_printoptions(threshold=sys.maxsize)

        qprint("\nOptimal error probabilities:", error_probabilitities_dictionary)

        symmetrizers_all[model_parameters] = {'symmetrizers_dictionary':symmetrizers_dictionary,
                                              'error_probabilities':error_probabilitities_dictionary}

        # qprint("DONE!\n")
        print("\n")
        # qprint("error probabilities:")

    anf.save_results_pickle(dictionary_to_save=symmetrizers_all,
                            directory=directory_to_open,
                            custom_name='symmetrizers_dictionaries')

#
#
else:
    with open(directory_to_open + 'symmetrizers_dictionaries.pkl', 'rb') as filein:
        symmetrizers_all = pickle.load(filein)
#
from typing import List, Tuple

def sort_clusters_division(clusters_division:List[Tuple[int]]):

    sorted_inside = [tuple(sorted(x)) for x in clusters_division]

    return tuple(sorted(sorted_inside, key = lambda x: x[0]))



unique_correction_indices = {}
unique_cluster_divisions = list(np.unique([sort_clusters_division(noise_model_dictionary['clusters_list'])
                                           for _, noise_model_dictionary, __ in noise_models_correction_indices]))
# for model_parameters, noise_model_dictionary, _ in noise_models_correction_indices:

model_parameters_dict_unique_clusters = {cluster_division:[] for cluster_division in unique_cluster_divisions}

models_dict = {model_parameters:noise_model_dictionary for model_parameters, noise_model_dictionary, _ in noise_models_correction_indices}


for model_parameters, noise_model_dictionary, _ in noise_models_correction_indices:
    clusters_list = sort_clusters_division(noise_model_dictionary['clusters_list'])
    model_parameters_dict_unique_clusters[clusters_list].append(model_parameters)

original_model_parameters = {}
model_parameters_dict_unique_clusters_formatted_correctly = {}
for cluster_division, model_parameters in model_parameters_dict_unique_clusters.items():
    one_list = tuple([p for params in model_parameters for p in params])
    model_parameters_dict_unique_clusters_formatted_correctly[cluster_division] = (one_list, models_dict[model_parameters[0]])

    original_model_parameters[one_list] = model_parameters[0]

number_of_samples = 10 ** 5
test_08, test_18, test_78, test_01 = 0, 0, 0, 0

print(len(unique_cluster_divisions))


# noise_model_identifiers = {}
# counter = 0
# for clusters_list, (model_parameters, noise_model_dictionary) \
#         in list(model_parameters_dict_unique_clusters_formatted_correctly.items())[0:]:
#
#     noise_model_identifiers[model_parameters] = counter
#
#     counter+=1

#
# anf.save_results_pickle(dictionary_to_save=noise_model_identifiers,
#                         directory=directory_to_open,
#                         custom_name='noise_model_identifiers_updated')

with open(directory_to_open + 'noise_model_identifiers_updated.pkl', 'rb') as filein:
    noise_model_identifiers = pickle.load(filein)

run_index = 0
qprint("\nRUN INDEX:",run_index,'red')
print()
# for clusters_list, model_parameters, noise_model_dictionary in model_parameters_dict_unique_clusters_formatted_correctly.items()
for clusters_list, (model_parameters, noise_model_dictionary) \
        in list(model_parameters_dict_unique_clusters_formatted_correctly.items())[run_index*3:(run_index+1)*3]:


    noise_model_identifier_now = noise_model_identifiers[model_parameters]


    # clusters_list = noise_model_dictionary['clusters_list']
    qprint("\nCurrent noise model", clusters_list)
    print(model_parameters)
    # raise KeyboardInterrupt
    # print(clusters_list==_sort_clusters_division(noise_model_dictionary['clusters_list']))
    # print([len(x) for x in clusters_list])
    # clusters_list = [(2*j,2*j+1) for j in range(10)]
    # clusters_list = [(j,) for j in range(number_of_qubits)]
    # raise KeyboardInterrupt
    qubits_to_clusters_map = {}
    for qubit in range(number_of_qubits):
        for cluster in clusters_list:
            if qubit in cluster:
                qubits_to_clusters_map[qubit] = cluster


    local_noise_matrices = {cluster: joint_noise_matrices_dictionary[cluster]['averaged'] for cluster
                            in clusters_list}

    params_key = original_model_parameters[model_parameters]

    symmetrizers_dictionary = symmetrizers_all[params_key]['symmetrizers_dictionary']
    error_probabilitities_dictionary = symmetrizers_all[params_key]['error_probabilities']
    # raise KeyboardInterrupt

    range_keys = list(results_dictionary.keys())

    # TODO added starting from different place

    results_dictionary_models = {}
    figures_of_merit_list = []
    for input_state in tqdm(range_keys[0:]):
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

        resampled_outcomes = np.random.multinomial(pvals=probability_distro,n=number_of_samples)


        resampled_outcomes_dictionary = {unique_counts[outcome_index]:resampled_outcomes[outcome_index]
                                         for outcome_index in range(len(resampled_outcomes))}

        equalized_samples = si.sample_from_multiple_input_states(input_states_dictionary=resampled_outcomes_dictionary,
                                                                 local_stochastic_matrices=symmetrizers_dictionary,
                                                                 print_progres_bar=False)


        energy_corrected = 0
        for subset in needed_marginals:
            local_energy = fda.estimate_energy_from_counts_dictionary(counts_dictionary=equalized_samples,
                                                                      weights_dictionary={subset:weights_dictionary[subset]})
            multiplier = 1

            for qi in subset:
                multiplier/=(1 - 2 * error_probabilitities_dictionary[(qi,)])

            energy_corrected+=multiplier*local_energy

        error_corrected = abs(energy_corrected-energy_ideal)/number_of_qubits
        error_estimated = abs(energy_estimated-energy_ideal)/number_of_qubits
        figure_of_merit = error_corrected/error_estimated
        figures_of_merit_list.append(figure_of_merit)

        # # qprint("Energy ideal:",energy_ideal)
        # qprint("\nEnergy estimated error per qubit:",np.round(error_estimated,3))
        # # qprint("Energy corrected:", energy_corrected)
        # qprint("Energy corrected error per qubit:", np.round(error_corrected,3))
        # qprint("Figure of merit", np.round(figure_of_merit,3))
        # print()

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
    # raise KeyboardInterrupt

    if saving:
        anf.save_results_pickle(dictionary_to_save=results_dictionary_models,
                                directory=directory_to_open + '/SCR_models_mitigation/s_10^6/',
                                custom_name="id_" + str(noise_model_identifier_now))


