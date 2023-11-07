
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import statistics
from qrem.functions_qrem import ancillary_functions as anf
from qrem.functions_qrem import functions_hamiltonians
from qrem.functions_qrem import functions_data_analysis as fdt

from qrem.backends_support.qiskit import qiskit_utilities
from qrem.noise_characterization.base_classes.OverlappingTomographyBase import OverlappingTomographyBase
from qrem.noise_characterization.tomography_design.overlapping.DOTMarginalsAnalyzer import DOTMarginalsAnalyzer
from qrem.noise_characterization.tomography_design.overlapping.SeparableCircuitsCreator import \
    SeparableCircuitsCreator
#MOcomm - it seams we are using two MarginalAnalizer classes to compute the same thing
from qrem.noise_characterization.tomography_design.overlapping.QDTMarginalsAnalyzer import \
    QDTMarginalsAnalyzer

from qrem.noise_mitigation.probability_distributions.MarginalsCorrector import MarginalsCorrector
from qrem.noise_simulation.CN import functions_sampling as fus
from qrem.noise_simulation.CN.noise_implementation import  simulate_noise_results_dictionary


from qrem.common import probability, math
from qrem.common.printer import qprint

import qrem.functions_qrem.povmtools as pt
from qrem import cn





#this is mostly


#function generating 2sat hamilonians for benchmarks
def create_hamiltonians_for_benchmarks(number_of_qubits, number_of_hamiltonians, clause_density):
    hamiltonian_name = '2SAT'
    hamiltonian_ordering_indices = list(range(0, number_of_hamiltonians))
    index_first, index_last = hamiltonian_ordering_indices[0], hamiltonian_ordering_indices[-1]
    # whether to attempt to solve Hamiltonians. WARNING: for big problems might be infeasible
    solve_hamiltonians = False
    all_hamiltonians = {}

    for hamiltonian_index in hamiltonian_ordering_indices:

        hamiltonian_data_now = functions_hamiltonians.generate_random_2SAT_hamiltonian(
            number_of_qubits=number_of_qubits,
            clause_density=clause_density)

        if solve_hamiltonians:
            hamiltonian_data_now = functions_hamiltonians.solve_2SAT_hamiltonian(hamiltonian_data=hamiltonian_data_now)

        all_hamiltonians[hamiltonian_index] = hamiltonian_data_now
    return all_hamiltonians





def create_eigenstates_benchmarks(number_of_qubits, number_of_states, noise_model):
    OT_creator = OverlappingTomographyBase(number_of_qubits=number_of_qubits,
                                           experiment_name='DDOT',
                                           maximal_circuits_number=number_of_states)

    circuits_QDOT = OT_creator.get_random_circuits_list(number_of_circuits=number_of_states)

    experiment_name = 'DDOT'
    circuits_creator = SeparableCircuitsCreator(SDK_name='qiskit',
                                                qubit_indices=range(number_of_qubits),
                                                quantum_register_size=5,
                                                classical_register_size=5,
                                                descriptions_of_circuits=circuits_QDOT,
                                                experiment_name=experiment_name)

    OT_circuits_list = circuits_creator.get_circuits()
    batches = anf.create_batches(circuits_list=OT_circuits_list, circuits_per_job=300)
    jobs_list = qiskit_utilities.run_batches(backend_name='qasm_simulator',
                                             batches=batches,
                                             shots=10 ** 4)

    unprocessed_results = common.providers.ibmutils.data_converters.get_counts_from_qiskit_jobs(jobs_list=jobs_list)

    processed_results = fdt.convert_counts_overlapping_tomography(counts_dictionary=unprocessed_results,
                                                                  experiment_name=experiment_name,
                                                                  reverse_bitstrings=True)
    noisy_results_dictionary = simulate_noise_results_dictionary(processed_results, noise_model[0], noise_model[1])

    subsets_of_qubits = subsets_of_qubits = math.get_k_local_subsets(number_of_elements=5,subset_size=2, all_sizes_up_to_k = True)
#MOcomm - we use here a very "low level class" - can't we use basic classes like marginals_analyzer_base? Janek?
    marginals_analyzer = QDTMarginalsAnalyzer(results_dictionary=noisy_results_dictionary,
                                              experiment_name=experiment_name)

    marginals_analyzer.compute_all_marginals(subsets_dictionary=subsets_of_qubits, show_progress_bar=True,
                                             multiprocessing=False)

    return noisy_results_dictionary, marginals_analyzer.marginals_dictionary


def eigenstate_energy_calculation_and_estimation(results_dictionary,marginals_dictionary, hamiltonians_data):
    results_energy_estimation = {}

    for state_index, input_state in tqdm(enumerate(marginals_dictionary.keys())):
        # Read Hamiltonian data
        hamiltonian_data_dictionary = hamiltonians_data[state_index]
        weights_dictionary = hamiltonian_data_dictionary['weights_dictionary']

        # Calculate ideal energy
        energy_ideal = fdt.get_energy_from_bitstring_diagonal_local(bitstring=input_state,
                                                                    weights_dictionary=weights_dictionary)

        # Read experimental results data
        marginals_dictionary_raw = marginals_dictionary[input_state]
        results_dictionary_raw = results_dictionary[input_state]

        missing_subsets = [x for x in weights_dictionary.keys() if x not in marginals_dictionary_raw.keys()]
        if len(missing_subsets) > 0:
            marginals_analyzer = DOTMarginalsAnalyzer({input_state: results_dictionary_raw})
            marginals_analyzer.compute_all_marginals(missing_subsets,
                                                     show_progress_bar=False,
                                                     multiprocessing=False)

            marginals_dictionary_raw = {**marginals_dictionary_raw,
                                        **marginals_analyzer.marginals_dictionary[input_state]}

        # Calculate experimentally estimated energy
        energy_raw = fdt.estimate_energy_from_marginals(weights_dictionary=weights_dictionary,
                                                        marginals_dictionary=marginals_dictionary_raw)

        dictionary_results_now = {'energy_ideal': energy_ideal,
                                  'energy_raw': energy_raw,
                                  'input_state': input_state,
                                  'weights_dictionary': weights_dictionary,
                                  'hamiltonian_index': state_index,
                                  }

        results_energy_estimation[state_index] = dictionary_results_now

    return results_energy_estimation

#JT: argument updated 
def run_benchmarks(number_of_qubits, results_dictionary, marginals_dictionary, energy_estimation_results, hamiltonians_data,
                   all_clusters_sets_dictionary, correction_matrices_dictionary, correction_indices_dictionary,mitigation_data_dictionary,noise_matrices_dictionary):
    noisy_energy_prediction = False
    noise_mitigation = True

    mitigation_method = 'lambda_inverse'
    method_kwargs = {'ensure_physicality': False}

    results_benchmarks = {state_index: {'energies': {}, 'errors': {}}
                          for state_index, input_state in enumerate(marginals_dictionary.keys())}

    
    #JT: This is a temporary addition, a dictionary that allows to test mitigation in an easier way, to be deleated

    #JT: Added for test purposes
    mitigated_marginals_dictionary_debugging={}

    
    for state_index, input_state in tqdm(enumerate(marginals_dictionary.keys())):
        #     print(f"{state_index}")

        
        energy_ideal, energy_raw = energy_estimation_results[state_index]['energy_ideal'], \
                                   energy_estimation_results[state_index]['energy_raw']

        energies_dictionary_now = {'energy_ideal': energy_ideal,
                                   'energy_raw': energy_raw}

        error_raw_now = abs(energy_ideal - energy_raw) / number_of_qubits

        errors_dictionary_now = {'error_raw': error_raw_now}

        # Read Hamiltonian data
        hamiltonian_data_dictionary = hamiltonians_data[state_index]
        weights_dictionary = hamiltonian_data_dictionary['weights_dictionary']
        needed_pairs = [x for x in weights_dictionary.keys() if len(x) == 2]

        # Read experimental results data
        marginals_dictionary_raw = marginals_dictionary[input_state]
        results_dictionary_raw = results_dictionary[input_state]

        missing_subsets = [x for x in weights_dictionary.keys() if x not in marginals_dictionary_raw.keys()]
        if len(missing_subsets) > 0:
            marginals_analyzer = DOTMarginalsAnalyzer({input_state: results_dictionary_raw})
            marginals_analyzer.compute_all_marginals(missing_subsets,
                                                     show_progress_bar=False,
                                                     multiprocessing=False)

            marginals_dictionary_raw = {**marginals_dictionary_raw,
                                        **marginals_analyzer.marginals_dictionary[input_state]}

        if noisy_energy_prediction:
            energies_dictionary_now['predicted_energies'] = {}
            errors_dictionary_now['energy_prediction_errors'] = {}

            for clusters_assignment, __ in all_clusters_sets_dictionary.items():
                noise_matrices_dictionary_model = {cluster: noise_matrices_dictionary[cluster]['averaged']
                                                   for cluster in clusters_assignment}

                predicted_energy = fus.get_noisy_energy_product_noise_model(input_state=input_state,
                                                                            noise_matrices_dictionary=noise_matrices_dictionary_model,
                                                                            needed_pairs=needed_pairs,
                                                                            weights_dictionary_tuples=weights_dictionary)

                energies_dictionary_now['predicted_energies'][clusters_assignment] = predicted_energy

                error_prediction = abs(predicted_energy - energy_raw) / number_of_qubits
                errors_dictionary_now['energy_prediction_errors'][clusters_assignment] = error_prediction

        if noise_mitigation:
            energies_dictionary_now['corrected_energies'] = {}
            errors_dictionary_now['corrected_errors'] = {}
            
            #the line below is added 
            mitigated_marginals_dictionary_debugging_now={}

            for clusters_assignment in all_clusters_sets_dictionary.keys():

                #marginals_dictionary_raw = marginals_dictionary[input_state]
                correction_indices_now = correction_indices_dictionary[clusters_assignment]
                correction_indices_now = mitigation_data_dictionary[clusters_assignment][0]
                correction_matrices_dictionary_new = mitigation_data_dictionary[clusters_assignment][1] 

                noise_matrices_dictionary_model = {cluster: noise_matrices_dictionary[cluster]['averaged']
                                                   for cluster in clusters_assignment}
                
                #JT: This needs to be changed
                correction_data_now = {'correction_indices': correction_indices_now,
                                       'correction_matrices_old': correction_matrices_dictionary,
                                       'correction_matrices': correction_matrices_dictionary_new, 
                                       'noise_matrices': noise_matrices_dictionary_model}

                missing_subsets = []

                for needed_subset in weights_dictionary.keys():
                    subset_for_correction = correction_indices_now[needed_subset]
             
                    if subset_for_correction not in marginals_dictionary_raw.keys():
                        missing_subsets.append(subset_for_correction)

                if len(missing_subsets) > 0:
                    marginals_analyzer = DOTMarginalsAnalyzer({input_state: results_dictionary_raw})
                    marginals_analyzer.compute_all_marginals(missing_subsets,
                                                             show_progress_bar=False,
                                                             multiprocessing=False)

                    marginals_dictionary_raw = {**marginals_dictionary_raw,
                                                **marginals_analyzer.marginals_dictionary[input_state]}

                marginals_corrector = MarginalsCorrector(
                    experimental_results_dictionary={input_state: results_dictionary_raw},
                    correction_data_dictionary=correction_data_now,
                    marginals_dictionary=marginals_dictionary_raw)

                marginals_dictionary_to_correct = {correction_indices_now[subset_to_correct]:
                                                       marginals_dictionary_raw[
                                                           correction_indices_now[subset_to_correct]]
                                                   for subset_to_correct in weights_dictionary.keys()}

                marginals_corrector.correct_marginals(marginals_dictionary=marginals_dictionary_to_correct,
                                                      method_kwargs=method_kwargs,
                                                      method=mitigation_method
                                                      )

                # Coarse-grain some of the corrected marginals_dictionary to obtain the ones that appear in Hamiltonian
                marginals_coarse_grained_corrected = \
                    marginals_corrector.compute_marginals_of_marginals(
                        weights_dictionary.keys(),
                        corrected=True)

                for key, marg in marginals_coarse_grained_corrected.items():
                    if not probability.is_valid_probability_vector(marg):
                        marginals_coarse_grained_corrected[key] = pt.find_closest_prob_vector_l2(marg).flatten()


                #the line below is added for debugging purposes
                mitigated_marginals_dictionary_debugging_now[clusters_assignment] = marginals_coarse_grained_corrected 
                
                energy_corrected = fdt.estimate_energy_from_marginals(weights_dictionary=weights_dictionary,
                                                       marginals_dictionary=
                                                       marginals_coarse_grained_corrected)
                #params_tuple = tuple(params)
               # print(type(clusters_assignment_data))
               # clusters_assignment_data = tuple([clusters_assignment,params_tuple])
                clusters_assignment_data = clusters_assignment
                energies_dictionary_now['corrected_energies'][clusters_assignment_data ] = energy_corrected

                error_correction = abs(energy_corrected - energy_ideal) / number_of_qubits
                errors_dictionary_now['corrected_errors'][clusters_assignment_data]= error_correction

        results_benchmarks[state_index]['energies'] = energies_dictionary_now
        results_benchmarks[state_index]['errors'] = errors_dictionary_now

        #JT: This is added for test purposes
        mitigated_marginals_dictionary_debugging[input_state]=mitigated_marginals_dictionary_debugging_now

    error_dictionary ={}
    for state_index in results_benchmarks.keys():
        for cluster_assigment,error in  results_benchmarks[state_index]['errors']['corrected_errors'].items():
            new_entry = {state_index:error}
            if cluster_assigment in error_dictionary.keys():
                error_dictionary[cluster_assigment] = {**error_dictionary[cluster_assigment],**new_entry}
            else:
                error_dictionary[cluster_assigment] =  new_entry
    
    for cluster_assigment, state_errors in error_dictionary.items():
        error_dictionary[cluster_assigment] = {**state_errors, **{'median': statistics.median(list(state_errors.values())), 'mean': statistics.mean(list(state_errors.values()))}}
        

     

      



    qprint("ALL DONE!", '', 'red')
    return results_benchmarks, error_dictionary#,   mitigated_marginals_dictionary_debugging


def calculate_mitigated_energy_for_hamiltonian(benchmark_hamiltonian, mitigated_marginals_dictionary):
    weights_dictionary = benchmark_hamiltonian['weights_dictionary']
    mitigated_energy = fdt.estimate_energy_from_marginals(weights_dictionary=weights_dictionary,
                                                       marginals_dictionary=
                                                       mitigated_marginals_dictionary)
    return mitigated_energy

def run_mitigation_benchmark_for_hamiltonian(benchmark_hamiltonian, results_dictionary,noise_model):
    weights_dictionary = benchmark_hamiltonian['weights_dictionary']
    state= list(results_dictionary.keys())[0]
    mitigated_marginals_dictionary={}
    for subset in weights_dictionary.keys():
        mitigated_marginals_dictionary[subset]=cn.mitigation.mitigate_marginal(marginal=subset,results_dictionary=results_dictionary,noise_model=noise_model)
    

        


def run_mitigation_benchmarks_for_hamiltonians(benchmark_hamiltonian_list, joint_mitigated_marginals_dictionary, noise_model):
    mitigated_energies_dictionary = {}
    for benchmark_hamiltonian, mitigated_marginals_dictionary in zip(benchmark_hamiltonian_list, joint_mitigated_marginals_dictionary):
        
        mitigated_energies_dictionary[mitigated_marginals_dictionary.key()] = calculate_mitigated_energy_for_hamiltonian(benchmark_hamiltonian=benchmark_hamiltonian,mitigated_marginals_dictionary=mitigated_marginals_dictionary) 
    return mitigated_energies_dictionary 
        






#divides indices of benchmark hamiltonians into traning and stest set
#outputs two lists: traning_set and test_ set with indices of respective hamiltonians
def create_traning_and_test_set(number_of_hamiltonians, traning_set_cardinality):
    traning_set=[]
    test_set =[]
    while (len(traning_set))<traning_set_cardinality:
        index=np.random.randint(number_of_hamiltonians)
        if index not in traning_set:
            traning_set.append(index)
    for i in range(number_of_hamiltonians):
        if i not in traning_set:
            test_set.append(i)
    return traning_set, test_set


# function calculates mean and median of mitigation benchmarks over a test/traning set
# input: benchmarks_results_mitigation - dictionary with mitigation results, traning_set - indicies of hamiltonians that form the traning set


def calculate_results_test_set(benchmark_results_mitigation, hamiltonian_set):
    # loop over differnt noise models (keys of benchmark_results_mitigation dictionary) and benchmark results for individual hamiltonians
    traning_median_mean = {}
    for noise_model, result in benchmark_results_mitigation.items():
        median, mean = calculate_median_mean(result['errors_list'], hamiltonian_set)
        traning_median_mean[noise_model] = {'median': median, 'mean': mean}
    return traning_median_mean


# helepr function that calculates median and mean for a single noise model
def calculate_median_mean(errors_dictionary, hamiltonian_set):
    median_list = []
    mean = 0.0
    for index in hamiltonian_set:
        error = errors_dictionary[index]
        mean = mean + error
        median_list.append(error)
    mean = mean / len(hamiltonian_set)
    return np.median(median_list), mean


#creates list of data to be plotted as a alpha vs benchmark result plot
#input clusters_dictionary (dictionary with clusters and details), and mitigation_data: list returned by calculate_results_test_set function
#outputs three lists 2d lists: plot_alpha_list - list of alpha (clustering parameter) for different localities
# plot_median_list - list with medians for different alphas/ localities
# plot_mean_list  - list with means different alphas/ localities
#the first index in the list encodes locality e.g. alpha_list[1] corresponds to list of alphas for clustering algorithm with locality =2
#corresponding values of medians for locality=2 are found in plot_median_list[1]
#this data is later used to create plots
def cerate_data_alpha_plot(clusters_dictionary, mitigation_data):
    parameters_dictionary = {}
    # loop over cluster assigments from clustering algorithm
    for cluster, clustering_params in clusters_dictionary.items():
        # this dictionary will store localities and corresponding alphas for a given cluster as well as median and mean for clustering
        locality_dic = {}
        max_locality = 0

        if clustering_params != None:
            # this is a loop over different configurations of localities and alpha for a given cluster structure
            for item in clustering_params:
                # data is rewritten as dictionary of a form locality : list of alphas
                if item[0] > max_locality:
                    max_locality = item[0]
                if item[0] in locality_dic.keys():
                    temp_list = locality_dic[item[0]]
                    temp_list.append(item[1])
                    locality_dic[item[0]] = temp_list

                else:

                    locality_dic[item[0]] = [item[1]]

        else:
            locality_dic[1] = [1]

        # to a given cluster value of meadian and mean of mitigation benchmark is added
        if 'median' in mitigation_data[cluster].keys():
            locality_dic['meadian'] = mitigation_data[cluster]['median']
            locality_dic['mean'] = mitigation_data[cluster]['mean']
        else:
            locality_dic['meadian'] = statistics.median(list(mitigation_data[cluster].values()))
            locality_dic['mean'] = statistics.mean(list(mitigation_data[cluster].values()))
        parameters_dictionary[cluster] = locality_dic
        # print(parameters_dictionary[cluster])

    plot_alpha_list = [[] for i in range(2, max_locality + 2)]
    plot_median_list = [[] for i in range(2, max_locality + 2)]
    plot_mean_list = [[] for i in range(2, max_locality + 2)]

    for clusters, parameters in parameters_dictionary.items():
        for i in range(1, max_locality + 1):
            if i in parameters.keys():
                for item in parameters[i]:
                    plot_alpha_list[i - 1].append(item)
                    plot_median_list[i - 1].append(parameters['meadian'])
                    plot_mean_list[i - 1].append(parameters['mean'])

    return plot_alpha_list, plot_median_list, plot_mean_list

#this function creates matplotlib plots for data created with cerate_data_alpha_plot
#plot_x_list is usually a list of alphas
#plot_y_list is a list of medians/means
#separable_value is a value for uncorelated noise model
#label is the plot description and name of the file where results are saved
def create_plots(plot_x_list, plot_y_list, separable_value, label,y_min=0,y_max=0.5):
    legend = []
    for i in range(len(plot_x_list)):

        # this if exludes plotting results for uncorelated noise model, they are added as a line below
        if len(plot_x_list[i]) > 1:
            plt.plot(plot_x_list[i], plot_y_list[i], marker='o', linestyle='none')
            title = label
            legend.append(f'Locality {i + 1}')
            min_val = min(plot_y_list[i])
            if min_val < separable_value:
                min_x = plot_x_list[i][plot_y_list[i].index(min_val)]
                plt.annotate(str(np.round(min_val, 5)), xy=(min_x, min_val))

            plt.title(title)
            fname = title + ".png"
    plt.ylim(y_min, y_max)

    legend.append("Uncorrelated")

    plt.plot([plot_x_list[1][0], plot_x_list[1][-1]], [separable_value, separable_value])
    plt.annotate(str(np.round(separable_value, 5)), xy=(2.5, separable_value))
    plt.legend(legend)
    plt.xlabel("alpha")
    plt.savefig(fname)


