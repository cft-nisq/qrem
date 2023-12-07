from qrem.visualisation.plotting_constants import color_map
from matplotlib import pyplot as plt
from qrem.mitigation.mitigation_routines import compute_mitigation_error_median_mean, compute_mitigation_errors

import numpy as np

from typing import List, Dict

def create_coherence_bound_histogram_data(coherence_bound_dictionary):
    coherence_bound_data_list = []
    for key in coherence_bound_dictionary.keys():
        coherence_bound_data=coherence_bound_dictionary[key][0]
        for data in coherence_bound_data:
            
            coherence_bound_data_list.append(data)
    return coherence_bound_data_list


def create_POVMs_distance_histogram(POVMs_errors_ibm, POVMs_errors_rig, path_to_save = None):
    
    single_qubit_errors_ibm = [POVMs_errors_ibm['worst_case']['classical'][(qubit,)] for qubit in range(127) ]

    single_qubit_errors_rig = [POVMs_errors_rig['worst_case']['classical'][(qubit,)] for qubit in range(79) ]



    
    color_1 = color_map(0.45)#0.45
    color_2 = color_map(0.99)
    color_3 = color_map(0.1) 
    
    
    w_max = 1

    w_min = 0.00

        
    w = (w_max)/1000

    bins_new=np.arange(w_min, w_max + w, w) 
    bins_new=np.append(bins_new,np.inf)
    plt.hist(single_qubit_errors_ibm,bins=bins_new,cumulative=True,density=True,histtype='step',alpha=0.5,color = color_1,linewidth=1.5)
    plt.hist(single_qubit_errors_rig,bins=bins_new,cumulative=True,density=True,histtype='step',alpha=0.5,color = color_2,linewidth=1.5)

    labels= ["IBM Cusco","Rigetti Aspen-M-3"]
    plt.legend(labels)
    plt.title("One-qubit POVMs distance to projective meaurements  ")
    plt.xlim((-0.01,1.01))
    plt.ylim((0.0,1.1))
    if path_to_save:
        plt.savefig(path_to_save)
    else:
        plt.savefig()
    plt.clf()


def create_correlations_distance_histogram(correlations_coefficients_matrix_ibm, correlations_coefficients_matrix_rig, path_to_save=None):



    color_1 = color_map(0.45)#0.45
    color_2 = color_map(0.99)
    color_3 = color_map(0.1) 
    
    w_max = max(0, max(correlations_coefficients_matrix_rig.flatten()))

    w_min = 0.00

        
    w = (w_max)/15000

    bins_new=np.arange(w_min, w_max + w, w) 
    bins_new=np.append(bins_new,np.inf)



    plt.hist(correlations_coefficients_matrix_ibm.flatten(),bins=bins_new,cumulative=True,density=True,histtype='step',alpha=0.5,color = color_1,linewidth=1.5)
    
    plt.hist(correlations_coefficients_matrix_rig.flatten(),bins=bins_new,cumulative=True,density=True,histtype='step',alpha=0.5,color = color_2,linewidth=1.5)

    labels= ["IBM Cusco","Rigetti Aspen-M-3"]
    plt.legend(labels)
    plt.title("Cumulative histogram of correlation coefficients")
    plt.xlim((-0.01,0.41))
    plt.ylim((0.95,1.005))
   
    if path_to_save:
        plt.savefig(path_to_save)
    else:
        plt.savefig()
    plt.clf()

def create_coherence_bound_histogram(coherence_bound_dictionary_ibm:Dict, coherence_bound_dictionary_rigetti:Dict, path_to_save=None):

    ibm_data =  create_coherence_bound_histogram_data( coherence_bound_dictionary_ibm)

    rigetti_data = create_coherence_bound_histogram_data(coherence_bound_dictionary_rigetti)

    
    color_rigetti = color_map(0.45)#0.45
    color_ibm = color_map(0.99)
    w = 0.001
    bins_new=np.arange(0.03, 0.1+ w, w) 
    plt.xlim((0.03,0.1))
    plt.ylim((0,42))

    entries_ibm, edges_ibm, _= plt.hist(ibm_data,density=True,color=color_ibm,alpha=0.5,bins=bins_new)
    entries_rig, edges_rig, _= plt.hist(rigetti_data,density=False,color=color_rigetti,alpha=0.5,bins=bins_new)
    labels= ["IBM Cusco","Rigetti Aspen-M-3"]

    plt.legend(labels)
    plt.title("Histogram of bound on Coherence Strength" )
    if path_to_save:
        plt.savefig(path_to_save)
    else:
        plt.savefig()
    plt.clf()

def create_error_mitigation_prediction_histogram(noise_models_mitigated_energy_dictionary:Dict,noise_models_predicted_energy_dictionary:Dict,energy_dictionary:Dict,number_of_qubits:int, plot_title ='', path_to_save =None):
    
    noise_models_mitigated_energy_dictionary_error= compute_mitigation_errors(noise_models_mitigated_energy_dictionary['corrected_energy'],hamiltonian_energy_dictionary=energy_dictionary,number_of_qubits=number_of_qubits)
    
    benchmark_results_mean_median_dictionary = compute_mitigation_error_median_mean(mitigation_errors_dictionary=noise_models_mitigated_energy_dictionary_error,print_results=False)


    #cluster assignment with minimal median is chosen
    minimal_median_cluster_assignment=min(benchmark_results_mean_median_dictionary['median'],key = benchmark_results_mean_median_dictionary['median'].get)

    #product noise model data are red from mitigation results
    product_model_error_list = list(noise_models_mitigated_energy_dictionary_error[next(iter(noise_models_mitigated_energy_dictionary_error.keys()))].values())

    #best (i.e. with minimal median) noise model data are red from mitigation results 
    best_cn_model_error_list = list(noise_models_mitigated_energy_dictionary_error[minimal_median_cluster_assignment].values())

    #plot colors are set
    color_product_model = color_map(0.45)#0.45
    color_cn_best_model = color_map(0.99)
    raw_error_color = color_map(0.1) 

    #raw error computation
    raw_errors_list = []
    error_prediction_best_cn =[]
    error_tpm = []  
    tpm_clusters = tuple(next(iter(noise_models_predicted_energy_dictionary["predicted_energy"].keys() ))) 
    for key, hamiltonian_energy_dictionary in energy_dictionary.items():
        raw_errors_list.append(abs(hamiltonian_energy_dictionary['energy_raw']-hamiltonian_energy_dictionary['energy_ideal'])/number_of_qubits)
        error_prediction_best_cn.append(abs(hamiltonian_energy_dictionary['energy_raw']- noise_models_predicted_energy_dictionary["predicted_energy"][minimal_median_cluster_assignment][key]["predicted_energy"])/number_of_qubits)
        error_tpm.append(abs(hamiltonian_energy_dictionary['energy_raw']- noise_models_predicted_energy_dictionary["predicted_energy"][tpm_clusters][key]["predicted_energy"])/number_of_qubits)

    
    #histogram is drawn 
    color_product_model = color_map(0.45)#0.45
    color_cn_best_model = color_map(0.99)
    raw_error_color = color_map(0.1) 
    w = (max(raw_errors_list) -min(best_cn_model_error_list))/60
    bins_new=np.arange(min(best_cn_model_error_list), max(raw_errors_list) + w, w) 
    plt.hist(product_model_error_list,bins=bins_new,alpha=0.5,color = color_product_model)
    plt.hist(best_cn_model_error_list ,bins=bins_new,alpha=0.5, color = color_cn_best_model)
    plt.hist(raw_errors_list , bins=bins_new,alpha=0.5, color = raw_error_color )
    plt.ylim(0, 80)
    labels= ["TPM","CN model locality " +str(len(max(minimal_median_cluster_assignment, key=len))), "$\Delta E_{EST}(H)$"]
    plt.legend(labels)
    plt.title("Histogram of $\Delta E_{MIT}(H)$ " + plot_title)

  

    if path_to_save:
        plt.savefig(path_to_save+"error_mitigation_histogram")
    else:
        plt.savefig()
    plt.clf()

       #histogram is drawn 
    color_product_model = color_map(0.45)#0.45
    color_cn_best_model = color_map(0.99)
    raw_error_color = color_map(0.1) 
    w = (max(raw_errors_list) -min(best_cn_model_error_list))/60
    bins_new=np.arange(min(best_cn_model_error_list), max(raw_errors_list) + w, w) 
    plt.hist(error_tpm,bins=bins_new,alpha=0.5,color = color_product_model)
    plt.hist(error_prediction_best_cn ,bins=bins_new,alpha=0.5, color = color_cn_best_model)
    plt.ylim(0, 80)
    labels= ["TPM","CN model locality " +str(len(max(minimal_median_cluster_assignment, key=len))), "$\Delta E_{EST}(H)$"]
    plt.legend(labels)
    plt.title("Histogram of $\Delta E_{PRED}(H)$ " + plot_title)

  

    if path_to_save:
        plt.savefig(path_to_save+"error_prediction_histogram")
    else:
        plt.savefig()
    plt.clf()


