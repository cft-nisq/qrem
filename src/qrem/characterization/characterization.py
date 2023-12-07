"""
refactored functions with new marginals format
"""
from typing import Dict, Tuple, List, Union
from pathlib import Path

import time
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from qrem.common import io, probability, printer, math as qmath

def compute_all_marginals(   results_dictionary: Dict[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]],
                                    subsets_of_qubits:List[Tuple],
                                    backup: Union[bool,str,Path]=False,
                                    overwrite: bool = False,
                                    verbose_log: bool = False):
        
    """
    #FBM: add multiprocessing for this task
    #FBM: change subsets_list to be dictionary
    Implements self.compute_marginals for all experimental keys.

    :param subsets_dictionary: list of subsets of qubits for which marginals_dictionary should be calculated
    :param show_progress_bar: if True, shows progress bar. Requires "tqdm" package
    """


    t0 = time.time()
    #[1] Compute normalized marginals for each experimental setting (each circuit label)
    marginals_dictionary = probability.compute_marginals( results_dictionary=results_dictionary,
                                                        subsets_list=subsets_of_qubits,
                                                        print_progress_bar = True,
                                                        normalization=True,
                                                        multiprocessing= True) 
    

    t1 = time.time()

    #[2] Output printing and backups
    if verbose_log:
        printer.qprint("\nCalculating marginals took:",t1-t0)


    if backup:
        path_to_backup =  io.prepare_outfile(   outpath= backup,
                                                overwrite= overwrite,
                                                default_filename = "marginals_dictionary.pkl")
        
        io.save(dictionary_to_save=marginals_dictionary, directory=path_to_backup.parent, custom_filename=path_to_backup.name, overwrite = overwrite)    
    
        if verbose_log:
            printer.qprint("\nBacked up the calculated marginals dictionary into: ",path_to_backup)

    return marginals_dictionary



def averaged_marginals_to_noise_matrix_ddot(
        averaged_marginals: Dict[str, np.ndarray]) -> np.ndarray:
    """Return noise matrix from counts dictionary.
    Assuming that the results are given only for qubits of interest.
    :param results_dictionary: dictionary with experiments of the form:

    results_dictionary[input_state_bitstring] = probability_distribution

    where:
    - input_state_bitstring is bitstring denoting classical input state
    - probability_distribution - estimated vector of probabilities for that input state

    :return: noise_matrix: the array representing noise on qubits
    on which the experiments were performed
    """

    # Register length
    number_of_qubits_of_interest = len(list(averaged_marginals.keys())[0])
    
    noise_matrix = np.zeros((2 ** number_of_qubits_of_interest, 2 ** number_of_qubits_of_interest))

    for input_state, probability_vector in averaged_marginals.items():
        numbers_input = [int(x) for x in list(input_state)]

        #Bad data check - qubits should be in state 0 or 1 for DDoT
        if np.any(np.array(numbers_input) > 1):
            raise ValueError("Input state should be given as a bitstring of 0s and 1s for DDoT")

        try:
            noise_matrix[:, int(input_state, 2)] = probability_vector[:]
        except(ValueError):
            noise_matrix[:, int(input_state, 2)] = probability_vector[:, 0]



    return noise_matrix

def compute_single_noise_matrix_ddot(   experiment_results:Dict[str,Dict[str,int]],
                                        noramlized_marginals:Dict[str,Dict[str,int]] ,
                                        subset: Tuple,
                                        fill_missing_columns_with_ideal_result = True,
                                        verbose_log: bool = False) -> Dict[str, np.ndarray]:
    """Noise matrix for subset of qubits, averaged over all other qubits

        :param subset: subset of qubits we are interested in

        By default takes data from self._marginals_dictionary. If data is not present, then it
        calculates marginals_dictionary for given subset
        and updates the class's property self.marginals_dictionary
    """

    averaged_marginals = probability.compute_average_marginal_for_subset(subset=subset,
                                                                            experiment_results=experiment_results, 
                                                                            noramlized_marginals = noramlized_marginals)
    
    noise_matrix_averaged = averaged_marginals_to_noise_matrix_ddot(averaged_marginals)
    
    
    if not qmath.is_matrix_stochastic(noise_matrix_averaged):
        # qprint_array(noise_matrix_averaged)
        message_now = f"\nNoise matrix not stochastic for subset: {subset}.\n" \
                        f"This most likely means that DDOT collection was not complete " \
                        f"for locality {len(subset)} and some " \
                        f"states were not implemented."
        
        if verbose_log:
            printer.warprint(message_now)
            printer.warprint("\nThat matrix looks like this:")
            printer.warprint(noise_matrix_averaged)

        if fill_missing_columns_with_ideal_result:
            printer.warprint("\nAdding missing columns.")
            for column_index in range(noise_matrix_averaged.shape[0]):
                column_now = noise_matrix_averaged[:, column_index]

                if np.all(column_now == 0):
                    noise_matrix_averaged[column_index, column_index] = 1.0

            printer.warprint("\nNow the matrix looks like this:")
            printer.warprint(noise_matrix_averaged)

        else:
            printer.warprint("\nNoise matrix not stochastic and missing collumns not filled in:")
    return noise_matrix_averaged

def compute_noise_matrices_ddot( experiment_results:Dict[str,Dict[str,int]],
                                noramlized_marginals:Dict[str,Dict[str,int]] ,
                                subset_of_qubits: List[Tuple] =[],
                                backup: Union[bool,str,Path]=False,
                                overwrite: bool = False,
                                verbose_log: bool = False) -> Dict[Tuple, Dict[str, np.ndarray]]:

    noise_matrices_dictionary = {}

    #Loop over all subsets of qubits. Can be multiprocessed in future if needed
    for subset in tqdm(subset_of_qubits):
        

        noise_matrix_averaged = compute_single_noise_matrix_ddot(   experiment_results=experiment_results,
                                                                    noramlized_marginals = noramlized_marginals,
                                                                    subset=subset,
                                                                    fill_missing_columns_with_ideal_result = True,
                                                                    verbose_log=verbose_log)
        
        noise_matrices_dictionary[subset] = {'averaged': noise_matrix_averaged}

    return noise_matrices_dictionary