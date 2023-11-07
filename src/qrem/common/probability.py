import copy
from typing import Dict,List, Tuple, Optional
import multiprocessing

from tqdm import tqdm
import numpy as np

from qrem.functions_qrem import functions_data_analysis as fda, ancillary_functions as anf

from qrem.common.printer import qprint
from qrem.common import convert


def is_valid_probability_vector(examined_vector: List[float], threshold=1e-5) -> bool:
    """Checks if given vector is valid probability vector i_index.e. has only positive values that sums to 1.

    Parameters
    ----------
    examined_vector: List[float]
        Vector of probabilities for which the validity is checked.
    threshold: float
        Error cluster_threshold when determining probabilities sum condition.

    Returns
    -------
        Bool information whether examined_vector is valid probability vector or not.
    """
    values_sum = 0

    for value in examined_vector:
        if value < 0:
            return False
        values_sum += value

    return abs(values_sum - 1) < threshold


def random_stochastic_matrix(size, type='left',diagonal = 1.0):
    """Get a random stochastic matrix of given type ( and size."""
    matrix = np.random.rand(size, size)+diagonal*np.identity(size)

    if (type == 'left'):
        matrix = matrix / matrix.sum(axis=0)[None, :]
    elif (type == 'right'):
        matrix = matrix / matrix.sum(axis=1)[:, None]
    elif (type == 'doubly' or type == 'double' or type == 'orto'):
        matrix = matrix = matrix / matrix.sum(axis=0)[None, :]
        matrix = matrix / matrix.sum(axis=1)[:, None]

    return matrix

# ===================================================
# Basic function used to compute marginals from experimental counts - probability distributions
# ===================================================

#JT: Function used to compute marginals
#MOcomm: this function doubles functionalities of some of 
def _compute_marginals(
        results_dictionary: Dict[str, Dict[str, int]],
        subsets_list: List[Tuple],
        print_progress_bar : bool =False,
        normalization : bool = False
):

    items_iterable = results_dictionary.items()

    if print_progress_bar:
        items_iterable = tqdm(items_iterable)

    marginals_dictionary = {}

    #JT: this for loop goes over experimental settings and results of experiments for a given setting

    for experiment_label, experimental_results in items_iterable:

               #JT: copy creates a shallow copy of an object (changes in original object influence copy)

        #subsets_list_temp = subsets_list[experiment_label]

        if isinstance(subsets_list, dict):
            subsets_list = subsets_list[experiment_label]
        
           #JT: the variable below stores unique lengths of qubits subsets

        unique_lengths = np.unique([len(x) for x in subsets_list])

        #JT: a nested dictionary
        #length -elements of unique_lengths list
        #index outcome - goes over all bit strings of length from unique_lengths list
        #index outcome is converted to a bitstring, if this is shorter than length it is filled with 0 to the left, this is key of the second dictionary
        #the value of the second dictionary is set to 0

        local_registers_dictionaries = {length:
                                            {convert.integer_to_bitstring(index_outcome,
                                                                    length):
                                                0.0
                                            for index_outcome in range(int(2 ** length))}
                                        for length in unique_lengths}
        
        vectors_dictionaries = {subset: copy.copy(local_registers_dictionaries[len(subset)]) for
                                subset in subsets_list}

        # vectors_list = [copy.copy(local_registers_dictionaries[len(subset)]) for
        #                         subset in subsets_list]
        # norm = 0


   

        #JT: this is a loop over different subsets

        for subset_index in range(len(subsets_list)):

            #JT: consecutive subsets are chosen

            subset = subsets_list[subset_index]

            #JT: For a fixed experimental setting a loop over different

            for outcome_bitstring, number_of_occurrences in experimental_results.items():

                #JT: outcome bitstring for a given subset is established

                tuple_now = ''.join([outcome_bitstring[qi] for qi in subset])

                #JT: otcome statistics for this subset and outcome string are added

                vectors_dictionaries[subset][tuple_now] += number_of_occurrences

        # vectors_dictionaries = {subsets_list[subset_index]:vectors_list[subset_index]
        #                         for subset_index in range(len(subsets_list))}


            # norm += number_of_occurrences

       #JT the dictionary has a structure key: subset value, value an array with number of occurrences
       #JT: I'm not sure, how the ordering of results id done, i.e. what is the relation between a place in array and the resulting bitstring
       #JT: I assume that 0 etry is all 0 result and so on, but this needs to be double-checked

        marginals_dict_for_this_experiment = {
            subset: np.array(list(vectors_dictionaries[subset].values())) for subset in
            subsets_list}
        marginals_dictionary[experiment_label] = marginals_dict_for_this_experiment

    return marginals_dictionary


# ===================================================
# This function is used in marginals_analyzer_base
# ===================================================

def compute_marginals(results_dictionary: Dict[str, Dict[str, int]],
        subsets_list:List[Tuple],
        use_multiprocessing : bool =False,
        number_of_threads=None,

        print_progress_bar : bool =False,
        normalization : bool = False
                                   ):
    if use_multiprocessing:
        if number_of_threads is None:
            number_of_threads = multiprocessing.cpu_count()-1

        experimental_keys = list(results_dictionary.keys())
        number_of_experiments = len(experimental_keys)

        batch_size = -int(number_of_experiments//(-number_of_threads))

        arguments_list = []
        for thread_index in range(number_of_threads):
            slice_indices_now = slice(thread_index * batch_size,
                            (thread_index + 1) * batch_size)

            print_progress_bar = False

            keys_slice_now = experimental_keys[slice_indices_now]


            #that's the last pool thread with maximal number of function calls
            # (last thread has the least of them)
            if thread_index in[number_of_threads-2]:
                print_progress_bar=True

            dictionary_res_now = {key:results_dictionary[key] for key in keys_slice_now}

            arguments_list.append((dictionary_res_now,
                                   subsets_list,
                                   print_progress_bar))
     


        pool = multiprocessing.Pool(number_of_threads)
        results = pool.starmap_async(_compute_marginals,
                                 arguments_list
                                 )
    

        pool.close()
        qprint("\nJoining pool...")
        pool.join()
        qprint("\nGetting results from pool...")

        res_multiprocessing = results.get()



        all_results = {}

        for dictionary_thread in res_multiprocessing:
            all_results = {**all_results, **dictionary_thread}

        return all_results
    else:
        return _compute_marginals(results_dictionary=results_dictionary,subsets_list=subsets_list)
# ===================================================
# Functions used to marginalize probability distributions
# ===================================================

def compute_marginal_of_probability_distribution(
        global_probability_distribution: np.ndarray,
        bits_of_interest: List[int],
        register_names: Optional[List[str]] = None) -> np.ndarray:
    """Return marginal distribution from vector of global distribution
    :param global_probability_distribution: distribution on all bits
    :param bits_of_interest: bits we are interested in (so we average over other bits)
                            Assuming that qubits are labeled
                            from 0 to log2(len(global_probability_distribution))
    :param register_names: bitstrings register, default is
                           '00...00', '000...01', '000...10', ..., etc.

    :return: marginal_distribution : marginal probability distribution

    NOTE: we identify bits with qubits in the variables bitstring_names

    #TODO FBM: do some speed tests on some details of those solutions
    """

    if len(bits_of_interest) == 0:
        print('0 length bits list')
        return global_probability_distribution

    try:
        if isinstance(global_probability_distribution[0][0], complex) or isinstance(
                global_probability_distribution[0][0], np.complex128):
            global_probability_distribution = global_probability_distribution.real
    except(IndexError):
        if isinstance(global_probability_distribution[0], complex) or isinstance(
                global_probability_distribution[0], np.complex128):
            global_probability_distribution = global_probability_distribution.real

    global_dimension = len(global_probability_distribution)
    global_number_of_qubits = int(np.log2(global_dimension))
    all_qubits = list(range(global_number_of_qubits))
    bits_to_average_over = list(set(all_qubits).difference(set(bits_of_interest)))

    number_of_bits_in_marginal = global_number_of_qubits - len(bits_to_average_over)
    dimension_of_marginal = 2 ** number_of_bits_in_marginal

    if register_names is None:
        bitstring_names = anf.get_classical_register_bitstrings(list(range(global_number_of_qubits)),
                                                                global_number_of_qubits)
    else:
        bitstring_names = register_names

    marginal_distribution = np.zeros((dimension_of_marginal, 1), dtype=float)
    for j in range(global_dimension):
        # this is slightly faster than going through "for bitstring_global in bitstring_names
        # and then converting bitstring_global to integer
        # and also faster than creating the global bitstring in situ
        bitstring_global = bitstring_names[j]

        bitstring_local = ''.join(
            [bitstring_global[qubit_index] for qubit_index in bits_of_interest])

        marginal_distribution[int(bitstring_local, 2)] += global_probability_distribution[j]

    return marginal_distribution