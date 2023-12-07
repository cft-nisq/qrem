"""
Quantum Readout Error Mitigation (QREM) Probability Module
==========================================================

This module is part of the QREM package, designed for performing various operations related to 
the computation of probability vectors, stochastic matrices, marginal distributions etc. 
in quantum experiments. It includes among others functions for  validating probability vectors, generating 
random stochastic matrices, and computing marginal distributions from experimental results. 


Functions
---------
is_valid_probability_vector(examined_vector, threshold=1e-5)
    Validates whether a given list of numbers forms a valid probability vector.

find_closest_prob_vector_l1(quasiprobability_vector: List[float], method='picos')
    Find the closest probability vector to a quasiprobability vector using L1 norm.
    
find_closest_prob_vector_l2find_closest_prob_vector_l2(quasiprobability_vector: List[float])
    Find the closest probability vector to a given quasiprobability vector using Euclidean norm

random_stochastic_matrix(size, type='left', diagonal=1.0)
    Generates a random stochastic matrix of a specified type and size.

compute_marginals(results_dictionary, subsets_list, print_progress_bar=False, normalization=False)
    Computes marginal distributions for given experimental settings and marginal lists.

compute_marginals_multiprocessing(results_dictionary, subsets_list, multiprocessing=False, number_of_threads=None, print_progress_bar=False, normalization=False)
    Computes marginals, optionally using multiprocessing for performance.

compute_marginal_of_probability_distribution(global_probability_distribution, bits_of_interest, register_names=None)
    Returns the marginal distribution from a vector of global distribution.

calculate_total_variation_distance(p: np.array, q: np.array)
    alculate the total variation distance between two probability vectors.
"""
import copy
from typing import Dict,List, Tuple, Optional
import multiprocessing

from tqdm import tqdm
import numpy as np
import numpy.typing as npt

from qrem.common.printer import qprint, warprint
from qrem.common import convert

from scipy import sparse

from qrem.common import povmtools


def is_valid_probability_vector(examined_vector: List[float], threshold=1e-5) -> bool:
    """
    Check if a given vector is a valid probability vector.

    Parameters
    ----------
    examined_vector : List[float]
        A list of float numbers representing a vector to be examined.
    threshold : float, optional
        A small value to allow for numerical imprecision in summing to 1.

    Returns
    -------
    bool
        Returns True if the vector is a valid probability vector, False otherwise.
    """

    values_sum = 0

    for value in examined_vector:
        if value < 0:
            return False
        values_sum += value

    return abs(values_sum - 1) < threshold

def find_closest_prob_vector_l2(quasiprobability_vector: List[float]) -> np.ndarray:
    """
    Find the closest probability vector to a given quasiprobability vector using Euclidean norm.

    Parameters
    ----------
    quasiprobability_vector : List[float]
        A vector, possibly containing negative elements, whose elements sum to 1.

    Returns
    -------
    np.ndarray
        The closest probability vector in the Euclidean norm sense.

    Notes
    -----
    This function computes the probability vector closest to the input quasiprobability vector in terms of the
    Euclidean (L2) norm. It employs an algorithm suitable for diagonal quantum states, leveraging the equivalence
    of the 2-norm for diagonal matrices with the 2-norm of their diagonal elements - Ref. [4]. The method is essential in
    quantum information theory for approximating non-physical probability distributions with valid quantum states.
    """

    if isinstance(quasiprobability_vector, list):
        d = len(quasiprobability_vector)
    elif isinstance(quasiprobability_vector, type(np.array(0))):
        d = quasiprobability_vector.shape[0]

    # format vector properly
    quasiprobability_vector = np.array(quasiprobability_vector).reshape(d, 1)

    # copy part of the vector
    p000 = list(quasiprobability_vector[:, 0])

    # For algorithm to work we need to rearrange probability vector elements, so we need to keep track of their
    # initial ordering
    p1 = [[i, p000[i]] for i in range(d)]

    # Sort elements in descending order
    p1_sorted = sorted(p1, reverse=True, key=lambda x: x[1])

    # Initiate accumulator
    a = 0

    # go from the i_index=d
    for i in np.arange(0, d)[::-1]:

        # get vector element
        mu_i = p1_sorted[i][1]

        # if mu_i +a/(i_index+1) is negative, do the following:
        if mu_i + a / (i + 1) < 0:
            # add mu_i to accumulator
            a += mu_i

            # set vector element to 0
            p1_sorted[i][1] = 0
        # otherwise
        else:
            # update proper elements of probability vector
            for j in range(i + 1):
                p1_sorted[j][1] += a / (i + 1)
            # finish algorithm - everything is positive now
            break

    # return to initial order
    ordered_p = sorted(p1_sorted, key=lambda x: x[0])

    # get rid of indices
    p_good_format = [ordered_p[i][1] for i in range(d)]

    # print(ordered_p)

    return np.array(p_good_format).reshape(d, 1)

def find_closest_prob_vector_l1(quasiprobability_vector: List[float],
                                method='picos') -> np.ndarray:
    # FBM: perform speed tests between picos/scipy
    """
    Find the closest probability vector to a quasiprobability vector using L1 norm.

    Parameters
    ----------
    quasiprobability_vector : List[float]
        A vector with elements summing to 1 but may contain negative values.
    method : str, optional
        The optimization method to use ('picos' or 'scipy'). Defaults to 'picos'.

    Returns
    -------
    np.ndarray
        The closest probability vector in the L1 norm sense.

    Notes
    -----
    This function calculates the probability vector closest to a given quasiprobability vector, in terms of the L1 norm.
    It formulates this task as a linear programming problem, which can be solved using either the 'picos' or 'scipy'
    optimization libraries. This approach is crucial in quantum information theory for reconciling non-physical
    probability distributions with valid quantum states under different norm considerations.
    """

    if isinstance(quasiprobability_vector, list):
        dimension = len(quasiprobability_vector)
    elif isinstance(quasiprobability_vector, type(np.array(0))):
        dimension = quasiprobability_vector.shape[0]

    # format vector properly
    # TODO: we probably don't need it here
    quasiprobability_vector = np.array(quasiprobability_vector).reshape(dimension, 1)

    if method.lower() in ['picos']:
        problem_picos = picos.Problem()

        probability_vector = [picos.RealVariable(name=f'p_{index_outcome}',
                                                 shape=1,
                                                 lower=0,
                                                 upper=1
                                                 )
                              for index_outcome in range(dimension)]
        problem_picos.add_constraint(picos.sum(probability_vector) == 1)

        mus = [picos.RealVariable(name=f'mu_{index_outcome}',
                                  shape=1) for index_outcome in range(dimension)]

        cost_function = 0
        for index_outcome in range(dimension):
            eta_i = quasiprobability_vector[index_outcome] - probability_vector[index_outcome]
            mu_i = mus[index_outcome]

            cost_function += mu_i
            problem_picos.add_constraint(eta_i <= mu_i)
            problem_picos.add_constraint(-eta_i <= mu_i)

            # problem_picos.add_constraint(probability_vector[index_outcome] <= 1)
            # problem_picos.add_constraint(etas[index_outcome] <= mu_i)

        problem_picos.set_objective(direction='min',
                                    expression=cost_function)
        # print(problem_picos)
        problem_picos.solve()

        closest_probability_vector = np.array([problem_picos.get_variable(f'p_{index_outcome}').value
                                               for index_outcome in range(dimension)]).reshape(dimension, 1)
    elif method.lower() in ['scipy']:

        c_vector = np.array([0 for _ in range(dimension)] + [1 for _ in range(dimension)])

        equalities_vector = np.array(1)
        equalities_matrix = np.array([1 for _ in range(dimension)] +
                                     [0 for _ in range(dimension)]).reshape(1, 2 * dimension)

        # inequalities for absolute values
        inequalities_vector = np.zeros((2 * dimension))
        inequalities_vector[0:dimension] = -quasiprobability_vector[:, 0]
        inequalities_vector[dimension:] = quasiprobability_vector[:, 0]

        inequalities_matrix = np.zeros((2 * dimension, 2 * dimension))
        for outcome_index in range(dimension):
            inequalities_matrix[outcome_index, outcome_index] = -1
            inequalities_matrix[outcome_index, outcome_index + dimension] = -1

            inequalities_matrix[outcome_index + dimension, outcome_index] = 1
            inequalities_matrix[outcome_index + dimension, outcome_index + dimension] = -1

        bounds = [(0, 1)
                  # (-quasiprobability_vector[outcome_index],1-quasiprobability_vector[outcome_index])
                  for outcome_index in range(dimension)] + \
                 [(None, None) for _ in range(dimension)]
        print(bounds)
        res = scopt.linprog(c=c_vector,
                            b_eq=equalities_vector,
                            A_eq=equalities_matrix,
                            b_ub=inequalities_vector,
                            A_ub=inequalities_matrix,
                            bounds=bounds,
                            method='interior-point',
                            options={'maxiter': 10 ** 5,
                                     'disp': True}
                            )

        closest_probability_vector = np.array([res.x[outcome_index]
                                               for outcome_index in range(dimension)]).reshape(dimension, 1)

        # print(res.x)

    return closest_probability_vector

def calculate_total_variation_distance(p: np.array, q: np.array) -> float:
    """
    Calculate the total variation distance between two probability vectors. See Refs. [1] and [2] for the relation
    between TV-distance and operational distance between quantum measurements.

    The total variation distance is a measure of the statistical distance between two probability distributions.

    Parameters
    ----------
    p : np.ndarray
        The first probability vector.
    q : np.ndarray
        The second probability vector.

    Returns
    -------
    float
        The total variation distance between vectors p and q.
    """
    return np.linalg.norm(p - q, ord=1) / 2

def random_stochastic_matrix(size, type='left',diagonal = 1.0):
    """
    Generate a random stochastic matrix of a specified size and type.

    Parameters
    ----------
    size : int
        The size of the square matrix.
    type : str, optional
        Type of the stochastic matrix: 'left', 'right', or 'double'.
    diagonal : float, optional
        Value to add to the diagonal elements, defaults to 1.0.

    Returns
    -------
    numpy.ndarray
        A numpy array representing the generated stochastic matrix.
    """
    
    matrix = np.random.rand(size, size)+diagonal*np.identity(size)

    if (type == 'left'):
        matrix = matrix / matrix.sum(axis=0)[None, :]
    elif (type == 'right'):
        matrix = matrix / matrix.sum(axis=1)[:, None]
    elif (type == 'doubly' or type == 'double' or type == 'orto'):
        matrix = matrix = matrix / matrix.sum(axis=0)[None, :]
        matrix = matrix / matrix.sum(axis=1)[:, None]

    return matrix

def compute_average_marginal_for_subset(
    subset: Tuple,
    experiment_results: Dict[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]],
    normalized_marginals: Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]]
) -> Dict[str, npt.NDArray[np.float_]]:

    """
    For a given subset of qubits, compute its marginal with marginalizing (averaging) also over the input settings / circuit labels.
    
    Parameters
    ----------
    subset: Tuple
        A list of quibts for which the marginal is computed.
    results_dictionary : Dict[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]]
        A dictionary with experimental setting labels and results.
    counts_dictionary: Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]]
        A dictionary of precomputed unnormalized marginals (raw counts)

    Returns
    -------
    Dict[str, npt.NDArray[np.float_]]
        A dictionary of computed marginals. Keys are the circuit labels that appeared on the subset qubit indices. Value is a probability vector.
    """
    averaged_marginals = {}
    for circuit_label in experiment_results.keys():
        circuit_label_on_subset_indices = ''.join([circuit_label[x] for x in subset])
        
        if subset not in normalized_marginals[circuit_label].keys():
            # marginal not in the precomputed dictionary, need to copmute it
            res_dict = {circuit_label: experiment_results[circuit_label]}
            marginal_vector = compute_marginals_single(res_dict, [subset], normalization = True)[circuit_label][subset]
            warprint(f"Calculated marginals are missing given subset: {subset}, calculating and filling int")
        else:
            marginal_vector = normalized_marginals[circuit_label][subset]
        if circuit_label_on_subset_indices not in averaged_marginals.keys():
            averaged_marginals[circuit_label_on_subset_indices] = marginal_vector
        else:
            averaged_marginals[circuit_label_on_subset_indices] += marginal_vector
    for label_subset in averaged_marginals.keys():
        averaged_marginals[label_subset] /= np.sum(averaged_marginals[label_subset])
    return averaged_marginals

def compute_marginals_single(
    results_dictionary: Dict[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]], # npt.NDArray[bool], npt.NDArray[int]
    subsets_list: List[Tuple],
    print_progress_bar : bool = False,
    normalization : bool = False
) -> Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]]:
    """
    Compute marginal distributions for given experimental settings and marginals lists.

    Parameters
    ----------
    results_dictionary : Dict[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]]
        A dictionary with experimental setting labels and results.
    subsets_list subset: List[Tuple]
        A list of tuples indicating the marginals to be computed, e.g. [(0,1,2), (0,4,5), ...]
    print_progress_bar : bool, optional
        If True, shows a progress bar.
    normalization : bool, optional
        If True, normalizes the results to form a probability vector, otherwise raw counts.

    Returns
    -------
    Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]]
        A dictionary of computed marginals, in big endian qubits ordering.
    """

    items_iterable = results_dictionary.items()
    if print_progress_bar:
        items_iterable = tqdm(items_iterable)
    computed_marginals_dict = {}
    for experiment_label, experimental_results in items_iterable:
        single_exp_marginals = {}

        bitstring_results = experimental_results[0]
        counts = experimental_results[1]

        for marginal in subsets_list:
            b = bitstring_results[:, marginal]

            # TODO PP IMPORTANT Order of marginals indexing in the output is reversed, not sure why (probably due to the way the probabiltiy distribution is stored)
            groups = b.dot(2**np.arange(b.shape[1])[::-1])
            marginal_size = 2**len(marginal)

            # artificially append zero counts so that each possible marginal appears at least once
            data = np.append(counts, np.zeros(marginal_size))
            ids  = np.append(groups, np.arange(marginal_size))
            res = np.bincount(ids, weights=data)
            if normalization:
                res = res / res.sum()
            single_exp_marginals[marginal] = res
        computed_marginals_dict[experiment_label] = single_exp_marginals
    return computed_marginals_dict

def compute_marginals_mew(
    results_dictionary: Dict[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]], # npt.NDArray[bool], npt.NDArray[int]
    subsets_list: List[Tuple],
    print_progress_bar : bool = False,
    normalization : bool = False,
    multiprocessing : bool = False,
    number_of_threads = None) -> Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]]:
    """
    Compute marginals with optional multiprocessing support.

    Parameters
    ----------
    results_dictionary : Dict[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]]
        A dictionary with experimental setting labels and results.
    subsets_list : List[Tuple]
        A list of tuples indicating the marginals to be computed, e.g. [(0,1,2), (0,4,5), ...]
    print_progress_bar : bool, optional
        If True, shows a progress bar.
    normalization : bool, optional
        If True, normalizes the results to form a probability vector, otherwise raw counts.
    multiprocessing : bool, optionalDict
        A dictionary containing the computed marginals.
        If True, uses multiprocessing to improve performance.
    number_of_threads : Optional[int], optional
        The number of threads to use in multiprocessing.

    Returns
    -------
    Returns
    -------
    Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]]
        A dictionary of computed marginals, in big endian qubits ordering.
    """
    if multiprocessing:
        qprint("\nInitializing multithreaded calculation of marginals...")
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
            if thread_index in [number_of_threads-2]:
                print_progress_bar=True
            dictionary_res_now = {key : results_dictionary[key] for key in keys_slice_now}
            arguments_list.append((dictionary_res_now,
                                   subsets_list,
                                   print_progress_bar, normalization))
        pool = multiprocessing.Pool(number_of_threads)
        results = pool.starmap_async(compute_marginals_single, arguments_list)
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
        return compute_marginals_single(results_dictionary, subsets_list, print_progress_bar, normalization)


def normalize_marginals(
    marginals_dictionary: Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]], # npt.NDArray[bool], npt.NDArray[int]
    print_progress_bar : bool = False
) -> Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]]:
    """
    Normalizes marginal distributions for given marginals dictionary.

    Parameters
    ----------
    marginals_dictionary : Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]]
        A dictionary of unnormalized  marginals, in big endian qubits ordering.
    print_progress_bar : bool, optional
        If True, shows a progress bar.

    Returns
    -------
    Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]]
        A dictionary of normalized marginals, in big endian qubits ordering.
    """

    items_iterable = marginals_dictionary.items()
    if print_progress_bar:
        items_iterable = tqdm(items_iterable)
    for experiment_label, marginals_per_circuit in items_iterable:
        
        items_marginal = marginals_per_circuit.items()
        for marginal_qubits in items_marginal.keys():
            marginals_per_circuit[marginal_qubits] = marginals_per_circuit[marginal_qubits]/marginals_per_circuit[marginal_qubits].sum()
        marginals_dictionary[experiment_label] = marginals_per_circuit
    return marginals_dictionary

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
        normalization : bool = False):
    """
    Compute marginals with optional multiprocessing support.

    Parameters
    ----------
    results_dictionary : Dict[str, Dict[str, int]]
        A dictionary of experimental results.
    subsets_list : List[Tuple]
        List of subsets for which marginals are to be computed.
    use_multiprocessing : bool, optional
        If True, uses multiprocessing to improve performance.
    number_of_threads : Optional[int], optional
        The number of threads to use in multiprocessing.
    print_progress_bar : bool, optional
        If True, displays a progress bar.
    normalization : bool, optional
        If True, normalizes the results to form probability vectors.

    Returns
    -------
    Dict
        A dictionary containing the computed marginals.
    """
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
    """
    Compute the marginal of a probability distribution
    (Return marginal distribution from vector of global distribution).

    Parameters
    ----------
    global_probability_distribution : np.ndarray
        The global probability distribution as an array.
    bits_of_interest : List[int]
        Indices of bits for which the marginal distribution is computed (so we average over other bits)
        Assuming that qubits are labeled
        from 0 to log2(len(global_probability_distribution)).
    register_names : Optional[List[str]], optional
        Bitstring register names, defaults to a standard binary representation: 
        '00...00', '000...01', '000...10', ..., etc.

    Returns
    -------
    np.ndarray
        The marginal probability distribution array.

    Notes
    -----
    This function identifies bits with qubits in the variable bitstring_names.
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
        bitstring_names = povmtools.get_classical_register_bitstrings(list(range(global_number_of_qubits)),
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
