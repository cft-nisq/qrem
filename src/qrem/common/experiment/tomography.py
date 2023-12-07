"""
Quantum Circuit Tomography
==========================

This module provides functions for generating quantum circuits. 
It is primarily used for preparing characterisation experimental setup for QREM package, 
including the creation of random circuits and the computation of necessary circuit counts 
based on various parameters. Functions include the ability to compute the optimal number 
of circuits, create random circuits, check the completeness of circuit sets, generate 
circuits with specific characteristics, and batch circuits for processing.

Functions
---------
    compute_number_of_circuits(experiment_type, number_of_qubits, subset_locality, ...):
        Computes the optimal number of circuits for experiment preparation.

    create_random_circuits(experiment_type, number_of_qubits, number_of_circuits):
        Creates a list of random circuits for experiment preparation.

    check_completeness(experiment_type, circuit_list, locality, symbols_list, subsets_list):
        Checks the completeness of a given set of circuits.

    generate_circuits(number_of_qubits, experiment_type, k_locality, ...):
        Generates a set of circuits based on specified parameters.

    batch_circuits(circuits_list, circuits_per_job):
        Batches a list of circuits for efficient processing.
"""
import itertools
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
import math
import scipy.special as sp

from qrem.qtypes import CircuitCollection
from qrem.common import constants
from qrem.common.experiment.tomoutils import algortithmic_circuits


def compute_number_of_circuits(experiment_type: str, 
                               number_of_qubits: int, 
                               subset_locality: Optional[int],
                               effect_distance: Optional[float] = constants.CIRC_EFFECT_DISTANCE,
                               probability_of_fail: Optional[float] =  constants.CIRC_PROBABILITY_OF_FAIL,
                               limited_circuit_randomness: Optional[bool] = False,
                               max_provider_circuit_count: Optional[int] = 4000 )-> dict:
    """
    Computes the optimal number of quantum circuits needed for the preparation stage of a quantum experiment.

    Parameters
    ----------
    experiment_type : str
        The type of quantum experiment, e.g., DDOT, QDOT (Diagonal/Quantum Detector Overlapping Tomography), which defines the set of symbols used for encoding gates.
    number_of_qubits : int
        The number of qubits involved in the experiment.
    subset_locality : int, optional
        The number of qubits on which the characterization will be performed. If 0, all qubits are considered.
    effect_distance : float, optional
        The tolerated distance between the estimated and actual effect operators.
    probability_of_fail : float, optional
        The probability of estimation error larger than the effect distance.
    limited_circuit_randomness : bool, optional
        True if the number of circuit able to be processed is limited by quantum machine provider, False otherwise.
    max_provider_circuit_count : int, optional
        The maximum number of circuits a provider can handle.

    Returns
    -------
    dict
        A dictionary with keys 'number_of_repetitions', 'total_circuits_count', and 'random_circuits_count', 
        representing the computed values for the quantum experiment.

    Raises
    ------
    ValueError
        If the provided experiment type is not recognized.
    """    

    try:
        number_of_symbols = constants.EXPERIMENT_TYPE_SYMBLOS[experiment_type.lower()]
    except:
        print("Error: wrong experiment type")

    if subset_locality==None:
        subset_locality = number_of_qubits
    n = number_of_qubits
    k = subset_locality
    s = number_of_symbols
    total_circuits_count = int(s**k*(np.log(math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))+(k+1)*np.log(2)+np.log(1./probability_of_fail))/(2*effect_distance**2))
    random_circuits_count = min(max_provider_circuit_count,total_circuits_count)
    number_of_repetitions = max(1,math.ceil(total_circuits_count/(max_provider_circuit_count)))

    return {'number_of_repetitions': number_of_repetitions,
            'total_circuits_count': total_circuits_count,
            'random_circuits_count': random_circuits_count}

def create_random_circuits(experiment_type: str, number_of_qubits: int, number_of_circuits: int)-> List:
    """
    Creates a list of random quantum circuits for the experiment preparation stage.

    Parameters
    ----------
    experiment_type : str
        The type of quantum experiment (e.g., DDOT, QDOT) which defines the set of symbols used for encoding gates.
    number_of_qubits : int
        The number of qubits in the experiment.
    number_of_circuits : int
        The maximum number of circuits to be generated.

    Returns
    -------
    List
        A list of randomly generated quantum circuits, each represented as a list of symbols encoding the gates acting on every qubit.

    Raises
    ------
    ValueError
        If the provided experiment type is not recognized.
    """

    try:
        number_of_symbols = constants.EXPERIMENT_TYPE_SYMBLOS[experiment_type.lower()]
    except:
        print("Error: wrong experiment type")
    
    circuits = np.random.randint(0, number_of_symbols,(number_of_circuits,number_of_qubits), dtype=constants.CIRCUIT_DATA_TYPE)


    return circuits

def check_completeness(experiment_type: str, circuit_list: List, locality = 2, symbols_list:Optional[List] = None,subsets_list = None):
    """
    Checks the completeness of a set of quantum circuits based on specified parameters.

    Parameters
    ----------
    experiment_type : str
        The type of quantum experiment (e.g., DDOT, QDOT) which defines the set of symbols used for encoding gates.
    circuit_list : List
        The list of quantum circuits to check for completeness.
    locality : int, optional
        The number of qubits each circuit should act on.
    symbols_list : List, optional
        The list of symbols used in the circuits. If None, defaults to a range based on the number of symbols.
    subsets_list : List, optional
        A list of qubit subsets to check for completeness. If None, all possible combinations are considered.

    Returns
    -------
    Dict
        A dictionary where keys are tuples representing subsets of qubits and values are lists of missing symbols in those subsets.

    Raises
    ------
    ValueError
        If the provided experiment type is not recognized.
    """
    try:
        number_of_symbols = constants.EXPERIMENT_TYPE_SYMBLOS[experiment_type.lower()]
    except:
        print("Error: wrong experiment type")

    if symbols_list==None:
        symbols_list = list(range(number_of_symbols))
    number_of_qubits = len(circuit_list[0])
    if subsets_list == None:
        subsets = list(itertools.combinations(range(number_of_qubits),locality))
    else:
        subsets = subsets_list


    circuits = np.array(circuit_list)
    absent_symbols_dict = {}
    for t in subsets:
        locality = len(t)
        symbols = list(itertools.product(symbols_list, repeat=locality))
        indices = np.array(t)
        circuits_on_subset = list(map(tuple,circuits[:,indices]))
        absent = set(symbols).difference(circuits_on_subset)
        if absent != set():
            absent_symbols_dict[t] = list(absent)
    return absent_symbols_dict



def generate_circuits(number_of_qubits:int,
                      experiment_type:str = "ddot", 
                      k_locality:Optional[int] = 2,
                      add_random_circuits:Optional[bool] = True,
                      symbols:Optional[List] = None,
                      check_completness:Optional[bool] = True,
                      limited_circuit_randomness:Optional[bool] = False,
                      imposed_max_random_circuit_count:Optional[int] = 4000,
                      imposed_max_number_of_shots:Optional[int] = 1000):
    """
    Generates a set of quantum circuits for characterisation experiment

    Parameters
    ----------
    number_of_qubits : int
        The number of qubits for the circuits.
    experiment_type : str, optional
        The type of quantum experiment (e.g., "ddot"), determining the encoding of gates.
    k_locality : int, optional
        The locality of the circuits, indicating how many qubits are involved in each.
    add_random_circuits : bool, optional
        Whether to add random circuits to the generated set.
    symbols : List, optional
        Specific symbols to be used in the circuit generation.
    check_completeness : bool, optional
        Flag to check the completeness of the generated circuits.
    limited_circuit_randomness : bool, optional
        Whether the randomness in circuit generation is limited.
    imposed_max_random_circuit_count : int, optional
        The maximum number of random circuits allowed.
    imposed_max_number_of_shots : int, optional
        The maximum number of shots (repetitions) for the circuits.

    Returns
    -------
    Tuple
        A tuple containing the generated circuits and related information, such as the number of circuits and the total circuit count.

    Notes
    -----
    The function is still under development and needs further detailing in its implementation and return values.

    Raises
    ------
    ValueError
        If the provided experiment type is not recognized.
    """
    #[1] First we need to compute nubmer of circuits we should use.
    # Either we will be able to send all of them, or not; if not - we want to know how many we loose
    circuit_count_dict = compute_number_of_circuits(experiment_type =experiment_type, 
                                                    number_of_qubits = number_of_qubits, 
                                                    subset_locality=k_locality,
                                                    limited_circuit_randomness = limited_circuit_randomness,
                                                    max_provider_circuit_count = imposed_max_random_circuit_count)
    
    number_of_shots = circuit_count_dict['number_of_repetitions']
    total_circuit_count = circuit_count_dict['total_circuits_count']
    random_circuit_count = circuit_count_dict['random_circuits_count']
    if(random_circuit_count != total_circuit_count):
        print(f"WARNING: Possible count of random circuits ({random_circuit_count}) is lower than desired total circuit count ({total_circuit_count}).")
    #[2] Then we need to generate a hopefully complete set of circuits, even on this low amount
    # - can we estimate what will be a minimal complete set?
    algorithm_circuits = algortithmic_circuits.generate_combinatorial_circuits(experiment_type=experiment_type,number_of_qubits=number_of_qubits,subset_locality=k_locality,symbols=symbols)

    if not add_random_circuits:
        return algorithm_circuits, len(algorithm_circuits), total_circuit_count, number_of_shots


    if(check_completness):
        circuits = algorithm_circuits.copy()
        incomplete_dict = check_completeness(experiment_type=experiment_type,circuit_list=circuits,locality=k_locality,symbols_list=symbols)
        print(f"completeness: {incomplete_dict == {}}")
        while incomplete_dict != {}:
            if len(circuits)+len(list(incomplete_dict.values())[0])<=random_circuit_count:
                circuit_list_update = algortithmic_circuits.complete_on_subset(experiment_type, circuits,
                                                                list(incomplete_dict.keys())[0],
                                                                list(incomplete_dict.values())[0])

                dict_update = check_completeness(experiment_type, circuit_list_update, k_locality)
                circuits = circuit_list_update
                incomplete_dict = dict_update
            else:
                subsets = math.factorial(number_of_qubits)/(math.factorial(k_locality)*math.factorial(number_of_qubits-k_locality))
                print(f"WARNING: Set of circuits of size {len(circuits)} still incomplete on {100*len(incomplete_dict.keys())/subsets}% of subsets")
                #what to to if len(circuits)==rcc but still incomplete?
        #        Future development: avoid this by
        #        - completing on partitions
        #        - algorithmic completeness for k=3,4
                break
        circuit_count = random_circuit_count - len(circuits)
        if circuit_count>0:
            random_circuits = create_random_circuits(experiment_type=experiment_type,
                                          number_of_qubits=number_of_qubits,
                                          number_of_circuits=circuit_count)
            print(f"Adding {circuit_count} random circuits to a {len(circuits)}-element set")
            circuits = np.append(circuits, random_circuits, axis=0)

    else:
        circuit_count = random_circuit_count - len(algorithm_circuits)
        circuits = create_random_circuits(experiment_type=experiment_type,
                                   number_of_qubits=number_of_qubits,
                                   number_of_circuits=circuit_count)
        print(random_circuit_count, circuit_count)
        circuits = np.append(circuits, algorithm_circuits, axis=0)

    incomplete_dict = check_completeness(experiment_type=experiment_type, circuit_list=circuits, locality=k_locality)
    print(f"Set of {len(circuits)} circuits, completeness: {incomplete_dict=={}}")

    if len(circuits)>random_circuit_count:
        print(f"WARNING: cutting off {len(circuits)-random_circuit_count} last circuits")

    return circuits[0:random_circuit_count], random_circuit_count, total_circuit_count, number_of_shots



def batch_circuits(circuits_list:List, circuits_per_job:int=300):
    """
    Organizes a list of quantum circuits into batches for efficient processing.

    Parameters
    ----------
    circuits_list : List
        A list of quantum circuits to be batched.
    circuits_per_job : int, optional
        The maximum number of circuits to include in each batch.

    Returns
    -------
    List
        A list of batches, each containing a subset of the input circuits, suitable for processing or analysis.

    Notes
    -----
    The function automatically adjusts the size of the last batch to accommodate the remaining circuits if the total number of circuits is not divisible by the batch size.
    """

    total_number_of_circuits = len(circuits_list)

    #computes total number of batches
    number_of_jobs = int(np.ceil(total_number_of_circuits / circuits_per_job))

    #initializes baches set
    batches = []
    #iteration over batches_index and creation of baches of size 
    for batch_index in range(number_of_jobs):
        circuits_now = circuits_list[batch_index * circuits_per_job : (batch_index + 1) * circuits_per_job] 
        batches.append(circuits_now)
    return batches
    # Because of how python treats lists/ arrays the last bach has automatically the right size (of the remaining circuits)

"""
    
def generate_circuits(circuit_collection: CircuitCollection):
    collection_dict = circuit_collection.get_dict_format()
    experiment_type = collection_dict['experiment_type']
    number_of_qubits = len(collection_dict['qubit_indices'])
    locality = collection_dict['locality']

    circuit_count_dict = compute_number_of_circuits(experiment_type, number_of_qubits, locality)
    random_circuit_count = circuit_count_dict['random_circuits_count']
    total_circuit_count = circuit_count_dict['total_circuits_count']

    circuit_list = create_circuits(experiment_type=experiment_type,number_of_qubits=number_of_qubits,number_of_circuits=random_circuit_count)
    incomplete_dict = check_completeness(experiment_type=experiment_type,circuit_list=circuit_list,locality=locality)

    while incomplete_dict!={}:
        circuit_list_update = complete_on_subset(experiment_type, circuit_list, list(incomplete_dict.keys())[0], list(incomplete_dict.values())[0])
        dict_update = check_completeness(experiment_type, circuit_list_update, locality)
        circuit_list = circuit_list_update
        incomplete_dict = dict_update

    random_circuit_count = len(circuit_list)

    return circuit_list, random_circuit_count, total_circuit_count
"""

if __name__ == "__main__":
    exp_types = ['coh']
    number_of_qubits = [128]
    locality = [2]
    for e in exp_types:
        for n in number_of_qubits:
            for k in locality:
                print(f"Experiment type: {e}, {n} qubits, locality {k}")
                print(compute_number_of_circuits(e,n,k))
                c = generate_circuits(number_of_qubits=n, experiment_type=e, k_locality=k,add_random_circuits=True,symbols=[4,5])







