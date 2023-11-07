import itertools
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
import math
import scipy.special as sp

from qrem.types import CircuitCollection
from qrem.common import constants
from qrem.common.experiment.tomoutils import algortithmic_circuits


def compute_number_of_circuits(experiment_type: str, 
                               number_of_qubits: int, 
                               subset_locality: Optional[int],
                               effect_distance: Optional[float] = constants.CIRC_EFFECT_DISTANCE,
                               probability_of_fail: Optional[float] =  constants.CIRC_PROBABILITY_OF_FAIL,
                               limited_circuit_randomness: Optional[bool] = False,
                               max_provider_circuit_count: Optional[int] = 4000 )-> dict:
    """Computes the optimal number of circuits for the experiment preparation stage

    Parameters:
        experiment_type (str): DDOT or QDOT (Diagonal/Quantum Detector Overlapping Tomography), defines the set of symbols used to encode the gates
        number_of_qubits (int): number of qubits in the experiment
        subset_locality (int): number of qubits on which the characterization will be performed, default value 0 means all qubits
        effect_distance (float): the tolerated distance between the estimated and actual effect operators
        probability_of_fail (float): probability of estimation error larger than effect_distance
        limited_circuit_randomness (bool): true if there is a limited number of circuits allowed by a provider
        maximum_provider_circuit_count (int):

    Returns:
        the number of circuits for which the probability of error larger  than effect_distance is bounded by probability_of_fail
    Raises:
        an error for a wrong experiment_type

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
    """Creates a list of random mcircuits for the experiment preparation stage

    Parameters:
        experiment_type (str): DDOT or QDOT (Diagonal/Quantum Detector Overlapping Tomography), defines the set of symbols used to encode the gates
        number_of_qubits (int): number of qubits in the experiment
        number_of_circuits (int): maximal number of circuits
    Returns:
        a list of circuits, each given by a list of symbols encoding the gates acting on every qubit
    Raises:
        an error for a wrong experiment_type
    """

    try:
        number_of_symbols = constants.EXPERIMENT_TYPE_SYMBLOS[experiment_type.lower()]
    except:
        print("Error: wrong experiment type")
    
    circuits = np.random.randint(0, number_of_symbols,(number_of_circuits,number_of_qubits), dtype=constants.CIRCUIT_DATA_TYPE)


    return circuits

def check_completeness(experiment_type: str, circuit_list: List, locality = 2, symbols_list:Optional[List] = None,subsets_list = None):
    #TODO: write docstring (KKM)

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



#TODO: WORK IN PROGESS (PP)
def generate_circuits(number_of_qubits:int,
                      experiment_type:str = "ddot", 
                      k_locality:Optional[int] = 2,
                      add_random_circuits:Optional[bool] = True,
                      symbols:Optional[List] = None,
                      check_completness:Optional[bool] = True,
                      limited_circuit_randomness:Optional[bool] = False,
                      imposed_max_random_circuit_count:Optional[int] = 4000,
                      imposed_max_number_of_shots:Optional[int] = 1000):
    
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
    #TODO.... finish based on generate_circuits(circuit_collection: CircuitCollection)
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
    '''
    Function creating list of batches in abstract format. It takes the following inputs
        circuits list - list with tomographic circuits
        circuits_per_job - number of circuits in a batch 
    '''

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
    #TODO: 1. write docstring (KKM)
    #TODO: 2. reference to yaml provider configs
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







