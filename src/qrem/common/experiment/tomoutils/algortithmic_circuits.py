"""
Combinatorial Quantum Circuit Generation Submodule

This module contains functions for generating and completing quantum circuits for quantum experiments, particularly in quantum tomography. It provides methods to generate combinatorial circuits, complete circuits based on missing symbols in given subsets, and find optimal partitions of qubit subsets for circuit analysis. The functions support various types of quantum experiments and are designed to handle different quantum tomography methods.

Functions
---------
    generate_combinatorial_circuits: 
        Generates a set of combinatorial circuits for a given quantum experiment.
    complete_on_subset: 
        Completes a set of circuits by adding missing symbols in specified subsets.
    find_partition: 
        Finds an optimal partition of qubit subsets for circuit analysis.
"""
import numpy as np
import math
import qrem.common.constants as constants
import qrem.common.experiment.tomography as tomography
import itertools
from typing import List, Dict, Optional, Callable, Tuple

#np.full(N,value,dtype=consts.CIRCUIT_DATA_TYPE)




def generate_combinatorial_circuits(experiment_type: str,
                               number_of_qubits: int,
                               subset_locality: int, symbols:Optional[List] = None)-> np.array:
    """
    Generates a set of combinatorial circuits for quantum tomography experiments.

    Parameters
    ----------
    experiment_type : str
        The type of quantum experiment (e.g., DDOT, QDOT) which defines the set of symbols used for encoding gates.
    number_of_qubits : int
        The number of qubits involved in the experiment.
    subset_locality : int
        The number of qubits on which each circuit acts.
    symbols : List, optional
        A list of symbols to be used in the circuits. If None, defaults to a range based on the number of symbols for the experiment.

    Returns
    -------
    np.array
        An array of generated quantum circuits, each circuit represented as a list of symbols.

    Raises
    ------
    ValueError
        If the provided experiment type is not recognized.
    """    
    try:
        number_of_symbols = constants.EXPERIMENT_TYPE_SYMBLOS[experiment_type.lower()]
    except:
        print("Error: wrong experiment type")
    if symbols==None:
        symbols = np.array(range(number_of_symbols), dtype = constants.CIRCUIT_DATA_TYPE)
    s = len(symbols)
    n = number_of_qubits
    k = subset_locality

    circuits = [[symbol for i in range(n)] for symbol in symbols]


    combinations_list = list(itertools.product(symbols, repeat=k))
    for c in combinations_list:
        if c == tuple(c[0] for i in range(len(c))):
            combinations_list.remove(c)
    #circuits.append(new_entry)
    m = np.ceil(np.log(n)/np.log(k))
    N = k**m

    for i in np.arange(m):
        for c in combinations_list:
            circ = [c[(int(j/k**i))%k] for j in np.arange(N)]
            circuits.append(circ[0:n])

    if k>2:
        pass

    print(len(circuits))
    return np.array(circuits,dtype = constants.CIRCUIT_DATA_TYPE)

def complete_on_subset(experiment_type: str, circuit_list: List, subset: Tuple, absent_symbols: List):
    """
    Completes a given set of circuits by adding circuits with absent symbols for a specified subset of qubits.

    Parameters
    ----------
    experiment_type : str
        The type of quantum experiment, dictating the encoding of gates.
    circuit_list : List
        The list of existing quantum circuits.
    subset : Tuple
        A tuple of qubits for which absent symbols need to be added.
    absent_symbols : List
        A list of symbols absent in the subset that need to be included in the new circuits.

    Returns
    -------
    List
        An updated list of quantum circuits, including the newly added circuits covering the absent symbols.

    Raises
    ------
    ValueError
        If the provided experiment type is not recognized.
    """
    try:
        number_of_symbols = constants.EXPERIMENT_TYPE_SYMBLOS[experiment_type.lower()]
    except:
        print("Error: wrong experiment type")
    symbols_list = list(range(number_of_symbols))
    number_of_qubits = len(circuit_list[0])
    new_circuit_list = circuit_list
    locality = len(subset)

    for a in absent_symbols:
        new_circuit =np.array( [[np.random.randint(0, number_of_symbols) for i in range(number_of_qubits)]],dtype = constants.CIRCUIT_DATA_TYPE)
        for i in range(locality):
            new_circuit[0][subset[i]] = a[i]
        new_circuit_list = np.append(new_circuit_list,new_circuit,axis=0)

    return new_circuit_list

def find_partition(subset_list, n):
    """
    Finds an optimal partition of a list of qubit subsets for circuit analysis.

    Parameters
    ----------
    subset_list : List
        A list of subsets of qubits to be partitioned.
    n : int
        The total number of qubits in the experiment.

    Returns
    -------
    List
        A list of subsets that form an optimal partition for circuit analysis.

    Notes
    -----
    The function aims to find a partition that evenly distributes the qubits across the subsets, ensuring efficient analysis.
    """    
    subset = subset_list[0]
    k = len(subset)
    p = int(n/k)
    partition = [subset]
    partition_set = set(subset)
    for s in subset_list:
        if partition_set.isdisjoint(set(s)):
            partition_set = partition_set.union(set(s))
            partition.append(s)
        if len(partition)==p:
            break

    return partition





if __name__ == "__main__":
    experiments = ["DDOT","QDOT"]
    number_of_qubits = [128]
    locality = [3]
    for e in experiments:
        for k in locality:
            for n in number_of_qubits:
                print(e,n,k)
                symbols = np.arange(constants.EXPERIMENT_TYPE_SYMBLOS[e.lower()])
                #print(number_of_circuits(e,n,k))
                circuits = generate_combinatorial_circuits(e,n,k)


                incomplete_dict = tomography.check_completeness(e,circuits,k)
                nk = math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
                print(len(incomplete_dict),"incomplete subsets out of ",nk,"fraction: ",1.*len(incomplete_dict)/nk)
                print(incomplete_dict.keys())

                print(find_partition(list(incomplete_dict.keys()),n))


                """
                if k==3:
                    while incomplete_dict != {}:
                        print(len(incomplete_dict.keys()))
                        circuit_list_update = tomography.complete_on_subset(e, circuits,
                                                                 list(incomplete_dict.keys())[0],
                                                                 list(incomplete_dict.values())[0])

                        dict_update = tomography.check_completeness(e, circuit_list_update, k)
                        circuits = circuit_list_update
                        incomplete_dict = dict_update
"""
