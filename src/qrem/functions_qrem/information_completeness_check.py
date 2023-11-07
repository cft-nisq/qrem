import itertools
import scipy
import numpy as np
import sys





def log_coefficient(number_of_symbols,locality):
    """
    A helper function needed to compute probabilistic bound on number of circuits, for which all combinations of tomographic settings are realized for all subsets of locality k
    Parameters
    ----------
    number_of_symbols: int
        number of different symbols used to construct tomographic circuits: 2 for DDOT circuits, 6 for QDOT circuits (overcomplete Pauli basis)  

    locality: int
        locality of sets for which all the symbols should appear 

    Returns
    -------

    A particular expression needed to calculate probabilistic bound on number of informationally (over)complete set of circuits

    """
    return 1/np.log(number_of_symbols**locality/(number_of_symbols**locality-1))


def circuits_number_estimation(number_of_symbols, locality, number_of_qubits,probability):
    """
    Function cacluclating a probabilitic bound on number of circuits, for which all combinations of tomographic settings are realized for all subsets of locality k
    Parameters
    ----------
    number_of_symbols: int
        number of different symbols used to construct tomographic circuits: 2 for DDOT circuits, 6 for QDOT circuits (overcomplete Pauli basis)  

    locality: int
        locality of sets for which all tomographic settings should appear

    number_of_qubits: 
        number of qubits for which tomographic circuits are realizeed

    probability:
        accepted probability of bound failiure 

    Returns
    -------

    number of circuits needed to realize all combinations of tomographic settings for all subsets of given locality
    """
    return log_coefficient(number_of_symbols,locality)*(np.log( scipy.special.binom(number_of_qubits, locality) ) +locality *np.log(number_of_symbols)+np.log(1./probability))





def information_completeness_verification_quantitative(tomographic_circuits_list, symbol_string_list, locality=2, subsets_list=None):

    """
    Function checking how many times a particular input setting to measurement tomography protocol is realized in a
    given set of circuits for all subsets of qubits of a fixed locality


    Parameters
    ----------
    tomographic_circuits_list : list
        the list of circuits that are used for measurement tomography protocol 

    locality : int
        number corresponding to locality of subsets for which information completness check is performed  

    symbol_string : string
    list of strings of symbols used to encode input settings, e.g. for 6 state overcomplete Pauli basis it's ['0','1','2','3','4','5']

      subsets_list : list of tuples
        list of qubits subsets for which check is performed, by defult set to None

    Returns
    -------
    symbols_subset_dictionary: dictionary a dictionary with keys of a form ((subset_of_qubits), input_setting)
    (e.g. ((0,1),'50')  and values corresponding to the number of times that a given input_setting is
    realized for this particular subset

    missing_symbols: dictionary a dictionary with keys corresponding to qubit_subsets (e.g. (0,1)) and values
    corresponding to list of settings that don't appear for a given subset e.g. ['00','15']
    """
    number_of_qubits = len(tomographic_circuits_list[0])

    #subsets_of_qubits = []
    #symbols_subset_dictionary = {}
    # a list of relevant marginals is created
    #for element in itertools.combinations(range(number_of_qubits), locality):
    #    subsets_of_qubits.append(element)

    symbols_subset_dictionary ={}


    #if no subsets list was provided the check is performed over all subsets of given locality
    if subsets_list==None:
        subsets_of_qubits = []
        symbols_subset_dictionary = {}
        
        # a list of relevant marginals is created
        #for element in itertools.combinations(range(number_of_qubits), locality):
        #    subsets_of_qubits.append(element)
            # a list of all combinations of relevant tomographic symbols in crated
        #symbol_list = itertools.product(symbol_string_list, repeat=locality)
        #tomographic_symbols_list = []
        # a list with all tomographic settings is created
        #for symbol in symbol_list:
            # a joint string corresponding to tomography setting is
        #    key_temp = ''
        #    for str_el in symbol:
        #        key_temp += str_el
        #    tomographic_symbols_list.append(key_temp)
    
    #if the subsets list was provided these subsets are used to perform check  
    else:
        subsets_of_qubits =subsets_list
    


    # this is a loop over all marginals
    for element in subsets_of_qubits:

        locality = len(element)
        # a list of all combinations of relevant tomographic symbols in crated
        symbol_list = itertools.product(symbol_string_list, repeat=locality)
        tomographic_symbols_list = []
        # a list with all tomographic settings is created
        for symbol in symbol_list:
            # a joint string corresponding to tomography setting is
            key_temp = ''
            for str_el in symbol:
                key_temp += str_el
            tomographic_symbols_list.append(key_temp)

        # total numbel of tomographic circuits is updated

        total_number_of_symbols = len(symbol_string_list) ** locality * len(element)
             

        # an empty dictionary with all possible combinations of symbols is created
        for symbol in tomographic_symbols_list:
            # a joint string corresponding to tomography setting is

            key = tuple((element, symbol))
            # a dictionary is initialized
            symbols_subset_dictionary[key] = 0

        # an inner loop over circuits
        for experimental_setting in tomographic_circuits_list:
            key_temp = ''

            # a circuit realized for a particular symbol is constructed
            for i in range(locality):
                key_temp += experimental_setting[element[i]]

            # a key for dictionary is constructed
            key = tuple((element, key_temp))

            # value of dictionary is updated
            if key in symbols_subset_dictionary.keys():
                temporary_value = symbols_subset_dictionary[key]
                temporary_value = temporary_value + 1
                symbols_subset_dictionary[key] = temporary_value

    # here settings that do not appear for particular marginal are saved
    missing_symbols = {}
    for key, occurrence in symbols_subset_dictionary.items():
        if occurrence == 0:
            if key[0] in missing_symbols.keys():
                temporary_value = missing_symbols[key[0]]
                temporary_value.append(key[1])
                missing_symbols[key[0]] = temporary_value
            else:
                missing_symbols[key[0]] = [key[1]]

    return symbols_subset_dictionary, missing_symbols





def information_completeness_verification_qualitative(tomographic_circuits_list, locality, symbol_string_list):
    """
    Function reports fraction of missing tomography settings in a given set of circuits for all subsets of qubits of a
    fixed locality
    Parameters
    ----------
    tomographic_circuits_list : list
        the list of circuits that are used for measurement tomography protocol

    locality : int
        number corresponding to locality of subsets for which information completeness check is performed

    symbol_string_list : string
        list of strings of symbols used to encode input settings, e.g. for 6 state overcomplete Pauli basis it's
        ['0','1','2','3','4','5']

    Returns
    -------
     missing_symbols_fraction: real
        a fraction of missing symbols for given subset structure
    """
    subsets_of_qubits = []

    number_of_qubits = len(tomographic_circuits_list[0])
    # a list of relevant marginals is created
    for element in itertools.combinations(range(number_of_qubits), locality):
        subsets_of_qubits.append(element)

    # How many symbols are involved in tomography
    symbols_number = len(symbol_string_list)
    # what is the total number of symbols times the total number of subsets
    total_number_of_symbols = symbols_number ** locality * scipy.special.comb(number_of_qubits, locality)

    symbols_subset_dictionary = {}
    # an outer loop over subsets
    for element in subsets_of_qubits:

        # an inner loop over circuits
        for experimental_setting in tomographic_circuits_list:
            key_temp = ''

            # a circuit realized for a particular symbol is constructed
            for i in range(locality):
                key_temp += experimental_setting[element[i]]

            # a key for dictionary is constructed
            key = tuple((element, key_temp))

            # a dictionary entry corresponding to realized subset symbol is created
            symbols_subset_dictionary[key] = 1

    symbols_number = len(symbols_subset_dictionary.keys())

    # a fraction of missing symbols is returned
    missing_symbols_fraction = (total_number_of_symbols - symbols_number) / (total_number_of_symbols)

    return missing_symbols_fraction
