"""
Convert Module for QREM
=======================

qrem.common.convert module contains helper functions, that allow to convert between various formats used to describe circuit labels, 
bitstrings, quantum registers etc.
"""
import numpy as np
from typing import List, Optional, Iterable, Dict, Type
from qrem.common import utils
import re
from qrem.common.constants import CIRCUIT_DATA_TYPE as c_type
from qrem.qtypes.characterization_data import CharacterizationData 
import copy 



# =======================================================================================
# BITSTRINGS CONVERSIONS
# =======================================================================================
def bitstring_to_integer(bitstring: str) -> int:
    return int(bitstring, 2)

def integer_to_bitstring(integer: int,
                         number_of_bits: int) -> str:
    """Return binary representation of an integer in a string form

    Parameters
    ----------
    integer: int
        input integer which bytes will be interpreted as bitstring
    number_of_bits: int
        Number of bits.It  can be greater than minimal needed to represent integer (but if the input number of bits
        is smaller then the minimal one, the latter one will be executed )
    """

    return "{0:b}".format(integer).zfill(number_of_bits)

#ndarrays
def bitstring_to_ndarray(bitstring: str, reverse = False) -> np.ndarray:
    """Return numpy array  representation of a bitstring

    Parameters
    ----------
    bitstring: string
        input bitstring
    reverse: bool
        The output array can be reversed in order. Default is False
    """
    
    arr = np.frombuffer(bytes(bitstring,"utf-8"), dtype=np.uint8) - ord('0')
    if reverse:
        return np.flip(arr)
    return arr

def ndarray_to_bitstring(bitarray: np.ndarray, reverse = False) -> str:
    if reverse:
        return np.array2string(np.flip(bitarray),separator="", max_line_width=9999)[1:-1]
    return np.array2string(bitarray,separator="", max_line_width=9999)[1:-1]

#dit-strings
def integer_to_ditstring(integer, base, number_of_dits):
    """Return nnary representation of an integer in a string form
    based on
    https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-to-a-string-in-any-base

    Parameters
    ----------
    integer: int
        input integer value
    base: int
        base for the ditstring
    number_of_dits: int
        Number of bits.It  can be greater than minimal needed to represent integer (but if the input number of bits
        is smaller then the minimal one, the latter one will be executed )
    """

    if integer == 0:
        return [0 for _ in range(number_of_dits)]
    digits = []
    while integer:
        digits.append(int(integer % base))
        integer //= base

    # digits = sorted(digits,key = lambda x: x[0])
    if len(digits) < number_of_dits:
        for _ in range(number_of_dits - len(digits)):
            digits.append(0)
    return digits[::-1]

def negate_bitstring(bitstring):
    '''This function creates a negation of a bitstring (assuming it is in string format)
    Parameters
    ----------
    bitstring: int
        input bitstring
    '''
    return ''.join('1' if x == '0' else '0' for x in bitstring)
    # potentially better version? Is it really useful?
    # def negate_bitstring(bit):
    #     if isinstance(bit,int) or isinstance(bit,float) or isinstance(bit,complex):
    #         if bit==0:
    #             return 1
    #         elif bit==1:
    #             return 0
    #         else:
    #             raise ValueError(f"Wrong bit: {bit}")
    #     elif isinstance(bit,str):
    #         if bit in ['0']:
    #             return '1'
    #         elif bit in ['1']:
    #             return '0'
    #         else:
    #             raise ValueError(f"Wrong bit: {bit}")
    #     else:
    #         raise ValueError(f"Wrong datatype of {bit} - '{type(bit)}'")
    pass

def sort_bitstring(string:str, new_order:Iterable) -> str:
    """Sort bits in string according to new_order"""
    sorted_string = utils.ort_things(list(string), new_order)
    sorted_string = ''.join([str(s) for s in sorted_string])
    return sorted_string

def sort_bitarray(bitarray: np.ndarray, new_order:Iterable) -> np.ndarray:
    """Sort bits in string according to new_order"""
    sorted_array = utils.sort_things(bitarray, new_order)
    array =np.ndarray(sorted_array,dtype=c_type)
    return array


# =======================================================================================
# REGISTERS
# =======================================================================================
#MOVE_TO >> core.utils core.numerics
def get_ditstrings_register(base, number_of_dits):
    """ Return a list of natural numbers fitting in number_of_dits with base dit-representation """
    return [integer_to_ditstring(j, base, number_of_dits=number_of_dits) for j in
            list(range(base ** number_of_dits))]

if __name__ == "__main__":
    a = bitstring_to_ndarray("100101",reverse =True)
    print(a)
    print(ndarray_to_bitstring(a,reverse =True))



    
# =======================================================================================
# QBIT INDICIES
# =======================================================================================
#MOVE_TO >> core.utils
def keystring_to_qubit_indices(qubits_string: str,
                                  with_q: Optional[bool] = False) -> List[int]:
    """Return list of qubit indices from the string of the form "q0q1q22q31"
    :param qubits_string (string): string which has the form of "q" followed by qubit index
    :param (optional) with_q (Boolean): specify whether returned indices
                                        should be in form of string with letter

    :return: list of qubit indices:

    depending on value of parameter "with_q" the mapping will be one of the following:

    if with_q:
        'q1q5q13' -> ['q1','q5','q13']
    else:
        'q1q5q13' -> [1,5,13]
    """

    numbers = re.findall(r'\d+', qubits_string)

    if with_q:
        qubits = ['q' + s for s in numbers]
    else:
        qubits = [int(s) for s in numbers]

    return qubits


def qubit_indices_to_keystring(list_of_qubits: List[int]) -> str:
    """ from subset of qubit indices get the string that labels this subset
        using convention 'q5q6q12...' etc.
    :param list_of_qubits: labels of qubits

    :return: string label for qubits

     NOTE: this function is "inverse operation" to keystring_to_qubit_indices.
    """

    if list_of_qubits is None:
        return None
    return 'q' + 'q'.join([str(s) for s in list_of_qubits])

def qubits_keystring_to_tuple(qubits_string):
    return tuple(keystring_to_qubit_indices(qubits_string))
    



def change_state_dependent_noise_matrix_format(noise_matrix:Dict) -> Dict:

    """
    .. note:: Deprecated in QREM 0.1.5
          Function transforming representation of noise matrices. Used to translate results obtained with QREM versions < 0.1.5 
     
    """

    state_dependent_noise_matrix_in_new_format = {}


    for key in noise_matrix.keys():
        
        if key != 'averaged':
            
        
            for neighbors_state, state_noise_matrix in noise_matrix[key].items():
        
                new_index = [int(character) for character in neighbors_state]
        
                state_dependent_noise_matrix_in_new_format[tuple(new_index)] = state_noise_matrix
            
        elif key == 'averaged':

            state_dependent_noise_matrix_in_new_format[key] = noise_matrix[key]
    
    return(state_dependent_noise_matrix_in_new_format)
    


def change_state_dependent_noise_matrices_dictionary_format(noise_matrices_dictionary:Dict) -> Dict:

    
    """
     .. note:: Deprecated in QREM 0.1.5
          Function transforming representation of noise matrices. Used to translate results obtained with QREM versions < 0.1.5 
        

    
   

    """

    state_dependent_noise_matrices_dictionary_in_new_format = {}


    for key, item in noise_matrices_dictionary.items():

        state_dependent_noise_matrices_dictionary_in_new_format[key] = change_state_dependent_noise_matrix_format(noise_matrix=item)
        
        
    
    return(state_dependent_noise_matrices_dictionary_in_new_format)

def change_format_of_cn_dictionary(cn_dictionary:Dict)->Dict:

    pattern='q[0-9]+'
    cn_dictionary_new = {}
    for key,entry in cn_dictionary.items():
        index_list = re.findall(pattern,key)

        cluster_arrangement = []
        for index in index_list:
            cluster_arrangement.append(int(index[1:]))
        if entry == []:
            new_key = None
        else:
            new_key = tuple(entry)
        cn_dictionary_new[tuple(cluster_arrangement)] = new_key
    return cn_dictionary_new

#### used to divide data in simulation 

def divide_data_into_characterization_benchmark_coherence_witness_data(characterization_data: Type[CharacterizationData]) -> CharacterizationData:

    
    
    if characterization_data.ground_states_list != None:
        
        characterization_data.benchmark_results_dictionary = {}
        characterization_data.benchmark_marginals_dictionary = {}

        
        for state in characterization_data.ground_states_list:
            characterization_data.benchmark_results_dictionary[state] = copy.copy(characterization_data.results_dictionary[state])
            characterization_data.benchmark_marginals_dictionary[state] =copy.copy( characterization_data.marginals_dictionary[state])
            del characterization_data.results_dictionary[state]
            del characterization_data.marginals_dictionary[state]
        
     

   
    
    if characterization_data.coherence_witnesses_list != None:
        characterization_data.coherence_witness_results_dictionary = {}
        characterization_data.coherence_witness_marginals_dictionary = {}
        for setting in characterization_data.coherence_witnesses_list:
            setting_string = ''
            
            for element in setting:
                setting_string = setting_string+ str(element)
            characterization_data.coherence_witness_results_dictionary[setting_string] = copy.copy(characterization_data.results_dictionary[setting_string])
            characterization_data.coherence_witness_marginals_dictionary[setting_string] =copy.copy( characterization_data.marginals_dictionary[setting_string])
            del characterization_data.results_dictionary[setting_string]
            del characterization_data.marginals_dictionary[setting_string]
       
   

    return characterization_data 

#### used in older versions of qrem

def reverse_bitstrings_in_counts_dictionary(counts_dictionary):
    """Used only by convert_counts_overlapping_tomography()"""
    counts_reversed = {}

    for bitstring, ticks in counts_dictionary.items():
        counts_reversed[bitstring[::-1]] = ticks
    return counts_reversed

def merge_multiple_counts_dictionaries(counts_dictionaries_list: List[Dict[str, int]]) -> Dict[
    str, int]:
    """
    Merge multiple counts dictionaries.
    This is useful when you have results of multiple implementations of the same experiment.

    :param counts_dictionaries_list: list of results of counts dictionaries of the form:
                                    {'bitstring":number of measurements}
    :return:
    """

    # first dictionary will be template to which we will add counts
    merged_counts = copy.deepcopy(counts_dictionaries_list[0])

    # we go through all other dictionaries
    for counts_dictionary in counts_dictionaries_list[1:]:
        for bitstring, ticks in counts_dictionary.items():
            if bitstring in merged_counts.keys():
                merged_counts[bitstring] += ticks
            else:
                merged_counts[bitstring] = ticks

    return merged_counts


#JT: 
def convert_counts_overlapping_tomography(counts_dictionary: Dict[str, Dict[str, int]],
                                          experiment_name: str,
                                          reverse_bitstrings=True,
                                          old_send_procedure=True):
    """
    This function converts unprocessed dictionary of experiment results, where multiple keys can describe identical
    circuits (eg. "DDOT-010no0", "DDOT-010no1", see description of string_cutter below), to a dictionary where a key
    uniquely corresponds to a circuit and value is the combined counts from all such circuits in the unprocessed
    dictionary. In resulting dictionary outcome bit strings and circuit names are in big-endian order.

    param counts_dictionary: dictionary where the key is circuit name (e.g. "DDOT-010no3", described in inner
                              string_cutter function below) and value is dictionary of counts, where the key is a string
                              denoting classical outcome and the value the number of its occurrences in given experiment.
    param experiment_name: string denoting the type of experiment whose results counts_dictionary contains, e.g. 'QDOT';
                            the valid names are specified in SeparableCircuitsCreator.__valid_experiments_names__
    param reverse_bitstrings: bool; if False, the bitstring denoting classical outcome in counts_dictionary will be
                               interpreted as big-endian (qubit 0 on the left); if True it will be interpreted as
                               little-endian (qubit 0 on the right) and will be reversed to conform to QREM convention.

    :return: big_counts_dictionary: dictionary where key is string describing circuit (e.g. '010' means gates:
                                    iden, X, iden on qubits 0, 1, 2 respectively) and value is
                                    dictionary where key is bitstring describing outcome (e.g. '100') and value is
                                    number of occurences of that outcome in the experiment; these strings are big-endian.
    """
    def string_cutter(circuit_name: str):
        """
        This inner function cuts the name of the circuit to the format that will later be used by
        tomography data analyzers.


        param circuit_name:
        It assumes the following convention:

        circuit_name = "experiment name" + "-" + "circuit label"+
        "no"+ "integer identifier for multiple implementations of the same circuit"

        for example the circuit can have name:
        "DDOT-010no3"

        which means that this experiment is Diagonal Detector Overlapping Tomography (DDOT),
        the circuit implements state "010" (i.e., gates iden, X, iden on qubits 0,1,2 - big-endian order), and
        in the whole circuits sets this is the 4th (we start counting from 0) circuit that implements
        that particular state.

        :return: big_counts_dictionary
        """
        # from qrem.noise_characterization.tomography_design.overlapping import SeparableCircuitsCreator

        # if experiment_name.lower() not in SeparableCircuitsCreator.__valid_experiments_names__:
        #     raise ValueError(f"ONLY the following experiments are supported:\n{SeparableCircuitsCreator.__valid_experiments_names__}")

        experiment_string_len = len(list(experiment_name))
        full_name_now = circuit_name[experiment_string_len + 1:]
        new_string = ''
        for symbol_index in range(len(full_name_now)):
            if full_name_now[symbol_index] + full_name_now[symbol_index + 1] == 'no':
                break
            new_string += full_name_now[symbol_index]
        return new_string

    big_counts_dictionary = {}

    for circuit_name, counts_dict_now in counts_dictionary.items():
        
        #the line below was used with data obtained with the old sending routine  
        
        if old_send_procedure:
            proper_name_now = string_cutter(circuit_name)
        else: 
            m=re.search(r'\d+', circuit_name)
            proper_name_now=m.group()
        if reverse_bitstrings:
            counts_dict_now = reverse_bitstrings_in_counts_dictionary(counts_dict_now)

        if proper_name_now not in big_counts_dictionary.keys():
            big_counts_dictionary[proper_name_now] = counts_dict_now
        else:
            big_counts_dictionary[proper_name_now] = merge_multiple_counts_dictionaries(
                [big_counts_dictionary[proper_name_now], counts_dict_now])

    return big_counts_dictionary

##check if this can be deprecated 

def change_state_dependent_noise_matrix_format(noise_matrix:Dict) -> Dict:

    """
    Function transforming old to new format of state dependent matrix 

    """

    state_dependent_noise_matrix_in_new_format = {}


    for key in noise_matrix.keys():
        
        if key != 'averaged':
            
        
            for neighbors_state, state_noise_matrix in noise_matrix[key].items():
        
                new_index = [int(character) for character in neighbors_state]
        
                state_dependent_noise_matrix_in_new_format[tuple(new_index)] = state_noise_matrix
            
        elif key == 'averaged':

            state_dependent_noise_matrix_in_new_format[key] = noise_matrix[key]
    
    return(state_dependent_noise_matrix_in_new_format)
    

def change_state_dependent_noise_matrices_dictionary_format(noise_matrices_dictionary:Dict) -> Dict:

    """
    Function transforming old to new format of state dependent matrix 

    """

    state_dependent_noise_matrices_dictionary_in_new_format = {}


    for key, item in noise_matrices_dictionary.items():

        state_dependent_noise_matrices_dictionary_in_new_format[key] = change_state_dependent_noise_matrix_format(noise_matrix=item)
        
        
    
    return(state_dependent_noise_matrices_dictionary_in_new_format)

def reorder_classical_register(new_order: Iterable) -> List:
    # reorder classical register according to new_order.
    n = len(new_order)

    # get standard classical register
    standard_register = [(bin(j)[2:].zfill(n)) for j in list(range(2 ** n))]

    return [sort_bitstring(s, new_order) for s in standard_register]

def reorder_probabilities(probabilities, new_order):
    # sort elements of probabilities vector or array according to new_order defined for bits, into a vector

    D = len(probabilities)
    array_format = False
    if isinstance(probabilities, np.ndarray):
        probabilities = probabilities.reshape(D, 1)
        array_format = True

        # get classical register according sorted to new order
    new_classical_register = reorder_classical_register(new_order)
    # sort probabilities entries according to new register
    sorted_probs = utils.sort_things(probabilities, new_classical_register)

    if array_format:
        return np.array(sorted_probs).reshape(D, 1)
    else:
        return sorted_probs
    


def convert_results_dictionary_to_new_format(noisy_results_dictionary):
    new_format_dict = {}
    for input, result in noisy_results_dictionary.items():
        keys = list(result.keys())
        values = list(result.values())

        bool_matrix = np.array([list(map(int, key)) for key in keys], dtype=bool)
        int_vector = np.array(values, dtype=int)
        new_format_dict[input] = (bool_matrix, int_vector)
    return new_format_dict