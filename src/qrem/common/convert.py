"""qrem.common.convert module contains helper functions, that allow to convert between various formats used to describe circuit labels, 
bitstrings, quantum registers etc.
"""
import numpy as np
from typing import List, Optional, Iterable
from qrem.common import utils

from qrem.common.constants import CIRCUIT_DATA_TYPE as c_type



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
        return np.array2string(np.flip(bitarray),separator="")[1:-1]
    return np.array2string(bitarray,separator="")[1:-1]

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
def get_qubit_indices_from_keystring(qubits_string: str,
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