"""
Registers Module for QREM
=========================

qrem.common.registers module contains helper functions, that allow to create, manipuilate  circuit labels, 
bitstrings, quantum registers etc.
"""
import numpy as np



def all_possible_bitstrings_of_length(n: int, rev=True, form=str):
    """
    Generate all possible bitstrings of a given length.

    This function creates a list of all possible bitstrings for a specified number of qubits. The bitstrings can be
    generated in normal or reverse order, and in string or list format.

    Parameters
    ----------
    n : int
        The number of qubits or length of the bitstrings.
    rev : bool, optional
        If True, the bitstrings are generated in reverse order (default is True).
    form : type, optional
        The format of the bitstrings, either string or list (default is str).

    Returns
    -------
    list
        A list of bitstrings in the specified format and order.
        Example: n=2 returns ['00', '01', '10', '11'].
    """
       
    if (form == str):
        if (rev == True):
            return [(bin(j)[2:].zfill(n))[::-1] for j in list(range(2 ** n))]
        else:
            return [(bin(j)[2:].zfill(n)) for j in list(range(2 ** n))]
    elif (form == list):
        if (rev == True):

            return [(list(bin(j)[2:].zfill(n))[::-1]) for j in list(range(2 ** n))]
        else:
            return [(list(bin(j)[2:].zfill(n))) for j in list(range(2 ** n))]
    # if (reversed == True):
    #     return [(bin(j)[2:].zfill(number_of_bits))[::-1] for j in list(range(2 ** number_of_bits))]
    # else:
    #     return [(bin(j)[2:].zfill(number_of_bits)) for j in list(range(2 ** number_of_bits))]
