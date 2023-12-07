"""
QREM Printer Module
===================

qrem.common.printer module contains helpful functions for logging and console printouts.

For now, it contains only console printing capabilities. In the future, it may provide options to log output to a specified log path, print on the console, or both.

Current contents of this module include:
- `round_matrix_for_print`: Round a matrix with specified decimal precision while handling Python artifacts.
- `zeros_to_dots_for_print`: Replace zeros with dots in a matrix for printing.
- `qprint`: Print colored text with optional additional information.
- `warprint`: Wrapper for `qprint` with a default color of 'YELLOW'.
- `errprint`: Wrapper for `qprint` with a default color of 'RED'.
- `qprint_array`: Print a human-readable representation of an array.


Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""


import copy

from colorama import Fore, Style

import numpy as np
import numpy.typing as npt
import pandas as pd

#==============================================
# Helper formatters
#==============================================
def round_matrix_for_print(matrix_to_be_rounded: npt.NDArray,
                 decimal: int=3) -> npt.NDArray:
    """
    Round a matrix with specified decimal precision while handling Python artifact ssuch as "-0.", "0j" etc.

    Parameters
    ----------
    matrix_to_be_rounded : npt.NDArray
        The matrix to be rounded.
    decimal : int, optional
        The number of decimal places to round to (default is 3).

    Returns
    -------
    npt.NDArray
        The rounded matrix.

    Examples
    --------
    >>> matrix = np.array([[0.123456, -0.0], [1.2345678, 0.0]])
    >>> round_matrix_for_print(matrix)
    array([[0.123, 0.   ],
           [1.235, 0.   ]])

    """
    data_type = type(matrix_to_be_rounded[0, 0])

    first_dimension = matrix_to_be_rounded.shape[0]
    second_dimension = np.size(matrix_to_be_rounded, axis=1)

    rounded_matrix = np.zeros((first_dimension, second_dimension), dtype=data_type)

    for first_index in range(0, first_dimension):
        for second_index in range(0, second_dimension):
            real_part = round(np.real(matrix_to_be_rounded[first_index, second_index]), decimal)

            if data_type == complex or data_type == np.complex128:
                imaginary_part = round(np.imag(matrix_to_be_rounded[first_index, second_index]),
                                       decimal)
            else:
                imaginary_part = 0

            # In the following we check whether some parts are 0 and then we leave it as 0
            # Intention here is to remove some Python artifacts such as leaving "-0" instead of 0.
            # see function's description.
            if abs(real_part) != 0 and abs(imaginary_part) != 0:
                if data_type == complex or data_type == np.complex128:
                    rounded_matrix[first_index, second_index] = real_part + 1j * imaginary_part
                else:
                    rounded_matrix[first_index, second_index] = real_part
            elif abs(real_part) == 0 and abs(imaginary_part) == 0:
                rounded_matrix[first_index, second_index] = 0
            elif abs(real_part) == 0 and abs(imaginary_part) != 0:
                if data_type == complex or data_type == np.complex128:
                    rounded_matrix[first_index, second_index] = 1j * imaginary_part
                else:
                    rounded_matrix[first_index, second_index] = 0
            elif abs(real_part) != 0 and abs(imaginary_part) == 0:
                rounded_matrix[first_index, second_index] = real_part

    return rounded_matrix

def zeros_to_dots_for_print(matrix, rounding_decimal = 3) -> npt.NDArray:
    """
    Replace zeros with dots in a matrix for printing.

    Parameters
    ----------
    matrix : npt.NDArray
        The matrix to be printed.
    rounding_decimal : int, optional
        The number of decimal places to consider as zero (default is 3).

    Returns
    -------
    npt.NDArray
        The matrix with zeros replaced by dots.

    Examples
    --------
    >>> matrix = np.array([[0.001, 0.0], [0.0, 0.002]])
    >>> zeros_to_dots_for_print(matrix)
    array([['0.001', '.'],
           ['.', '0.002']], dtype='<U32')

    """   
    m = matrix.shape[0]
    n = matrix.shape[1]

    B = np.zeros((m, n), dtype=dict)

    for i in range(m):
        for j in range(n):
            el = matrix[i, j]
            if (abs(np.round(el, rounding_decimal)) >= 0):
                B[i, j] = el
            else:
                B[i, j] = '.'

    return B

def qprint(colored_string: str, stuff_to_print = '', color='CYAN', print_floors=False) -> None:
    """Print colored text with optional additional information.
    FUTURE: Should be extended/changed to have also logging capabilities

    Parameters
    ----------
    colored_string : str
        The main text to be printed with color.
    stuff_to_print : str or any, optional
        Additional information to be printed without color (default is '').
    color : {'CYAN', 'RED', 'YELLOW', 'GREEN', 'BLUE'}, optional
        The color for the main text (default is 'CYAN').
    print_floors : bool, optional
        Whether to print floor separators (default is False).

    Examples
    --------
    >>> qprint("Hello, World!", color='GREEN')
    >>> qprint("Error:", "Something went wrong", color='RED')

    """    
    if print_floors:
        print("_________________________")

    if isinstance(color, str):
        if color.lower() == 'red':
            color = Fore.RED
        elif color.lower() == 'yellow':
            color = Fore.YELLOW
        elif color.lower() == 'green':
            color = Fore.BLUE
        elif color.lower() == 'blue':
            color = Fore.GREEN
        elif color.lower() == 'cyan':
            color = Fore.CYAN
        else:
            color = Fore.CYAN

    if isinstance(stuff_to_print, str):
        if stuff_to_print == '':
            print(color + Style.BRIGHT + str(colored_string) + Style.RESET_ALL)
        elif stuff_to_print == '\n':
            print(color + Style.BRIGHT + str(colored_string) + Style.RESET_ALL)
            print()
        else:
            print(color + Style.BRIGHT + str(colored_string) + Style.RESET_ALL, repr(stuff_to_print))
    else:
        print(color + Style.BRIGHT + str(colored_string) + Style.RESET_ALL, repr(stuff_to_print))
    if print_floors:
        print("_________________________")

def warprint(colored_string: str, stuff_to_print = '', color='YELLOW', print_floors=False) -> None:
    """
    Warning print helper function. Wrapper for qprint with a default color of 'YELLOW'.

    Parameters
    ----------
    colored_string : str
        The main text to be printed with color.
    stuff_to_print : str or any, optional
        Additional information to be printed without color (default is '').
    color : {'CYAN', 'RED', 'YELLOW', 'GREEN', 'BLUE'}, optional
        The color for the main text (default is 'YELLOW').
    print_floors : bool, optional
        Whether to print floor separators (default is False).

    Examples
    --------
    >>> warprint("Warning:", "This is a warning")

    """
    qprint(colored_string=colored_string , stuff_to_print = stuff_to_print, color=color, print_floors=print_floors)

def errprint(colored_string: str, stuff_to_print = '', color='RED', print_floors=False) -> None:
    """
    Print helper function for printing errors. Wrapper for qprint with a default color of 'RED'.

    Parameters
    ----------
    colored_string : str
        The main text to be printed with color.
    stuff_to_print : str or any, optional
        Additional information to be printed without color (default is '').
    color : {'CYAN', 'RED', 'YELLOW', 'GREEN', 'BLUE'}, optional
        The color for the main text (default is 'RED').
    print_floors : bool, optional
        Whether to print floor separators (default is False).

    Examples
    --------
    >>> errprint("Error:", "Something went wrong")

    """
    qprint(colored_string=colored_string , stuff_to_print = stuff_to_print, color=color, print_floors=print_floors)

def qprint_array(arrray_to_print, rounding_decimal:int=3):
    """
    Print a human-readable representation of an array.

    Parameters
    ----------
    array_to_print : npt.NDArray
        The array to be printed.
    rounding_decimal : int, optional
        The number of decimal places for rounding (default is 3).

    Examples
    --------
    >>> array = np.array([[0.123456, -0.0], [1.2345678, 0.0]])
    >>> qprint_array(array)

    """

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 110)
    try:
        if (arrray_to_print.shape[0] == 1 or arrray_to_print.shape[1] == 1):
            B = copy.deepcopy(arrray_to_print)
            if (arrray_to_print.shape[0] == 1 and arrray_to_print.shape[1] == 1):
                print(np.round(arrray_to_print[0, 0], rounding_decimal))
            elif (arrray_to_print.shape[0] == 1):
                print([np.round(x[1], rounding_decimal) for x in arrray_to_print])
            elif (arrray_to_print.shape[1] == 1):
                print([np.round(x[0], rounding_decimal) for x in arrray_to_print])
        else:
            B = copy.deepcopy(arrray_to_print)
            C = round_matrix_for_print(B, rounding_decimal)
            D = zeros_to_dots_for_print(C, rounding_decimal)
            print(pd.DataFrame(D))
    except(IndexError):
        if len(arrray_to_print.shape) == 1:
            print([np.round(x, rounding_decimal) for x in arrray_to_print])
        else:
            print(pd.DataFrame(np.array(np.round(arrray_to_print, rounding_decimal))))


