"""qrem.common.printer module contains all helpful functions for logging and console printouts.
For now it contains only console printing capabilities. 
In future it will be a choice - either you will be able to log output into a specified log path, print on console or both,

Current contents of this module contain:
- pretty text print and array print for preview

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
    This function rounds matrix in a nice way.
    "Nice" means that it removes funny Python artifacts such as "-0.", "0j" etc.

    Parameters
    ----------
    matrix_to_be_rounded: npt.NDArray
    decimal: int

    Returns
    -------
    npt.NDArray
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
    This function changes zeros into dots in printing; values are treated as 0 with rounding_decimal significant digits ( default is 3) 

    Parameters
    ----------
    matrix: npt.NDArray
    rounding_decimal: int

    Returns
    -------
    npt.NDArray
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
    """A bit fancier print function that uses colors for output. 
    Should be extended/changed to have also logging capabilities

    Parameters
    ----------
    colored_string: str
        colored_string is printed with color
    stuff_to_print: str
        stuff_to_print is printed without color, after colored_string
    color: str
        color, you can use 'red','green','blue',or 'cyan' currently
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
    qprint(colored_string=colored_string , stuff_to_print = stuff_to_print, color=color, print_floors=print_floors)

def errprint(colored_string: str, stuff_to_print = '', color='RED', print_floors=False) -> None:
    qprint(colored_string=colored_string , stuff_to_print = stuff_to_print, color=color, print_floors=print_floors)

def qprint_array(arrray_to_print, rounding_decimal:int=3):
    """
    This function prints array in a human-readable format, removing artifacts such as "-0.", "0j" etc.
    Matrix is rounded up to rounding_decimal significant digits

    Parameters
    ----------
    arrray_to_print: npt.NDArray
    rounding_decimal: int

    Returns
    -------
    npt.NDArray
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


