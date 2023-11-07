"""qrem.common.math module contains all mathematical funcitons, 
that are useful and used throughout the qrem package.

"""

import warnings #standard imports go first; kets make it those that are python/system/code -related
import functools as ft
import copy
import itertools
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np #then 3rd party imports
import numpy.typing as npt #then 3rd party imports
import cmath # lets add internal meth related imports here

# then internal old imports

import qrem.common.utils #then internal new imports
from qrem.common.constants import SIGNIFICANT_DIGITS

#TODO_PP >> talk with Paweł some format semantics
class Constants:
    """ Contains static methods that return helpful mathematical constants, matrices, arrays etc.

    Attributes
    ----------
    pauli_dict_test : test for attribute documentation

    Notes
    -----
    The choice to use static methods rather than variables was made for documentation/pylint readability. 
    Before Python 3.11 the way to go would be to wrap @classmethod around @property, so you could access this methods as e.g Constant.pauli_sigmas['id'].
    It is only possible however in Python 3.9-3.10. 
    
    Another possibility would be to use (i) module variable or (ii) class variable, however first one is hard to document to provide
    e.g. nice hints in PyCharm or VSCode, with the second you can do this for example with Attributes section at class-level docstring,
    but it won't give you e.g. nice hover hints in VSCode. 
    For that reason now we need to bear with a bit ugly:
    Constant.pauli_sigmas()['id']

    """

    #(PP) just for testing some documentation stuff
    pauli_dict_test = {1,2,3}

    @staticmethod
    def pauli_sigmas():
        """Returns a dictionary with Paili sigmas: id, I, X, Y, Z; all dictionary values are numpy arrays with complex elements. """
        #(PP) put module level variables into classes to group them and be able to document their purpose. 
        
        return {
            'id': np.array([[1., 0j], [0, 1]]),
            'I':np.array([[1., 0j], [0, 1]]),
            'X': np.array([[0.j, 1],
                        [1, 0]]),
            'Y': np.array([[0., -1j],
                        [1j, 0]]),
            'Z': np.array([[1., 0j],
                        [0, -1]])
        }

    @staticmethod
    def bell_states():
        """Returns a dictionary with Bell states : phi+, phi-; psi+; psi-; all dictionary values are numpy arrays. """


        return {'phi+': 1 / np.sqrt(2) * np.array([1, 0, 0, 1]),
                'phi-': 1 / np.sqrt(2) * np.array([1, 0, 0, -1]),
                'psi+': 1 / np.sqrt(2) * np.array([0, 1, 1, 0]),
                'psi-': 1 / np.sqrt(2) * np.array([0, 1, -1, 0]), }



    @staticmethod
    def standard_gates():
        """Returns a dictionary with Standard Gates : 1,I,id,X,Y,Z,S,T,H,SWAP; all dictionary values are numpy complex arrays. """
        #MOVE TO >> common.quantum.QConstans in the future

        return {'1': np.array([[1., 0j], [0, 1]], dtype=complex),
                'I': np.array([[1., 0j], [0, 1]], dtype=complex),
                'id': np.array([[1., 0j], [0, 1]], dtype=complex),
                'X': np.array([[0.j, 1], [1, 0]], dtype=complex),
                'Y': np.array([[0., -1j], [1j, 0]], dtype=complex),
                'Z': np.array([[1., 0j], [0, -1]], dtype=complex),
                'S': np.array([[1, 0], [0, 1j]], dtype=complex),
                'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
                'H': 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex),
                'SWAP': np.array(  [[1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]], dtype=complex)}

    @staticmethod
    def pauli_eigenkets():
        """Returns a dictionary with Pauli Eigenkets : x+,x-,y+,y-,z+,zi; all dictionary values are numpy complex arrays. """
        #MOVE TO >> common.quantum.QConstans in the future ?

        return {'x+': 1 / np.sqrt(2) * np.array([[1], [1]], dtype=complex),
                       'x-': 1 / np.sqrt(2) * np.array([[1], [-1]], dtype=complex),
                       'y+': 1 / np.sqrt(2) * np.array([[1], [1j]], dtype=complex),
                       'y-': 1 / np.sqrt(2) * np.array([[1], [-1j]], dtype=complex),
                       'z+': np.array([[1], [0]], dtype=complex),
                       'z-': np.array([[0], [1]], dtype=complex)}

    @staticmethod
    def pauli_eigenkets_listed(order="ZXY"):
        """Returns a list with Pauli Eigenkets, Default order z+z-x+x-y+y-. Order is defined by a strin, e.g. "ZYX or "XYZ" """
        #MOVE TO >> common.quantum.QConstans in the future ?
        output_list = []
        for el in order:
            if el =="Z":
                output_list.append(Constants.pauli_eigenkets("z+"))
                output_list.append(Constants.pauli_eigenkets("z-"))
            if el =="Y":
                output_list.append(Constants.pauli_eigenkets("y+"))
                output_list.append(Constants.pauli_eigenkets("y-"))
            if el =="X":
                output_list.append(Constants.pauli_eigenkets("x+"))
                output_list.append(Constants.pauli_eigenkets("x-"))
            else:
                raise ValueError(f"Wrong label in the order definition: '{order}' - a letter was not recognized.")

        return output_list

                       
    @staticmethod
    def pauli_measurements():
        """Returns a dictionary with Pauli Measurments on 'X', 'Y', 'Z'. """
        #MOVE TO >> common.quantum.QConstans in the future 

        return {'X': [ketbra(Constants.pauli_eigenkets()['x+']),
                        ketbra(Constants.pauli_eigenkets()['x-'])],
                'Y': [ketbra(Constants.pauli_eigenket()['y+']),
                        ketbra(Constants.pauli_eigenkets()['y-'])],
                'Z': [ketbra(Constants.pauli_eigenkets()['z+']),
                        ketbra(Constants.pauli_eigenkets()['z-'])],
                }

# ===================================================
# Helpful Vector/matrix math
# ===================================================

def outer_product(ket1, ket2=None):
    """Outer product of two vectors.

    Notes
    -----
    - PP: should it just be np.outer(ket1, ket2)?
    - MO: what about complex entries?  
    """

    if ket2 is None:
        ket2 = ket1
    return ket1 @ np.matrix.getH(ket2)
    
def ketbra(ket):
    """Ketbra operation. Can be simplified with numpy
    """

    return outer_product(ket1=ket, ket2=None)

def matrix_zero_check(potential_zero_matrix: np.ndarray,
               significant_digits=SIGNIFICANT_DIGITS) -> bool:
    """
    Functions that checks if matrix is zero. Wraps numpy allclose.

    Parameters
    ----------
    potential_zero_matrix : NDArray
    significant_digits : int

    Returns
    -------
    bool
        True if matrix is 0 with given significant digits accuracy
    """
    threshold = 10 ** (-significant_digits)
    size = list(potential_zero_matrix.shape)

    zeros = np.zeros(size, dtype=type(potential_zero_matrix[0, 0]))
    return np.allclose(potential_zero_matrix, zeros, rtol=threshold)

def identity_check(m_a, significant_digits=SIGNIFICANT_DIGITS) -> bool:
    """Function that checks if matrix is identity (up to global phase), with significant_digits accuracy. 

    Returns
    -------
    bool
        If matrix is identity  returns True
    """
    size = np.size(m_a, 0)
    m_b = round_array_to_ndigits(m_a)

    m_b_phase = cmath.phase(m_b[0, 0])
    m_b_prime = cmath.exp(-1j * m_b_phase) * m_b

    identity_matrix = np.identity(size)

    checking_zeros = round_array_to_ndigits(m_b_prime - identity_matrix)

    return True if matrix_zero_check(checking_zeros, significant_digits) else False

#TEST_ME - old version changed
def round_array_to_ndigits(m_a: np.ndarray, significant_digits=SIGNIFICANT_DIGITS) -> np.ndarray:
    """perform round operation on a given array m_a.

    Notes
    -----
    - PP: refactored from some very werid construction. Should have the same functionality. 
    """
    # OLD VERSION
    
    # # TODO PP: should be: np.around(m_a,  decimal)
    # # TODO TR: Comment this method_name, and intention behind it.
    # m_b = np.array(copy.deepcopy(m_a))
    # with np.nditer(m_b, op_flags=['readwrite']) as it:
    #     for x in it:
    #         x[...] = np.round(x, decimal)

    # return m_b
    return np.around(m_a,  decimals=significant_digits)

#TEST_ME e.g. if m_a has different size then pauli sigmas
def is_pauli_x(m_a) -> bool:
    """Is a matrix an X Pauli Sigma? """
    return True if matrix_zero_check(m_a - Constants.pauli_sigmas()['X']) else False

# TEST_ME Not sure why the negative entries are being cut - check if this is an isue somewhere (MO)
def is_matrix_stochastic(potentially_stochastic_matrix: np.ndarray,
                        stochasticity_type: Optional[str] = 'left',
                        significant_digits: Optional[float] = SIGNIFICANT_DIGITS) -> bool:
    """Test stochasticity of a given input matrix: left|right|ortho. Default is left. 
    You can specify significant digits of the check, default is 9 (accuracy 10^9).

    Parameters
    ----------
    potentially_stochastic_matrix : NDArray 
    stochasticity_type: {'left', 'right', 'ortho', 'doubly'}
        string specyfing what type of stochasticity we want to tests; otrho option is equal to doubly
    significant_digits, default=9
        significant digits specifying the accuracy to which entries need to be positive rows/ columns

    Returns
    -------
        bool 
            is matrix stochastic 

    Raises
    ------
        ValueError
            Wrong stochasticity_type of stochasticity
    
    Notes
    -----
    
    """
    accuracy = 10 ** (-significant_digits)
    shape = potentially_stochastic_matrix.shape[0]

    potentially_stochastic_matrix2 = copy.deepcopy(potentially_stochastic_matrix)
    potentially_stochastic_matrix2[abs(potentially_stochastic_matrix2) < accuracy] = 0
    if np.any(potentially_stochastic_matrix2 < 0):
        return False

    if stochasticity_type == 'left':
        for index_row in range(shape):
            one_now = sum(potentially_stochastic_matrix[:, index_row])

            if abs(1 - one_now) >= accuracy:
                return False
    elif stochasticity_type == 'right':
        for index_row in range(shape):
            one_now = sum(potentially_stochastic_matrix[index_row, :])

            if abs(1 - one_now) >= accuracy:
                return False
    elif stochasticity_type == 'ortho' or stochasticity_type == 'doubly':
        for index_both in range(shape):
            one_now = sum(potentially_stochastic_matrix[:, index_both])
            one_now2 = sum(potentially_stochastic_matrix[index_both, :])
            if abs(1 - one_now) >= accuracy or abs(1 - one_now2) >= accuracy:
                return False

    else:
        raise ValueError('Wrong stochasticity_type of stochasticity')

    return True

def kronecker_product(arguments: List[npt.ArrayLike]) -> npt.NDArray:
    """applies kronecker product to all arrays in the arguments list"""
    return ft.reduce(np.kron, arguments)


# ===================================================
# Helpful Quantum-related
# ===================================================

def apply_unitary_channel(matrix_to_be_rotated, unitary_operator) ->np.ndarray:
    """Apply unitary channel to a matrix: unitary_operator @ matrix_to_be_rotated @ np.matrix.getH(unitary_operator)
    """
    return unitary_operator @ matrix_to_be_rotated @ np.matrix.getH(unitary_operator)

def get_k_local_subsets(number_of_elements: int, subset_size: int, all_sizes_up_to_k = False) -> List:
    """Create all subsets of size (locality) k=subset_size  of n=number_of_elements elements set.
    If all_sizes_up_to_k is set to True, will return all subsets of size from 1 up to k.

    Parameters
    ----------
    number_of_elements: int
        number of qubits
    subset_size: int
        max number of element in a subset
    all_sizes_up_to_k: bool
        should all sizes from 1 up to k be taken into consideration
    Returns
    -------
    List
        list of all subsets up to k=subset_size number of elements
    """
    output = list(itertools.combinations(range(number_of_elements), subset_size))
    
    if not all_sizes_up_to_k:
        return output
    else:
        for i in range(subset_size-1):
            output += list(itertools.combinations(range(number_of_elements), i + 1))
        return output



# ===================================================
# Helpful functions - probability distributions
# ===================================================
def get_sample_from_multinomial_distribution(probability_distribution: npt.ArrayLike, seed=None,
                                                    method='numpy') -> int:
    """Draw one single sample from multinomial distribution (numpy implementation only for now)

    Parameters
    ----------
    probability_distribution: List(int)
        list of probability distribution values
    method: str
        method of drawing, currently based on numpy random and multinomial functions
    seed: int
        seed to initialize numpy random. Default is None

    Returns
    -------
    int
        single sample
    """

    if method.upper() == 'NUMPY':
        rng = np.random.default_rng(seed)
        samples = rng.multinomial(1, probability_distribution)
    else:
        raise ValueError("wrong sampling method")

    return np.argmax(samples)

# ===================================================
# Helpful functions - TVD calculation
# ===================================================



def compute_TVD(p:np.ndarray,q:np.ndarray):
    """
    Calculation of TVD between two probability distributions. 
    """
    return  0.5*np.linalg.norm(p.flatten()-q.flatten(),ord=1)



# ===================================================
# Helpful functions - matrix/vector SWAP 
# ===================================================


def permute_composite_matrix(qubits_list,noise_matrix):
    qubits_list_sorted = copy.copy(qubits_list)
    noise_matrix_permuted = copy.copy(noise_matrix)
    
    qubits_list_sorted.sort()
    while qubits_list != qubits_list_sorted:
        for index_qubit in range(len(qubits_list)-1):
            if qubits_list[index_qubit+1]<qubits_list[index_qubit]:
                noise_matrix_permuted = permute_matrix(noise_matrix_permuted,len(qubits_list),[index_qubit+1,index_qubit+2])
                qubits_list[index_qubit], qubits_list[index_qubit+1] = qubits_list[index_qubit+1], qubits_list[index_qubit]  
    return noise_matrix_permuted

def permute_composite_vector(qubits_list,vector):
    qubits_list_sorted = copy.copy(qubits_list)
    vector_permuted = copy.copy(vector)
    
    qubits_list_sorted.sort()
    while qubits_list != qubits_list_sorted:
        for index_qubit in range(len(qubits_list)-1):
            if qubits_list[index_qubit+1]<qubits_list[index_qubit]:
                vector_permuted= permute_vector(vector_permuted,len(qubits_list),[index_qubit+1,index_qubit+2])
                qubits_list[index_qubit], qubits_list[index_qubit+1] = qubits_list[index_qubit+1], qubits_list[index_qubit]  
    return vector_permuted


def permute_vector(vector, n, transposition):
    # Swap qubits (subspaces) in 2**number_of_qubits dimensional matrix
    # number_of_qubits - number of qubits
    # transposition - which qubits to SWAP.
    # IMPORTANT: in transposition they are numbered from 1

    swap = qubit_swap(n, transposition)
    return swap @ vector


def permute_matrix(matrix, n, transposition):
    # Swap qubits (subspaces) in 2**number_of_qubits dimensional matrix
    # number_of_qubits - number of qubits
    # transposition - which qubits to SWAP.
    # IMPORTANT: in transposition they are numbered from 1

    swap = qubit_swap(n, transposition)
    return swap @ matrix @ swap


def qubit_swap(n, transposition=(1, 1)):
    # create swap between two qubits in 2**number_of_qubits dimensional space
    # labels inside transpositions start from 1.

    D = 2 ** n
    # renumerate for Python convention
    i, j = transposition[0] - 1, transposition[1] - 1

    names = [(bin(j)[2:].zfill(n)) for j in list(range(2 ** n))]

    new_names = copy.copy(names)
    # exchange classical register bits with labels in transposition
    # this defines new order in classical register which respects qubit SWAP
    for k in range(len(new_names)):
        string = list(new_names[k])
        string[i], string[j] = string[j], string[i]
        new_names[k] = ''.join([s for s in string])

    transformation = np.eye(D)

    for x in range(len(names)):
        bit = int(new_names[x], 2)

        # if we need to change the bit, let's do this
        if bit != x:
            transformation[x, x] = 0
            transformation[bit, bit] = 0
            transformation[bit, x] = 1
            transformation[x, bit] = 1

    return transformation