"""
Quantum Readout Error Mitigation (QREM) POVM Utility Module
===========================================================

This module, as part of the QREM package, offers a comprehensive set of functions for dealing with 
Positive Operator-Valued Measures (POVMs) in the context of quantum readout error mitigation. It includes 
methods for analyzing, modifying, and applying various transformations to POVMs. Additionally, the module 
provides utility functions for handling vector and matrix permutations, calculating statistical and 
coherent error bounds, and manipulating frequency data from quantum experiment outcomes.


References
----------
- F. B. Maciejewski, Z. Zimborás, M. Oszmaniec, "Mitigation of readout noise in near-term quantum devices
  by classical post-processing based on detector tomography", Quantum 4, 257 (2020).
- F. B. Maciejewski, F. Baccari, Z. Zimborás, M. Oszmaniec, "Modeling and mitigation of cross-talk effects 
  in readout noise with applications to the Quantum Approximate Optimization Algorithm", Quantum 5, 464 (2021).
- Z. Puchała, Ł. Pawela, A. Krawiec, R. Kukulski, "Strategies for optimal single-shot discrimination of 
  quantum measurements", Phys. Rev. A 98, 042103 (2018), https://arxiv.org/abs/1804.05856.
- T. Weissman, E. Ordentlich, G. Seroussi, S. Verdul, M. J. Weinberger, Technical Report HPL-2003-97R1, 
  Hewlett-Packard Labs (2003).
- J. Smolin, J. M. Gambetta, G. Smith, "Maximum Likelihood, Minimum Effort", Phys. Rev. Lett. 108, 070502
  (2012), https://arxiv.org/abs/1106.5458.

Functions
---------
- euler_angles_1q
- get_su2_parametrizing_angles
- get_unitary_change_ket_qubit
- get_offdiagonal_povm_part
- get_diagonal_povm_part
- apply_stochastic_map_to_povm
- get_stochastic_map_from_povm
- get_povm_from_stochastic_map
- get_coherent_part_of_the_noise
- get_CBT_norm
- get_choi_from_POVM
- get_POVM_from_choi
- check_if_povm
- get_statistical_error_bound
- get_coherent_error_bound
- get_correction_error_bound_from_data_and_statistical_error
- get_correction_error_bound_from_data
- get_correction_error_bound_from_parameters
- get_correction_matrix_from_povm
- get_averaged_lambda

- check_if_projector_is_in_computational_basis
- computational_projectors
- computational_basis

- find_closest_prob_vector_l2
- find_closest_prob_vector_l1
- reorder_probabilities

- reorder_classical_register
- get_enumerated_rev_map_from_indices
- all_possible_bitstrings_of_length
- get_classical_register_bitstrings
- get_possible_n_qubit_outcomes

- counts_dict_to_frequencies_vector
Notes
-----
The module is designed to support various operations on POVMs and related quantum measurement data. 
It is an integral part of the Quantum Readout Error Mitigation framework, aiding in the accurate analysis 
and correction of quantum measurement errors.

    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""

import copy
from typing import List, Optional, Iterable,Dict,Tuple

import cmath as c
import numpy as np
import numpy.typing as npt
import scipy as sc

import math
from scipy import optimize as scopt
import picos


from qrem.common import utils, convert, probability, math as qmath
# TODO_PP >> Talk with Paweł about import shortcut convention
from qrem.common.math import Constants as Const
from qrem.common.constants import SIGNIFICANT_DIGITS_POVM
from qrem.common.math import identity_check
import  qrem.common.registers as qregisters
from qrem.common.tomography import tomography
from qrem.qtypes.qunatum_detector_tomography_data import QDTCalibrationSetup
import itertools
from tqdm import tqdm 




from qrem.common.scipy_optimizers_wrappers import \
    wrapper_scipy_optimize as scop_wrapper



try:
    import qiskit
except(ModuleNotFoundError):
    pass


threshold = 10 ** (-SIGNIFICANT_DIGITS_POVM)

# =======================================================================================
# POVM - helpers
# =======================================================================================

def check_if_projector_is_in_computational_basis(projector: npt.ArrayLike, d=2):
    """
    Check if a given projector is in the computational basis.

    Parameters
    ----------
    projector : npt.ArrayLike
        The projector to be checked.
    d : int, optional
        The dimension of the computational basis, default is 2.

    Returns
    -------
    bool
        Returns True if the projector is in the computational basis, False otherwise.
    """
        
    n = int(math.log(np.shape(projector)[0], d))
    computational_basis_projectors = computational_projectors(d, n)

    for base_projector in computational_basis_projectors:
        projectors_difference = base_projector - projector  # independent from global phase
        norm = np.linalg.norm(projectors_difference)
        if abs(norm) < threshold:
            return True
    return False

def computational_projectors(d, n=1):
    """
    Generate projectors for the computational basis.

    Parameters
    ----------
    d : int
        The dimension of the computational basis.
    n : int, optional
        The number of qubits, default is 1.

    Returns
    -------
    list
        A list of projectors for each state in the computational basis.
    """    
    return [qmath.get_density_matrix(computational_basis(d, n)[i]) for i in range(d ** n)]

def computational_basis(d, n=1):
    m_d = d ** n
    eye = np.eye(m_d)
    return [np.array(eye[:, i]).reshape(m_d, 1) for i in range(m_d)]

def euler_angles_1q(unitary_matrix):
    """
    Compute Euler angles for a single-qubit unitary operation.

    Determines the Euler angles (theta, phi, lambda) for the given 2x2 unitary matrix.
    The unitary matrix is decomposed as U = phase * Rz(phi) * Ry(theta) * Rz(lambda), 
    following the SU(2) parameterization.

    Parameters
    ----------
    unitary_matrix : np.ndarray
        A 2x2 unitary matrix representing a single-qubit operation.

    Returns
    -------
    tuple
        A tuple (theta, phi, lambda) representing the Euler angles.

    Raises
    ------
    ValueError
        If the input matrix is not 2x2.

    Notes
    -----
    This function uses a slightly modified version of a Qiskit implementation and has a cutoff precision of 10^(-7).
    """

    _CUTOFF_PRECISION = 10 ** (-7)

    """Compute Euler angles for single-qubit gate.

    Find angles (theta, phi, lambda) such that
    unitary_matrix = phase * Rz(phi) * Ry(theta) * Rz(lambda)

    Args:
        unitary_matrix (ndarray): 2x2 unitary matrix

    Returns:
        tuple: (theta, phi, lambda) Euler angles of SU(2)

    Raises:
        QiskitError: if unitary_matrix not 2x2, or failure
    """
    if unitary_matrix.shape != (2, 2):
        raise ValueError("euler_angles_1q: expected 2x2 matrix")

    import scipy.linalg as la
    phase = la.det(unitary_matrix) ** (-1.0 / 2.0)
    U = phase * unitary_matrix  # U in SU(2)
    # OpenQASM SU(2) parameterization:
    # U[0, 0] = exp(-i_index(phi+lambda)/2) * cos(theta/2)
    # U[0, 1] = -exp(-i_index(phi-lambda)/2) * sin(theta/2)
    # U[1, 0] = exp(i_index(phi-lambda)/2) * sin(theta/2)
    # U[1, 1] = exp(i_index(phi+lambda)/2) * cos(theta/2)
    theta = 2 * math.atan2(abs(U[1, 0]), abs(U[0, 0]))

    # Find phi and lambda
    phiplambda = 2 * np.angle(U[1, 1])
    phimlambda = 2 * np.angle(U[1, 0])
    phi = (phiplambda + phimlambda) / 2.0
    lamb = (phiplambda - phimlambda) / 2.0

    # Check the solution
    Rzphi = np.array([[np.exp(-1j * phi / 2.0), 0],
                      [0, np.exp(1j * phi / 2.0)]], dtype=complex)
    Rytheta = np.array([[np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                        [np.sin(theta / 2.0), np.cos(theta / 2.0)]], dtype=complex)
    Rzlambda = np.array([[np.exp(-1j * lamb / 2.0), 0],
                         [0, np.exp(1j * lamb / 2.0)]], dtype=complex)
    V = np.dot(Rzphi, np.dot(Rytheta, Rzlambda))
    if la.norm(V - U) > _CUTOFF_PRECISION:
        raise ValueError("compiling.euler_angles_1q incorrect result norm(V-U)={}".
                         format(la.norm(V - U)))
    return theta, phi, lamb

def get_su2_parametrizing_angles(m_a):
    """
    Get the SU(2) parametrizing angles for a single-qubit unitary.

    Determines the parametrizing angles theta, phi, and lambda for a single-qubit 
    unitary matrix that belongs to the SU(2) group. Special cases for Pauli-X and 
    identity matrices are handled separately.

    Parameters
    ----------
    m_a : np.ndarray
        The single-qubit unitary matrix for which to compute the angles.

    Returns
    -------
    list
        A list containing the parametrizing angles ['theta', 'phi', 'lambda'] or 
        special identifiers ['x'] for Pauli-X and ['id'] for the identity matrix.

    Notes
    -----
    The function uses a threshold of 10^(-7) to round near-zero angles to zero and 
    adjusts the global phase of the matrix before decomposition.
    """    
    # Get three angles theta, phi, lambda which parametrize single-qubit unitary from SU(2)

    if qmath.is_pauli_x(m_a):
        return ['x']
    elif qmath.identity_check(m_a):
        return ['id']

    m_a = qmath.round_array_to_ndigits(m_a)

    eps_angles = 10 ** (-7)
    determinant = np.linalg.det(m_a)
    delta = c.phase(determinant) / 2
    m_a = c.exp(-1j * delta) * m_a

    decomposer = qiskit.quantum_info.synthesis.one_qubit_decompose.OneQubitEulerDecomposer()

    euler_theta_phi_lambda = decomposer.angles(m_a)

    angles = [euler_theta_phi_lambda[0], euler_theta_phi_lambda[1], euler_theta_phi_lambda[2]]

    for i in range(0, 3):
        if abs(angles[i]) < eps_angles:
            angles[i] = 0

    return angles

def get_unitary_change_ket_qubit(ket):
    """
    Prepare unitary transformation to change the initial state of qubits in the computational basis for a given probe state.

    This function assumes a perfect qubit initialization in the |0000...0> initial state and generates the unitary
    transformation needed to obtain the desired probe state from this initial state.

    Parameters
    ----------
    ket : np.ndarray
        The ket vector representing the desired probe state.

    Returns
    -------
    np.ndarray
        The unitary matrix that transforms the initial state into the desired probe state.

    Raises
    ------
    ValueError
        If the provided state is not in the computational basis or is an invalid state.
    """    
    state = qmath.get_density_matrix(ket)
    if check_if_projector_is_in_computational_basis(state):
        if state[0][0] == 1:
            return np.eye(2)
        elif state[1][1] == 1:
            return Const.pauli_sigmas()["X"]
        else:
            raise ValueError('error')
    else:
        U = np.zeros((2, 2), dtype=complex)
        U[:, 0] = ket[:, 0]
        ket_comp = np.array([[1], [0]]).reshape(2, 1)
        ket_perp = ket_comp - np.vdot(ket_comp, ket) * ket
        ket_perp = ket_perp / np.linalg.norm(ket_perp)
        U[:, 1] = ket_perp[:, 0]

        determinant = np.linalg.det(U)
        delta = c.phase(determinant) / 2

        U = c.exp(-1j * delta) * U

        return U


# =======================================================================================
# POVM - utils
# =======================================================================================

def get_offdiagonal_povm_part(povm: List[np.ndarray]) -> List[np.ndarray]:
    """
    Extract the off-diagonal part of each matrix in a POVM (Positive Operator-Valued Measure).

    This function iterates through each matrix (effect) in a given POVM, and extracts its off-diagonal part.
    The off-diagonal part of a matrix consists of all the elements that are not on the main diagonal.

    Parameters
    ----------
    povm : List[np.ndarray]
        A list of numpy ndarrays representing the effects of a POVM.

    Returns
    -------
    List[np.ndarray]
        A list of numpy ndarrays where each ndarray is the off-diagonal part of the corresponding POVM effect.
    """

    # implement function get_off_diagonal_from_matrix for each effect Mi in povm
    return [qmath.get_offdiagonal_of_matrix(Mi) for Mi in povm]

def get_diagonal_povm_part(povm: List[np.ndarray]) -> List[np.ndarray]:
    """
    Extract the diagonal part of each matrix in a POVM (Positive Operator-Valued Measure).

    Each matrix (effect) in the POVM is processed to obtain its diagonal part. 
    The diagonal part of a matrix is composed of the elements along its main diagonal.

    Parameters
    ----------
    povm : List[np.ndarray]
        A list of numpy ndarrays representing the effects of a POVM.

    Returns
    -------
    List[np.ndarray]
        A list of numpy ndarrays where each ndarray is a diagonal matrix 
        formed from the diagonal elements of the corresponding POVM effect.
    """

    # JT: np.diag returns a 1D list of diagonal elements, np.diagflat transforms this into 2D diagonal matrix

    return [np.diagflat(np.diag(effect)) for effect in povm]

def apply_stochastic_map_to_povm(povm: List[np.ndarray],
                                 stochastic_map: np.ndarray) -> List[np.ndarray]:
    """
    Apply a stochastic map to each effect in a POVM.

    This function computes the transformation of each POVM effect under a given stochastic map.
    A stochastic map is a square matrix where each column represents a probability distribution.
    The transformation is carried out by matrix multiplication of the stochastic map with each POVM effect.

    Parameters
    ----------
    povm : List[np.ndarray]
        A list of numpy ndarrays representing the effects of a POVM.
    stochastic_map : np.ndarray
        A numpy ndarray representing the stochastic map.

    Returns
    -------
    List[np.ndarray]
        A list of numpy ndarrays representing the transformed POVM effects under the given stochastic map.
    """    
    number_of_outcomes = len(povm)

    return [sum([stochastic_map[i, j] * povm[j] for j in range(number_of_outcomes)]) for i in range(number_of_outcomes)]

def get_stochastic_map_from_povm(povm: List[np.ndarray]) -> np.ndarray:
    """
    Generate a stochastic map from the diagonal parts of a POVM.

    This function constructs a stochastic map based on the diagonal parts of each POVM effect.
    The diagonal elements of each effect are used to form the rows of the stochastic map.
    A stochastic map is a square matrix where each column represents a probability distribution.

    Parameters
    ----------
    povm : List[np.ndarray]
        A list of numpy ndarrays representing the effects of a POVM.

    Returns
    -------
    np.ndarray
        A numpy ndarray representing the stochastic map constructed from the POVM.
    """

    # JT: diagonal elements of a POVM are taken, result is a list of diagonal effects

    diagonal_parts = get_diagonal_povm_part(povm)

    # JT: an empty noise matix is created

    noise_matrix = np.zeros((len(povm), len(povm)), dtype=float)

    for column_index in range(len(povm)):
        # print(diagonal_parts[column_index])

        # JT: a row corresponding to diagonal of a noise matrix is set

        noise_matrix[column_index, :] = np.diag(diagonal_parts[column_index])[:].real

    return noise_matrix

def get_povm_from_stochastic_map(stochastic_map: np.ndarray) -> List[np.ndarray]:
    """
    Construct a list of diagonal matrices (POVMs) from a stochastic map.

    Parameters
    ----------
    stochastic_map : np.ndarray
        A square numpy array representing a stochastic map.

    Returns
    -------
    List[np.ndarray] 
        A list of diagonal numpy arrays, each derived from the rows of the input stochastic map.

    Notes
    -----
    This function generates a list of Positive Operator-Valued Measures (POVMs) by diagonalizing each row of the
    input stochastic map. 
    """    

    return [np.diagflat(stochastic_map[i, :]) for i in range(stochastic_map.shape[0])]

def establish_POVMs_dictionary_from_noise_matrices_dictionary(noise_matrices_dictionary:Dict[Tuple[int],np.array]) -> Dict[Tuple[int],List[np.ndarray]]:
    """
    A wrapaer for get_povm_from_stochastic_map
    """  
    POVMs_dictionary = {}
    
    for subset, noise_matrix in noise_matrices_dictionary.items():
    
        POVMs_dictionary[subset] = get_povm_from_stochastic_map(noise_matrix['averaged'])

    return POVMs_dictionary 



def get_coherent_part_of_the_noise(povm: List[np.ndarray]) -> List[np.ndarray]:
    """
    Extract the off-diagonal elements from each matrix in a list of POVMs.

    Parameters
    ----------
    povm : List[np.ndarray]
        A list of numpy arrays, each representing a POVM effect matrix.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays, each containing the off-diagonal part of the corresponding POVM effect.

    Notes
    -----
    This function processes a list of POVM (Positive Operator-Valued Measure) effect matrices by extracting the
    off-diagonal elements of each matrix. These elements represent the coherent part of the noise in quantum
    information theory, which is important for understanding the non-classical aspects of quantum measurements.
    """
    return get_offdiagonal_povm_part(povm)


#-------------------------------------
# MOVE_TO >> common.registers or common.bitstrings// DUPLICATE - remove less relevant one
#-------------------------------------
#TODO LOW why not use enumerated_dict function from core /ancillaryfunctions?
def get_enumerated_rev_map_from_indices(indices):
    """
    Create a reverse mapping from a list of indices.

    Given a list of indices, this function creates a dictionary that maps each index to its position in the list.

    Parameters
    ----------
    indices : list
        A list of indices to be mapped.

    Returns
    -------
    dict
        A dictionary where keys are indices from the input list and values are their corresponding positions.
    """
    # TODO: move this function somewhere else
    enumerated_dict = dict(enumerate(indices))
    rev_map = {}
    for k, v in enumerated_dict.items():
        rev_map[v] = k
    return rev_map

# MOVE_TO >> common.bitstrings or common.bitstrings// DUPLICATE
def get_classical_register_bitstrings(  qubit_indices: List[int],
                            quantum_register_size: Optional[int] = None,
                            rev: Optional[bool] = False):
    """
    Generate register names for specified qubits. Non specified qubits are excluded from the register names (they will be always 0 in bitstring).
    Qubits indices are indexed from right if rev is False.

    This function creates names for qubit registers based on the qubits involved and the desired order. It excludes
    qubits not in use and can reverse the order of the names.

    Parameters
    ----------
    qubit_indices : list
        List of qubits to be included in the register names.
    quantum_register_size : int
        The total number of qubits in the register.
    rev : bool, optional
        If True, the order of qubits in the register names is reversed (default is False).

    Returns
    -------
    list
        A list of register names as strings, taking into account the specified qubits and order.
    """    
   # again assumes that qubit_indices contains unique values
    if quantum_register_size is None:
        quantum_register_size = len(qubit_indices)

    if quantum_register_size == 0:
        return ['']

    if quantum_register_size == 1:
        return ['0', '1']

    all_bitstrings= qregisters.all_possible_bitstrings_of_length(quantum_register_size, rev)
    not_used = []

    for j in list(range(quantum_register_size)):
        if j not in qubit_indices:
            not_used.append(j)

    bad_names = []
    for bitstring in all_bitstrings: #0000111
        for k in (not_used):
            rev_name = bitstring[::-1] #1110000 reverses order of string - why?
            if rev_name[k] == '1':
                bad_names.append(bitstring)

    relevant_names = []
    for bitstring in all_bitstrings:
        if bitstring not in bad_names:
            relevant_names.append(bitstring)

    return relevant_names




def get_CBT_norm(J, n, m, rev=False):
    import cvxopt as cvx
    import picos as pic
    # Get completely bounded trace norm of Choi-matrix J representing quantum channel from number_of_qubits-dimensional space to m-dimensional space
    J = cvx.matrix(J)
    prob = pic.Problem(verbose=0)
    X = prob.add_variable("X", (n * m, n * m), vtype='complex')

    I = pic.new_param('I', np.eye(m))

    rho0 = prob.add_variable("rho0", (n, n), vtype='hermitian')
    rho1 = prob.add_variable("rho1", (n, n), vtype='hermitian')
    prob.add_constraint(rho0 >> 0)
    prob.add_constraint(rho1 >> 0)

    prob.add_constraint(pic.trace(rho0) == 1)
    prob.add_constraint(pic.trace(rho1) == 1)

    if (rev == True):
        # FBM: tests which conention is good.
        # FBM: add reference to paper

        # This is convention REVERSED with respect to the paper,
        # and seems that this is a proper one????
        C0 = pic.kron(rho0, I)
        C1 = pic.kron(rho1, I)
    else:
        C0 = pic.kron(I, rho0)
        C1 = pic.kron(I, rho1)

    F = pic.trace((J.H) * X) + pic.trace(J * (X.H))

    prob.add_constraint(((C0 & X) // (X.H & C1)) >> 0)

    prob.set_objective('max', F)

    prob.solve(verbose=0)

    if prob.status.count("optimal") > 0:
        #        print('solution optimal')
        1
    elif (prob.status.count("optimal") == 0):
        print('uknown_if_solution_optimal')

    else:
        print('solution not found')

    cbt_norm = prob.obj_value() / 2

    if (abs(np.imag(cbt_norm)) >= 0.00001):
        raise ValueError
    else:
        cbt_norm = np.real(cbt_norm)

    return cbt_norm


def get_choi_from_POVM(POVM):
    # get Choi matrix of POVM channel
    dimension = POVM[0].shape[0]
    number_of_outcomes = len(POVM)
    choi_matrix = np.zeros((number_of_outcomes * dimension, number_of_outcomes * dimension), dtype=complex)

    for index_outcome in range(number_of_outcomes):
        choi_matrix[index_outcome * dimension:
                    (index_outcome + 1) * dimension, index_outcome * dimension:(index_outcome + 1) * dimension] \
            = (POVM[index_outcome].T)[:, :]

    return choi_matrix / dimension


def get_POVM_from_choi(povm_choi,
                       number_of_outcomes,
                       dimension):
    povm = []

    povm_choi *= dimension
    for index_outcome in range(number_of_outcomes):
        block_now = povm_choi[index_outcome * dimension:
                              (index_outcome + 1) * dimension,
                    index_outcome * dimension:(index_outcome + 1) * dimension] \
        #\
        povm.append(block_now.T)
        # = (POVM[index_outcome].T)[:, :]

    return povm




def check_if_povm(povm):
    for Mi in povm:
        eigvals = np.round(sc.linalg.eigvalsh(Mi), 10)

        if any(np.array(eigvals) < 0):
            return False

    return identity_check(sum(povm), significant_digits=10)


def get_statistical_error_bound(number_of_measurement_outcomes: int,
                                number_of_samples: int,
                                statistical_error_mistake_probability: float,
                                number_of_marginals=1) -> float:
    """
    Description:
        Get upper bound for tv-distance of estimated probability distribution from ideal one. See Ref. [3] for
        details.

    Parameters:
        :param number_of_measurement_outcomes: Number of outcomes in probabiility distribution (2^(number_of_qubits) for standard measurement)
        :param number_of_samples: Number of samples for experiment for which statistical error bound is being calculated.
        :param statistical_error_mistake_probability: Parameter describing infidelity of returned error bound.

    Return:
        Statistical error upper bound in total variance distance.
    """

    if number_of_marginals == 0:
        number_of_marginals = 1

    if number_of_measurement_outcomes < 16:
        # for small number of outcomes "-2" factor is not negligible
        return np.sqrt(
            (np.log(2 ** number_of_measurement_outcomes - 2)
             - np.log(statistical_error_mistake_probability) + np.log(
                        number_of_marginals)) / 2 / number_of_samples
        )
        # for high number of outcomes "-2" factor is negligible
    else:
        return np.sqrt(
            (number_of_measurement_outcomes * np.log(2) - np.log(
                statistical_error_mistake_probability) + np.log(
                number_of_marginals)) / 2 / number_of_samples
        )


def get_coherent_error_bound(povm: np.ndarray) -> float:
    """
    Description:
        Get distance between diagonal part of the POVM and the whole POVM. This quantity might be interpreted as a
        measure of "non-classicality" or coherence present in measurement noise. See Ref. [1] for details.

    Parameters:
        :param povm: POVM for which non-classicality will be determined.
    Return:
        Coherent error bound for given POVM.
    """

    return operational_distance_POVMs(povm, get_diagonal_povm_part(povm))

def get_correction_error_bound_from_data_and_statistical_error(povm: List[np.ndarray],
                                                               correction_matrix: np.ndarray,
                                                               statistical_error_bound: float,
                                                               alpha: float = 0) -> float:
    """
        Description:
            Get upper bound for the correction error using classical error-mitigation via "correction matrix".

            Error arises from three factors - non-classical part of the noise, statistical fluctuations and eventual
            unphysical "first-guess" (quasi-)probability vector after the correction.

            This upper bound tells us quantitatively what is the maximal TV-distance of the corrected probability vector
            from the ideal probability distribution that one would have obtained if there were no noise and the
            infinite-size statistics.

            See Ref. [1] for details.

        Parameters:
            :param povm: POVM representing measurement device.
            :param correction_matrix: Correction matrix obtained via out Error Mitigator object.
            :param statistical_error_bound: Statistical error bound (epsilon in paper).
            confidence with which we state the upper bound. See Ref. [3] for details.
            :param alpha: distance between eventual unphysical "first-guess" quasiprobability vector and the closest
            physical one. default is 0 (which corresponds to situation in which corrected vector was proper probability
            vector).


        Return:
            Upper bound for correction error.

        """

    norm_of_correction_matrix = np.linalg.norm(correction_matrix, ord=1)
    coherent_error_bound = get_coherent_error_bound(povm)
    return norm_of_correction_matrix * (coherent_error_bound + statistical_error_bound) + alpha


def get_correction_error_bound_from_data(povm: List[np.ndarray],
                                         correction_matrix: np.ndarray,
                                         number_of_samples: int,
                                         error_probability: float,
                                         alpha: float = 0) -> float:
    """
    Description:
        Get upper bound for the correction error using classical error-mitigation via "correction matrix".

        Error arises from three factors - non-classical part of the noise, statistical fluctuations and eventual
        unphysical "first-guess" (quasi-)probability vector after the correction.

        This upper bound tells us quantitatively what is the maximal TV-distance of the corrected probability vector
        from the ideal probability distribution that one would have obtained if there were no noise and the
        infinite-size statistics.

        See Ref. [0] for details.

    Parameters:
        :param povm: POVM representing measurement device.
        :param correction_matrix: Correction matrix obtained via out Error Mitigator object.
        :param number_of_samples: number of samples (in qiskit language number of "shots").
        :param error_probability: probability with which statistical upper bound is not correct. In other word, 1-mu is
        confidence with which we state the upper bound. See Ref. [3] for details.
        :param alpha: distance between eventual unphysical "first-guess" quasiprobability vector and the closest
        physical one. default is 0 (which corresponds to situation in which corrected vector was proper probability
        vector).


    Return:
        Upper bound for correction error.

    """
    dimension = povm[0].shape[0]

    norm_of_correction_matrix = np.linalg.norm(correction_matrix, ord=1)

    statistical_error_bound = get_statistical_error_bound(dimension, number_of_samples,
                                                          error_probability)
    coherent_error_bound = get_coherent_error_bound(povm)

    return norm_of_correction_matrix * (coherent_error_bound + statistical_error_bound) + alpha


def get_correction_error_bound_from_parameters(norm_of_correction_matrix: float,
                                               statistical_error_bound: float,
                                               coherent_error_bound: float,
                                               alpha: float = 0) -> float:
    """
    Description:
        See description of function "get_correction_error_bound_from_data". This function can be used if one has the
        proper parameters already calculated and wishes to not repeat it (for example, in case of calculating something
        in the loop).

    Parameters:
        :param norm_of_correction_matrix : 1->1 norm of correction matrix (it is not trace norm!), see Ref. [0],
        or definition of np.linalg.norm(X,ord=1)
        :param statistical_error_bound: upper bound for statistical errors. Can be calculated using function
        get_statistical_error_bound.
        :param coherent_error_bound: magnitude of coherent part of the noise. Can be calculated using function
        get_coherent_error_bound.
        :param alpha: distance between eventual unphysical "first-guess" quasi-probability vector and the closest
        physical one. default is 0 (which corresponds to situation in which corrected vector was proper probability
        vector)

    Return:
        Upper bound for correction error.
    """

    return norm_of_correction_matrix * (coherent_error_bound + statistical_error_bound) + alpha


# ORGANIZE: This manipulates results data and should probably fit better in functions_data_analysis
def counts_dict_to_frequencies_vector(count_dict: dict, reverse_order=False) -> list:
    """
    Description:
        Generates and returns vector of frequencies basing on given counts dict. Mostly used with qiskit data.
    :param count_dict: Counts dict. Possibly from qiskit job.
    :return frequencies: Frequencies list for possible states in ascending order.
    """

    frequencies = []

    qubits_number = len(list(count_dict.keys())[0])  # Number of qubits in given experiment counts.
    possible_outcomes = get_possible_n_qubit_outcomes(qubits_number)
    dict_keys = count_dict.keys()  # So we don't call the method_name every time.
    counts_sum = 0

    for outcome in possible_outcomes:
        if dict_keys.__contains__(outcome):
            frequencies.append(count_dict[outcome])
            counts_sum += count_dict[outcome]
        else:
            frequencies.append(0)

    for i in range(len(frequencies)):
        frequencies[i] = frequencies[i] / counts_sum

    if reverse_order:
        return convert.reorder_probabilities(frequencies, range(qubits_number)[::-1])
    else:
        return frequencies


def get_possible_n_qubit_outcomes(n: int) -> list:
    """
    Description:
        For given number of qubits <n> generates a list of possible outcome states (as strings) and returns them in
        ascending order. All states len is number_of_qubits.
    :param n: Number of qubits.
    :return: List of possible outcomes as strings.
    """
    max_value = pow(2, n)
    possible_outcomes = []

    for i in range(max_value):
        possible_outcomes.append(bin(i)[2:].zfill(n))

    return possible_outcomes


def get_correction_matrix_from_povm(povm):
    number_of_povm_outcomes = len(povm)
    dimension = povm[0].shape[0]

    transition_matrix = np.zeros((number_of_povm_outcomes, number_of_povm_outcomes), dtype=float)

    for k in range(number_of_povm_outcomes):
        current_povm_effect = povm[k]

        # Get diagonal part of the effect. Here we remove eventual 0 imaginary part to avoid format conflicts
        # (diagonal elements of Hermitian matrices are real).
        vec_p = np.array([np.real(current_povm_effect[i, i]) for i in range(dimension)])

        # Add vector to transition matrix.
        transition_matrix[k, :] = vec_p[:]

    return np.linalg.inv(transition_matrix)


def get_averaged_lambda(big_lambda,
                        bits_of_interest):
    big_N = int(np.log2(big_lambda.shape[0]))
    small_N = len(bits_of_interest)
    normalization = 2 ** (big_N - small_N)

    classical_register_big = ["{0:b}".format(i).zfill(big_N) for i in range(2 ** big_N)]
    classical_register_small = ["{0:b}".format(i).zfill(small_N) for i in range(2 ** small_N)]

    indices_small = {}

    for s in classical_register_big:
        small_string = ''.join([list(s)[b] for b in bits_of_interest])
        indices_small[s] = small_string

    small_lambda = np.zeros((2 ** small_N, 2 ** small_N))

    for i in range(2 ** big_N):
        for j in range(2 ** big_N):
            lambda_element = big_lambda[i, j]
            ideal_state = "{0:b}".format(j).zfill(big_N)
            measured_state = "{0:b}".format(i).zfill(big_N)

            ideal_state_small = indices_small[ideal_state]
            measured_state_small = indices_small[measured_state]

            small_lambda[int(measured_state_small, 2), int(ideal_state_small,
                                                           2)] += lambda_element / normalization
    return small_lambda



# REMOVE:
# ---------------------------------------------------------------
# def get_tensored_POVM(POVMS_list):
#     from qrem.functions_qrem.PyMaLi import \
#         GeneralTensorCalculator  # old version: from PyMaLi import GeneralTensorCalculator
#     gtc = GeneralTensorCalculator()
#     return gtc.calculate_tensor_to_increasing_list(objects=POVMS_list)


# def get_unitary_change_state(state):
#     if check_if_projector_is_in_computational_basis(state):
#         if state[0][0] == 1:
#             return np.eye(2)
#         elif state[1][1] == 1:
#             return anf.pauli_sigmas['X']
#         else:
#             raise ValueError('error')
#     else:
#         m_a, m_u = spectral_decomposition(state)

#         d = m_u.shape[0]
#         determinant = np.linalg.det(m_a)

#         delta = c.phase(determinant) / d

#         m_u = c.exp(-1j * delta) * m_u

#         return np.matrix.getH(m_u)



############################# moved from function_distances 

"""
Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""





# =======================================================================================
# POVM - functions used to compute distances between POVMs
# =======================================================================================




__worst_case_distance_names__ = ['operational', 'worst', 'worst_case', 'worst-case', 'wc']
__average_case_distance_names__ = ['average', 'average_case', 'average-case', 'ac']





def operational_distance_POVMs(POVM1,
                               POVM2=None,
                               method='direct',
                               classical=False):

    #JT: computation of classical operational distance is performed

    if classical:
        noise_map_M, noise_map_P = get_stochastic_map_from_povm(
            POVM1), get_stochastic_map_from_povm(POVM2)
        return 1 / 2 * np.linalg.norm(noise_map_M - noise_map_P, ord=1)

    #JT: calculation of quantum worst case distance starts

    #JT: number of effects of POVM! is set to m

    m = len(POVM1)

    #JT: If POVM2 is not specified it is set to idel computational basis projective measurement

    if POVM2 is None:

        difference = []
        for i in range(m):
            diff_now = copy.deepcopy(POVM1[i])
            diff_now[i, i] -= 1
            difference.append(diff_now)
        difference = (difference)

    #JT: If POVM2 is speciffied differnce in effects is computed

    else:

        difference = ([POVM1[i] - POVM2[i] for i in range(m)])


    #JT: for CBTN computation is performed as below


    if (method == 'CBTN'):

        n = POVM1[0].shape[0]

        #JT: Choi-Jamiolkowski state corresponding to difference of POVMs is computed
        J = get_choi_from_POVM(difference)

        #JT: old comment below:
        # this calculates completely bounded trace norm of the channel which is the upper bound for operational distance
        cbt_norm = get_CBT_norm(J, n, m)

        return cbt_norm / 2


    #JT: defult computation method

    elif (method == 'direct'):
        # calculate operational distance directly via bruteforce search over subsets_list of outcomes
        biggest_norm = 0

        #JT: k goes from number of effects m to 0

        for k in list(range(m))[::-1]:

            #JT: combinations of k elements from all effects diferrences are created
            #k starts from number of effects

            current_list = list(itertools.combinations(difference, k + 1))

            #JT: tis loop goes over all elements forming a particulat combination

            for l in current_list:
                current_sum = sum(l)

                #JT: Norm of sum of differences is computed

                current_norm = np.linalg.norm(current_sum, ord=2)

                #JT: a check whether the currectly computed norm is higher than the previous ones

                if (current_norm > biggest_norm):
                    biggest_norm = current_norm

        return biggest_norm


#JT: This function is used to compute average case distance between POVMs

def average_distance_POVMs(POVM1,
                           POVM2=None,
                           classical=False):
    """
    Description:
         Get average (over all quantum states) operational distance between two povms
         COMMENT: Assuming that both POVMs have the same number of outcomes
    Parameter:
        :param POVM1: list of arrays representing effects of POVM
        :param POVM2: list of arrays representing effects of POVM, if None assuming it's computational basis
    Return:
        average (over all quantum states) operational distance between two povms
    """

    #JT: classical is a boolean flag determining if av case calassical distance is computed

    if classical:

        #JT:Stochastic noise acting on POVM1 and POVM2 are established

        noise_map_M, noise_map_P = get_stochastic_map_from_povm(
            POVM1), get_stochastic_map_from_povm(POVM2)

        #JT: average case distance is computed and its value is returned

        return 1 / 2 * np.sqrt(
            1 / 2 * np.linalg.norm(noise_map_M-noise_map_P, ord='fro') ** 2 + np.trace(
                noise_map_M - noise_map_P) ** 2)

    #JT: calculatioof quantum distance starts

    #JT: number of rows of POVM1 is set as dimension

    dimension = POVM1[0].shape[0]

    #JT: number of effects is set as n

    n = len(POVM1)

    dist = 0

    #JT a loop over effects

    for i in range(n):

        #A difference between POVMs effects is computed

        if POVM2 is not None:
            Di = POVM1[i] - POVM2[i]

        #JT: If POVM2 is not specified at input it is assumed that it corresponds to computational basis measurement
        #the difference between effects is then computed by subtracting 1 from diagonal element of POVM1

        else:
            # assuming comp basis
            Di = copy.deepcopy(POVM1[i])
            Di[i, i] -= 1
        # print(Di)

        #JT:Value of distace is updated

        dist += np.sqrt((np.trace(Di) ** 2).real + (np.trace(Di @ Di)).real)


    #

    # Factor 1/2 is okay
    return 1 / dimension * dist / 2


#JT: This function is used to calculate distances between POVMs
#Usage 1: Computation between reconstructed POVMs and thier ideal projective measurements counterparts

def calculate_distance_between_POVMs(POVM_1,
                                     POVM_2,
                                     distance_type_tuple=('average_case', 'classical')):
    distance_type = distance_type_tuple[0]
    errors_type = distance_type_tuple[1]

    classical_errors = False
    if errors_type.lower() in ['classical']:
        classical_errors = True

    #JT: __average_case_distance_names__ - a constant holding  references to average case distance
    #Here computation of av case distance is performed

    if distance_type in __average_case_distance_names__:
        distance_calculated = average_distance_POVMs(POVM1=POVM_1,
                                                     POVM2=POVM_2,
                                                     classical=classical_errors)


    #JT:  __worst_case_distance_names__ - a constant holding  references to worst case distance
    #Here computation of wc case distance is performed

    elif distance_type in __worst_case_distance_names__:
        distance_calculated = operational_distance_POVMs(POVM1=POVM_1,
                                                         POVM2=POVM_2,
                                                         classical=classical_errors)

    #JT: error message when wrong parameters are passed

    else:
        raise ValueError(f"Wrong distance type: {distance_type_tuple}")

    return distance_calculated




# =======================================================================================
# POVM - computation of correlation coefficients 
# =======================================================================================

def __solve_subproblem_average_case(povm_2q_input,
                                    signs_bitstring,
                                    target_qubit
                                    ):
    signs = [(-1) ** int(s) for s in list(signs_bitstring)]

    povm_2q = __swap_or_not(povm_2q_input=povm_2q_input,
                            target_qubit=target_qubit)

    problem = picos.Problem()

    local_difference = picos.HermitianVariable(name='local_difference',
                                               shape=(2, 2))
    problem.add_constraint(picos.trace(local_difference) == 0)
    problem.add_constraint(local_difference << np.eye(2))
    problem.add_constraint(local_difference >> -np.eye(2))

    Madded = povm_2q[0] + povm_2q[1]

    # Madded2 = povm_2q[2]+povm_2q[3]

    # raise KeyboardInterrupt

    M00 = Madded[0:2, 0:2]
    M01 = Madded[0:2, 2:4]
    M10 = Madded[2:4, 0:2]
    M11 = Madded[2:4, 2:4]

    # Ms = [M00, M01, M11]

    # first_part = picos.sum([picos.trace(local_difference*Mi)**2 for Mi in Ms])

    t00 = signs[0] * picos.trace(local_difference * M00)
    t01 = signs[1] * picos.trace(local_difference * (M01 + M10))
    t11 = signs[2] * picos.trace(local_difference * M11)

    t_add = signs[3] * picos.trace((M00 + M11) * local_difference)
    cost_function = t00 + t01 + t11 + t_add

   
    problem.set_objective(direction='max',
                          expression=cost_function)

 
    problem.solve(solver='mosek')



    dav = 1 / 2 * np.sqrt(
        t00.coherences_values ** 2 + t01.coherences_values ** 2 + t11.coherences_values ** 2 + t_add.coherences_values ** 2)

    return dav.real


def calculate_correlations_coefficients_average_case(povm_2q):
    full_register = get_classical_register_bitstrings(qubit_indices=range(4),
                                                          quantum_register_size=4,
                                                          )

    solutions_list_i_j, solutions_list_j_i = [], []
    for signs_bitstring in full_register:
        dav_now_i_j = __solve_subproblem_average_case(povm_2q_input=povm_2q,
                                                      target_qubit=0,
                                                      signs_bitstring=signs_bitstring)
        solutions_list_i_j.append(dav_now_i_j)

        dav_now_j_i = __solve_subproblem_average_case(povm_2q_input=povm_2q,
                                                      target_qubit=1,
                                                      signs_bitstring=signs_bitstring)
        solutions_list_j_i.append(dav_now_j_i)

    # how "i" is affected by "j"
    c_i_j = np.max(solutions_list_i_j)

    # how "j is affected by "i"
    c_j_i = np.max(solutions_list_j_i)

    # print(solutions_list_i_j)
    # print(solutions_list_j_i)

    return c_i_j, c_j_i


def _solve_max_dop_problem_for_fixed_input_state(input_state,
                                                 povm_2q,
                                                 sign):
    # print('hejunia')
    problem = picos.Problem()

    local_difference = picos.HermitianVariable(name='local_difference',
                                               shape=(2, 2))
    problem.add_constraint(picos.trace(local_difference) == 0)
    problem.add_constraint(local_difference << np.eye(2))
    problem.add_constraint(local_difference >> -np.eye(2))

    embedded_difference = picos.kron(np.eye(2), local_difference)

    local_effect = picos.partial_trace((povm_2q[0] + povm_2q[1]) * embedded_difference,
                                       subsystems=1,
                                       dimensions=2)

    cost_function = sign * picos.trace(input_state * local_effect)

    problem.set_objective(direction='max',
                          expression=cost_function)

    # print(problem)

    problem.solve(solver='mosek')

    dop = cost_function.coherences_values

    return dop.real







def _cost_function_difference_dac(parameters_difference,
                                  Ms_list):
    nx, ny, nz = parameters_difference[0], parameters_difference[1], parameters_difference[2]
    local_difference = nx * Const.pauli_sigmas()['X'] + ny * Const.pauli_sigmas()['Y'] + nz * Const.pauli_sigmas()[
        'Z']

    M00, M01, M10, M11 = Ms_list
    a = np.trace(local_difference @ M00).real
    b = np.trace(local_difference @ M11).real
    c = np.trace(M01 @ local_difference)
    d = np.trace(M10 @ local_difference)

    

    trace_part = a + b

    norm_squared = a ** 2 + b ** 2 + c ** 2 + d ** 2 + trace_part ** 2
    norm_squared = norm_squared.real
 

    return -norm_squared


def _cost_function_difference_dwc(parameters_difference,
                                  Ms_list):
    nx, ny, nz = parameters_difference[0], parameters_difference[1], parameters_difference[2]

    # norm_check = nx**2+ny ** 2 + nz ** 2
    #
    # if norm_check>1.0:
    #     raise ValueError("damn")
    #     return 100*norm_check

    local_difference = nx * Const.pauli_sigmas()['X'] + ny * Const.pauli_sigmas()['Y'] + nz * Const.pauli_sigmas()[
        'Z']

    M00, M01, M10, M11 = Ms_list
    a00 = np.trace(local_difference @ M00).real
    a01 = np.trace(local_difference @ M01)
    # a10 = np.conj(a01)
    a10 = np.trace(local_difference @ M10)
    # np.trace(local_difference @ (M01 + M10)).real
    a11 = np.trace(local_difference @ M11).real
    #
    # det = a00*a10-a01*a10
    # mean = (a00+a11)/2
    #
    # second_part = np.sqrt(mean**2-det)

    operator_norm = np.linalg.norm(np.array([[a00, a01],
                                             [a10, a11]]),
                                   ord=2).real

    return -operator_norm


#
# def _direct_optimization_dac(povm2q,
#                              target_qubit,
#                              printing=False):
def direct_optimization_difference(povm2q,
                                   target_qubit,
                                   distance_type,
                                   printing=False):
    povm_2q = __swap_or_not(povm_2q_input=povm2q,
                            target_qubit=target_qubit)

    Madded = povm_2q[0] + povm_2q[1]
    M00 = Madded[0:2, 0:2]
    M01 = Madded[0:2, 2:4]
    M10 = Madded[2:4, 0:2]
    M11 = Madded[2:4, 2:4]

    Ms_list = (M00, M01, M10, M11)

    if distance_type.lower() in ['average', 'average-case', 'ac']:
        cost_function_now = _cost_function_difference_dac
    elif distance_type.lower() in ['worst', 'worst-case', 'wc']:
        cost_function_now = _cost_function_difference_dwc
    else:
        raise ValueError(f"Unsuported distance type: {distance_type}")

    additional_arguments = (Ms_list,
                            )

    bounds_list = [(-1, 1) for _ in range(3)]

    ball = lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2
    constraints = scopt.NonlinearConstraint(fun=ball, lb=0, ub=1)
    optimizer_inital = 'differential_evolution'

    maxiter_initial = 10 * 100 ** 1

    params_starting = [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]

    differential_evolution_options_initial = {
        'maxiter': maxiter_initial,
        'popsize': 24,
        'workers': 1,
        'updating': 'deferred',
        # 'constraints':constraints,
        # 'updating': 'immediate',
        'disp': printing
    }
    # print('ok1')

    # print('ok')

    initial_kwargs = {
        'differential_evolution_kwargs': differential_evolution_options_initial}

    # if optimizer_inital.upper() in ['DIFFERENTIAL_EVOLUTION']:
    #     basinhopping = False
    # else:
    #     basinhopping = True

    # qprint("running differential evolution")
    res = scop_wrapper(
        initial_parameters=params_starting,
        target_function=cost_function_now,
        optimizer_name=optimizer_inital,
        # basinhopping=basinhopping,
        additional_arguments=additional_arguments,
        bounds_list=bounds_list,
        differential_evolution_constraints=constraints,
        # options={'maxiter': maxiter_initial},
        **initial_kwargs

        # differential_evolution_kwargs=differential_evolution_options_initial
    )
    best_xs = res.x
    best_funopt = res.fun

    # print(sum([x**2 for x in best_xs]))

    if distance_type.lower() in __average_case_distance_names__:
        distance = 1 / 2 * np.sqrt(-best_funopt)
    elif distance_type.lower() in __worst_case_distance_names__:
        distance = -best_funopt

    return distance


def _optimization_function_dwc(parameters_state,
                               povm_2q,
                               sign
                               # qubit_index=0
                               ):
    ny, nz = parameters_state[0], parameters_state[1]
    # print('here1')
    norm = ny ** 2 + nz ** 2

    # if nx<0 or ny<0:
    # return 1000*

    if norm > 1.0:
        return 100 * norm

    nx = np.sqrt(1 - norm)

    # half_nx, half_ny, half_nz = nx/2, ny/2, nz/2

    # print('here2')
    # print(nx,ny,nz)
    rho = 1 / 2 * np.eye(2, dtype=complex)
    rho[0, 1] = (nx - 1j * ny) / 2
    rho[1, 0] = (nx + 1j * ny) / 2
    rho[0, 0] += nz / 2
    rho[1, 1] -= nz / 2

    # print('hejunia2')

    return -_solve_max_dop_problem_for_fixed_input_state(input_state=rho,
                                                         povm_2q=povm_2q,
                                                         sign=sign)


def _find_maximal_dop_fixed_sign(povm2q,
                                 target_qubit,
                                 sign,
                                 printing=False):
    povm_2q = __swap_or_not(povm_2q_input=povm2q,
                            target_qubit=target_qubit)
    cost_function_now = _optimization_function_dwc
    additional_arguments = (povm_2q,
                            sign
                            )

    bounds_list = [(-1, 1) for _ in range(2)]

    # bounds_list = None
    optimizer_inital = 'differential_evolution'

    maxiter_initial = 10 * 10 ** 1

    params_starting = [0, 0]

    differential_evolution_options_initial = {
        'maxiter': maxiter_initial,
        'popsize': 24,
        'workers': -1,
        'updating': 'deferred',
        # 'updating': 'immediate',
        'disp': printing
    }
    # print('ok1')

    # print('ok')

    initial_kwargs = {
        'differential_evolution_kwargs': differential_evolution_options_initial}

    # if optimizer_inital.upper() in ['DIFFERENTIAL_EVOLUTION']:
    #     basinhopping = False
    # else:
    #     basinhopping = True

    # qprint("running differential evolution")
    res = scop_wrapper(
        initial_parameters=params_starting,
        target_function=cost_function_now,
        optimizer_name=optimizer_inital,
        # basinhopping=basinhopping,
        additional_arguments=additional_arguments,
        bounds_list=bounds_list,
        # options={'maxiter': maxiter_initial},
        **initial_kwargs
        # differential_evolution_kwargs=differential_evolution_options_initial
    )
    # qprint("GOT IT")
    best_xs = res.x
    best_funopt = res.fun


    return best_funopt


def find_maximal_dop(povm2q,
                     target_qubit):
    # print('hejunia4')
    dop_plus = -_find_maximal_dop_fixed_sign(povm2q=povm2q,
                                             target_qubit=target_qubit,
                                             sign=1)
    dop_minus = -_find_maximal_dop_fixed_sign(povm2q=povm2q,
                                              target_qubit=target_qubit,
                                              sign=-1)

    return np.max([dop_plus, dop_minus])


def get_reduced_noise_matrix_2q(stochastic_map2q,
                                target_qubit,
                                spectator_state):
    stochastic_map2q_output = copy.deepcopy(stochastic_map2q)
    if target_qubit == 1:
        stochastic_map2q_output = Const.standard_gates()['SWAP'].real @ stochastic_map2q_output @ \
                                  Const.standard_gates()['SWAP'].real

    # stochastic map convention
    # 00, 01, 10, 11  <- input state
    # 00----------------
    # 01
    # 10----------------
    # 11
    # ^
    # |
    # output state

    # forgot about convention here :-d
    stochastic_map2q_output = stochastic_map2q_output.T
    if spectator_state in [0, '0']:
        p_0_0 = (stochastic_map2q_output[0, 0] + stochastic_map2q_output[0, 1])
        p_0_1 = (stochastic_map2q_output[2, 0] + stochastic_map2q_output[2, 1])

    elif spectator_state in [1, '1']:
        p_0_0 = (stochastic_map2q_output[1, 0] + stochastic_map2q_output[1, 1])
        p_0_1 = (stochastic_map2q_output[3, 0] + stochastic_map2q_output[3, 1])

    else:
        raise ValueError(f"Wrong neighbors state: {spectator_state}")

    p_1_0 = 1 - p_0_0
    p_1_1 = 1 - p_0_1

    return np.array([[p_0_0, p_0_1],
                     [p_1_0, p_1_1]])


def get_correlation_coefficient_classical(stochastic_map_2q_or_povm,
                                          target_qubit,
                                          distance_type):
    if isinstance(stochastic_map_2q_or_povm, list):
        stochastic_map_2q_or_povm = get_stochastic_map_from_povm(stochastic_map_2q_or_povm)

    noise_maps = [get_reduced_noise_matrix_2q(stochastic_map2q=stochastic_map_2q_or_povm,
                                              target_qubit=target_qubit,
                                              spectator_state=i) for i in range(2)]

    povm_0, povm_1 = tuple(
        [get_povm_from_stochastic_map(stochastic_map=stochastic_map) for stochastic_map
         in noise_maps])

    if distance_type.lower() in __average_case_distance_names__:
        return average_distance_POVMs(povm_0, povm_1)
    elif distance_type.lower() in __worst_case_distance_names__:
        return operational_distance_POVMs(povm_0, povm_1, classical=True)


def find_correlations_coefficients(povm_2q,
                                   distance_type,
                                   classical=False,
                                   direct_optimization=True):
    if classical:
        c_i_j = get_correlation_coefficient_classical(stochastic_map_2q_or_povm=povm_2q,
                                                      target_qubit=0,
                                                      distance_type=distance_type)
        c_j_i = get_correlation_coefficient_classical(stochastic_map_2q_or_povm=povm_2q,
                                                      target_qubit=1,
                                                      distance_type=distance_type)




    else:
        if distance_type.lower() in __average_case_distance_names__:

            if direct_optimization:
                c_i_j = direct_optimization_difference(povm2q=povm_2q,
                                                       target_qubit=0,
                                                       distance_type='ac')
                c_j_i = direct_optimization_difference(povm2q=povm_2q,
                                                       target_qubit=1,
                                                       distance_type='ac')
            else:

                c_i_j, c_j_i = calculate_correlations_coefficients_average_case(povm_2q=povm_2q)

        elif distance_type.lower() in __worst_case_distance_names__:

            if direct_optimization:
                c_i_j = direct_optimization_difference(povm2q=povm_2q,
                                                       target_qubit=0,
                                                       distance_type='wc')
                c_j_i = direct_optimization_difference(povm2q=povm_2q,
                                                       target_qubit=1,
                                                       distance_type='wc')

            else:
                c_i_j = find_maximal_dop(povm2q=povm_2q,
                                         target_qubit=0)
                c_j_i = find_maximal_dop(povm2q=povm_2q,
                                         target_qubit=1)

        else:
            raise ValueError(f"Distance type '{distance_type}' incorrect.")

    return c_i_j, c_j_i


###depracated 

def __swap_2q_povm(povm_2q_input):
    povm_2q = [Const.standard_gates()['SWAP'] @ Mi @ Const.standard_gates()['SWAP'] for Mi in
               povm_2q_input]
    povm_2q[1], povm_2q[2] = povm_2q[2], povm_2q[1]

    return povm_2q


def __swap_or_not(povm_2q_input,
                  target_qubit=0):
    if target_qubit == 0:
        povm_2q = copy.deepcopy(povm_2q_input)

    elif target_qubit == 1:
        povm_2q = __swap_2q_povm(povm_2q_input=povm_2q_input)

    else:
        raise ValueError(f"Index of qubit '{target_qubit}' is incorrect, should be 0 or 1.")

    return povm_2q

def __construct_local_difference(parameters):
    nx, ny, nz = parameters

    local_difference = nx * Const.pauli_sigmas()['X'] + ny * Const.pauli_sigmas()['Y'] + nz * Const.pauli_sigmas()[
        'Z']
    return local_difference






#JT This method is used to compute POVMs once marginals are computed
def compute_subsets_POVMs_averaged( results_dictionary:Dict, 
                                    marginals_dictionary:Dict,
                                    subsets_of_qubits: List[List[int]],
                                    averaged_marginals_dictionary = None, 
                                    estimation_method:str='pls',
                                    show_progress_bar: Optional[bool] = True):

    POVMs_dictionary = {}
    
    #Here number of subsets is established
 
    


    #JT: onec averadged marginals dictionary is ready computation of POVMs is done

    for subset in subsets_of_qubits:
        POVMs_dictionary[subset] =  compute_subset_POVM(results_dictionary =results_dictionary,marginals_dictionary=marginals_dictionary,subset=subset,
                                    estimation_method=estimation_method)
        
    return POVMs_dictionary

#JT: This method is used to get noise matrices from reconstructed POVMs, used e.g. after clustering to get noise matrices for computed clusters
# subsets list - list of subsets for which computation should be performed
        

def compute_subset_POVM( results_dictionary:Dict, 
                                    subset:Tuple,
                                    marginals_dictionary:Dict,
                                    averaged_marginals_dictionary = None, 
                                    estimation_method:str='pls',
                                    show_progress_bar: Optional[bool] = True):
        # subset_key = 'q' + 'q'.join([str(s) for s in subset])

        #JT: the if below is redundant (at least for as we use it, as the same codition is checked in compute_subsets_POVMs_averaged)

        if averaged_marginals_dictionary != None and  subset in averaged_marginals_dictionary.keys():

            averaged_marginals = averaged_marginals_dictionary[subset]
            
        
        else:

            averaged_marginals =  probability.compute_average_marginal_for_subset(subset=subset,experiment_results=results_dictionary,normalized_marginals=marginals_dictionary)

            


        computed_POVM = compute_POVM_from_marginals(marginals_dictionary=averaged_marginals,
                                          qubit_indices=subset,
                                          estimation_method=estimation_method)
        
        return computed_POVM
        




def compute_POVM_from_marginals(     marginals_dictionary: Dict[str, np.ndarray],
                                    qubit_indices: List[int], 
                                    estimation_method='pls'):
    # print('printing')
    # print(marginals_dictionary)
    # print(qubit_indices)
    # raise KeyboardInterrupt
    number_of_qubits = len(qubit_indices)
    dimension = int(2 ** number_of_qubits)

    #FBM: fix this
    keys_list = sorted(list(marginals_dictionary.keys()))


        
    # keys_list = anf.register_names_qubits(qubit_indices=qubit_indices,
    #                                       )

    # print(marginals_dictionary)
    #First index - probe ket
    #Second index - outcome label
    frequencies_array = np.zeros((len(keys_list), dimension), dtype=complex)

    #FBM: debug
    # frequencies_array = np.full((len(keys_list), dimension), 10**(-6), dtype=complex)

    # print(keys_list)

        

    probe_kets = []

    #JT: A loop over all keys of marginals dictionary

    for index_state in range(len(keys_list)):
        key_now = keys_list[index_state]
        if estimation_method.lower() in ['maximal_likelihood', 'ml']:
            probe_kets.append(tomography.get_tensored_ket(key_now))

        #JT: for PLS here a PLS estimator is computed

        elif estimation_method.lower() in ['least_squares', 'ls', 'projected_least_squares', 'pls']:
            #TODO FIX PLS FOR SIMULATIONS
            probe_kets.append(tomography.get_tensored_ket_LS(key_now))



        try:
            frequencies_array[index_state, :] = marginals_dictionary[key_now][:]


        except(IndexError):
            frequencies_array[index_state, :] = marginals_dictionary[key_now][:, 0]

        #JT: This is a different error than that above, solution appers to be the same
        except(ValueError):
            frequencies_array[index_state, :] = marginals_dictionary[key_now][:, 0]
        except(KeyError):
            pass

    # qprint_array(frequencies_array)

    #JT the classes below are used to get physical quantities out of POVM estimators

    setup_QDT = QDTCalibrationSetup(qubits_number=number_of_qubits,
                                    probe_kets=probe_kets,
                                    frequencies_array=frequencies_array,
                                    states_labels=keys_list)

    


    #JT: Here physical POVM is reconstructed

    POVM_now = tomography.get_POVM_estimator(calibration_setup=setup_QDT,
                                                method=estimation_method)

    # if isinstance(qubit_indices, list):
    #     qubit_indices = convert.qubit_indices_to_keystring(qubit_indices)

    return POVM_now


def compute_noise_matrices_from_POVMs(subsets_of_qubits: List[List[int]],
                                      POVM_dictionary:Dict,
                                    show_progress_bar: Optional[bool] = True) -> Dict:

    noise_matrices_dictionary = {}
    
    for subset in tqdm(subsets_of_qubits,
                        disable = not show_progress_bar):

        #JT: computation of stochastic noise from a quantum POVM, unig a function from povmtools

        reduced_noise_matrix_now = get_stochastic_map_from_povm(povm=POVM_dictionary[subset])

           
        noise_matrices_dictionary[subset]= {'averaged':reduced_noise_matrix_now}

    return noise_matrices_dictionary