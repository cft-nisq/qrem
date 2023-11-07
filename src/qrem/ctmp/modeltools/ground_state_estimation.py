import numpy as np
from typing import Optional
from collections import defaultdict
from sympy import S
from qrem.ctmp.modeltools.ncpol2sdpa.sdp_relaxation import SdpRelaxation
from qrem.ctmp.modeltools.ncpol2sdpa.nc_utils import generate_variables, flatten
from qrem.functions_qrem.functions_benchmarks import create_hamiltonians_for_benchmarks


from typing import Dict, Tuple, List

"""Glue function to access old code
"""
def estimate_gamma(n: int, rates: List[Tuple]) -> np.float:
    # convert rates to format from FM's code
    rates_dict = {}
    for error in rates:
        i, j, error_in, _, rate = error
        if i==j:
            key = (i,)
        else:
            key = (i, j)
        if key not in rates_dict:
                rates_dict[key] = defaultdict(float)
        rates_dict[key][error_in] = rate
    _, gamma = _find_optimal_noise_strength_CTMP_SDP_relaxation(rates_dict, n)
    return gamma

"""Generates a collectinon of ground state approximations for random Hamiltonians together with ground state 
energy lower and upper bound.

Returns a list of tuples (state, low, up), where:
state - string describing the approximate ground state
low, up - lower and upper bound on the ground state energy
h - Hamiltonian
"""
def generate_random_ground_states(n_qubits: int, n_hamiltonians: int, clause_density: float) -> List[Tuple]:
    result = []
    random_hamiltonians = create_hamiltonians_for_benchmarks(n_qubits, n_hamiltonians, clause_density)
    for h in random_hamiltonians.values():
        weights = h['weights_dictionary']
        low, _, state = find_ground_state_with_SDP_relaxations(weights, n_qubits, return_ground_state_approximation=True)
        mat = h['weights_matrix'].copy()
        d = np.diag(mat).copy()
        np.fill_diagonal(mat, 0)
        vec = np.array([-2*float(v)+1 for v in state])
        # once state Psi is found, upper bound on the ground state energy by <Psi|H|Psi>
        up = np.dot(vec.T, np.dot(mat, vec)) + np.dot(d.T, vec)
        result.append((state, low, up, h))
    return result


"""Code below copy pasted from Filip's code
"""

def generate_commuting_measurements(party, label):
    measurements = []
    for i in range(len(party)):
        measurements.append(generate_variables(label + '%s' % i,
                                               party[i] - 1,
                                               hermitian=True))
    return measurements


def get_symbolic_hamiltonian_from_weights_dictionary(weights_dictionary: Dict[Tuple[int], float],
                                                     spins):

    spins_list = flatten(spins)
    hamiltonian = 0
    for qubits_subset, weight in weights_dictionary.items():
        if weight != 0:
            if len(qubits_subset) == 1:
                qi = qubits_subset[0]
                hamiltonian += float(weight) * spins_list[qi]
            elif len(qubits_subset) == 2:
                (qi, qj) = qubits_subset
                hamiltonian += float(weight) * spins_list[qi] * spins_list[qj]
    return hamiltonian

def find_ground_state_with_SDP_relaxations(weights_dictionary: Dict[Tuple[int], float],
                                           number_of_qubits: int,
                                           hierarchy_level: Optional[float] = 2,
                                           get_also_upper_bound=True,
                                           return_ground_state_approximation=False,
                                           solver_name="cvxpy"):
    configuration = [2 for _ in range(1)]
    spins = [generate_commuting_measurements(configuration, chr(65 + i))
             for i in range(number_of_qubits)]
    hamiltonian_polynomial = get_symbolic_hamiltonian_from_weights_dictionary(weights_dictionary, spins)
    substitutions = {M ** 2: S.One for M in flatten(spins)}
    sdp = SdpRelaxation(flatten(spins), verbose=2)
    sdp.get_relaxation(hierarchy_level,
                       substitutions=substitutions,
                       objective=hamiltonian_polynomial)
    sdp.solve(solver=solver_name)
    #sdp.solve(solver="mosek")
    low = sdp.primal
    if sdp.status is not 'optimal':
        print('ERROR!!! The status is not optimal')
    if return_ground_state_approximation:
        ground_state_candidate = __extract_ground_state_from_moments_matrix(
            moments_matrix=sdp.x_mat[0],
            number_of_qubits=number_of_qubits)
        ground_state_candidate = ''.join([str(s) for s in ground_state_candidate])
    if get_also_upper_bound:
        sdp.set_objective(objective=-hamiltonian_polynomial)

        #sdp.solve(solver="mosek")
        sdp.solve(solver=solver_name)
        up = -sdp.primal
    else:
        up = None
    if sdp.status is not 'optimal':
        print('ERROR!!! The status is not optimal')
    if return_ground_state_approximation:
        return low, up, ground_state_candidate
    else:
        return low, up

def __convert_rates_dictionary_into_hamiltonian(ctmp_rates_dictionary,
                                                number_of_qubits):
    constant = 0
    hamiltonian_weights = defaultdict(float)
    for qubits_subset, rates_dic in ctmp_rates_dictionary.items():
        if len(qubits_subset) == 1:
            r0, r1 = rates_dic['0'], rates_dic['1']
            constant += (r0 + r1) / 2
            hamiltonian_weights[qubits_subset] += (r0 - r1) / 2
        elif len(qubits_subset) == 2:
            r00, r01, r10, r11 = rates_dic['00'], rates_dic['01'], rates_dic['10'], rates_dic['11']
            constant += (r00 + r01 + r10 + r11) / 4
            hamiltonian_weights[(qubits_subset[0],)] += (r00 + r01 - r10 - r11) / 4
            hamiltonian_weights[(qubits_subset[1],)] += (r00 - r01 + r10 - r11) / 4
            hamiltonian_weights[qubits_subset] += (r00 - r01 - r10 + r11) / 4
    return hamiltonian_weights, constant

def _find_optimal_noise_strength_CTMP_SDP_relaxation(
        ctmp_rates_dictionary: Dict[Tuple[int], Dict[str, float]],
        number_of_qubits: int,
        hierarchy_level: Optional[int] = 1) -> Tuple[float]:
    hamiltonian_dictionary, constant = __convert_rates_dictionary_into_hamiltonian(
        ctmp_rates_dictionary=ctmp_rates_dictionary,
        number_of_qubits=number_of_qubits)
    lower_bound, upper_bound = find_ground_state_with_SDP_relaxations(
        weights_dictionary=hamiltonian_dictionary,
        number_of_qubits=number_of_qubits,
        hierarchy_level=hierarchy_level)
    return lower_bound + constant, upper_bound + constant

def __extract_ground_state_from_moments_matrix(moments_matrix,
                                               number_of_qubits):
    dimension = moments_matrix.shape[0]
    # number_of_qubits = int((dimension-1)/2)
    # print(number_of_qubits)

    spin_configuration = []

    for index_now in range(number_of_qubits):

        sign = np.sign(moments_matrix[index_now + 1, 0])
        if sign == -1:
            spin_now = 1
        elif sign == 1:
            spin_now = 0
        else:
            raise ValueError()
        spin_configuration.append(spin_now)

    return spin_configuration
