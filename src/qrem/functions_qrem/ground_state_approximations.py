"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""

from typing import Dict, Tuple, Optional

import numpy as np
from qrem.functions_qrem import ancillary_functions as anf, functions_data_analysis as fda
from qrem.ctmp.modeltools.ncpol2sdpa import generate_variables, flatten, SdpRelaxation
from sympy import S


def generate_commuting_measurements(party,
                                    label):
    measurements = []
    for i in range(len(party)):
        measurements.append(generate_variables(label + '%s' % i,
                                               party[i] - 1,
                                               hermitian=True))
    return measurements


def find_ground_state_energy_bruteforce(weights_dictionary,
                                        number_of_qubits):
    classical_states = anf.get_classical_register_bitstrings(qubit_indices=range(number_of_qubits),
                                                             )

    ground_state_energy = 10 ** 6
    ground_state = None
    for bitstring in classical_states:

        energy_now = fda.get_energy_from_bitstring_diagonal_local(bitstring=bitstring,
                                                                  weights_dictionary=weights_dictionary)

        if energy_now < ground_state_energy:
            ground_state_energy = energy_now
            ground_state = bitstring

    return ground_state_energy, ground_state


def get_symbolic_hamiltonian_from_weights_dictionary(weights_dictionary: Dict[Tuple[int], float],
                                                     spins):

    spins_list = flatten(spins)
    hamiltonian = 0
    # print(spins_list)
    for qubits_subset, weight in weights_dictionary.items():
        if weight != 0:
            if len(qubits_subset) == 1:
                qi = qubits_subset[0]
                hamiltonian += float(weight) * spins_list[qi]
            elif len(qubits_subset) == 2:
                (qi, qj) = qubits_subset
                hamiltonian += float(weight) * spins_list[qi] * spins_list[qj]


            # if len(qubits_subset) == 1:
            #     qi = qubits_subset[0]
            #     hamiltonian += weight * spins[qi][0][0][0]
            # elif len(qubits_subset) == 2:
            #     (qi, qj) = qubits_subset
            #     hamiltonian += weight * spins[qi][0][0][0] * spins[qj][0][0][0]

    return hamiltonian


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


from qrem.functions_qrem.CBB import branchandbound_tools as bb_tools, spin_models as spin_tools
from qrem.ctmp.modeltools.ncpol2sdpa.chordal_extension import find_variable_cliques


def find_ground_state_with_CBB(weights_dictionary: Dict[Tuple[int], float],
                               number_of_qubits: int,
                               verbose:Optional[int]=0):

    # configuration = [2 for _ in range(1)]
    # spin_variables = [generate_commuting_measurements(configuration, chr(65 + i))
    #          for i in range(number_of_qubits)]
    # print(spin_variables)

    # length = number_of_qubits
    # number_of_qubits = 2
    configuration = [2]
    spin_variables = spin_tools.get_square2D_spins(k=number_of_qubits,
                                                   l=1,
                                                   configuration=configuration)
    substitutions = {M ** 2: S.One for M in flatten(spin_variables)}

    # spin_variables = spin_tools.get_square2D_spins(k=number_of_qubits,
    #                                                l=1,
    #                                                configuration=configuration)
    #
    # local = spin_tools.get_2Dsquare_localdisorder(k=number_of_qubits, l=1, sigma=0.5)
    # hamiltonian_polynomial = spin_tools.get_2Dsquare_ferromagneticdisorder_hamiltonian(k=number_of_qubits, l=1,
    #                                                                         local=local,
    #                                                                         s_variables=spin_variables)
    #


    # print(hamiltonian_polynomial)
    #
    # #
    hamiltonian_polynomial = get_symbolic_hamiltonian_from_weights_dictionary(weights_dictionary=weights_dictionary,
                                                                   spins=spin_variables)

    # hamiltonian_polynomial = spin_variables[0][0][0][0]*1
    # #
    # print(hamiltonian_polynomial)


    cliques = find_variable_cliques(variables=flatten(spin_variables),
                                    objective=hamiltonian_polynomial)

    # print(cliques)

    z_low, z_up, ground_state_candidate_list = bb_tools.get_groundBandB(s_variables=spin_variables,
                                                                   substitutions=substitutions,
                                                                   hamiltonian=hamiltonian_polynomial,
                                                                   cliques=cliques,
                                                                   verbose=verbose)


    ground_state_candidate = ''.join(['1' if s==-1 else '0' for s in ground_state_candidate_list]
                                     )




    return z_low, z_up, ground_state_candidate


def find_ground_state_with_SDP_relaxations(weights_dictionary: Dict[Tuple[int], float],
                                           number_of_qubits: int,
                                           hierarchy_level: Optional[float] = 2,
                                           get_also_upper_bound=True,

                                           return_ground_state_approximation=False):
    # number_of_qubits = weights_dictionary.shape[0]
    # generates the symbolic spin variables
    configuration = [2 for _ in range(1)]
    spins = [generate_commuting_measurements(configuration, chr(65 + i))
             for i in range(number_of_qubits)]

    # gets the hamiltonian in the form of a polynomial in the spins
    hamiltonian_polynomial = get_symbolic_hamiltonian_from_weights_dictionary(weights_dictionary, spins)

    # properties of the spin operators: square equal to identity
    substitutions = {M ** 2: S.One for M in flatten(spins)}

    # print(spins)
    # generating the sdp moment matrix
    sdp = SdpRelaxation(flatten(spins), verbose=2)
    sdp.get_relaxation(hierarchy_level,
                       substitutions=substitutions,
                       objective=hamiltonian_polynomial)

    # solving for the lower bound
    sdp.solve(solver="mosek")
    low = sdp.primal

    if sdp.status is not 'optimal':
        print('ERROR!!! The status is not optimal')

    # print(low)
    # print(sdp.__dict__.keys())
    # print(sdp.block_struct)

    if return_ground_state_approximation:
        ground_state_candidate = __extract_ground_state_from_moments_matrix(
            moments_matrix=sdp.x_mat[0],
            number_of_qubits=number_of_qubits)
        ground_state_candidate = ''.join([str(s) for s in ground_state_candidate])

    # solving for the upper bound
    if get_also_upper_bound:
        sdp.set_objective(objective=-hamiltonian_polynomial)

        sdp.solve(solver="mosek")
        up = -sdp.primal
    else:
        up = None


    if sdp.status is not 'optimal':
        print('ERROR!!! The status is not optimal')

    if return_ground_state_approximation:
        return low, up, ground_state_candidate
    else:
        return low, up

#
#
# raise KeyboardInterrupt


# ||\Lambda^{-1}||_{1-->1}

# 5q, 5q, 5q, 5q, 5q
# lam0 \otimes lam1 \otimes lam2

# q0, q6, q11
# exp(15) ---> exp(3)

# 2q, 2q
# lam0 \otimes lam1
# 2q 2q -------------------------> 1q, 1q, 1q, 1q
# A0 * lam0 \otimes A1 lam1 = s0 \otimes s1 \otimes s2 \otimes s3
