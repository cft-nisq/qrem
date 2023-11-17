"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""

from typing import Optional
import time

import numpy as np
from qrem.functions_qrem import functions_data_analysis as fda

def __get_mapping_2SAT(clause_now,
                       j):
    variables = clause_now[1]
    signs = clause_now[0]
    if j not in variables:
        return 0
    else:
        location = int(np.where(np.array(variables) == j)[0])
        return signs[location]


def __get_local_field_1q(qubit_index,
                         clauses_list):
    variable_index = qubit_index

    local_field = -sum([__get_mapping_2SAT(ck, variable_index) for ck in clauses_list])

    return local_field

def __get_local_field_2q(qubits_indices,
                         clauses_list):
    variable_index0, variable_index1 = qubits_indices[0], qubits_indices[1]

    local_field = sum(
        [__get_mapping_2SAT(ck, variable_index0) * __get_mapping_2SAT(ck, variable_index1) for
         ck in clauses_list])

    return local_field


def get_random_2SAT_clauses(number_of_qubits,
                            clause_density):

    number_of_variables = number_of_qubits
    number_of_clauses = int(np.floor(clause_density * number_of_variables))

    clauses_list = []
    while len(clauses_list) < number_of_clauses:
        var_i = np.random.randint(0, number_of_qubits)
        var_j = np.random.randint(0, number_of_qubits)
        while var_i == var_j:
            var_j = np.random.randint(0, number_of_qubits)
        sign_i, sign_j = np.random.choice([-1, 1]), np.random.choice([-1, 1])

        clause_now = ((int(sign_i), int(sign_j)), tuple(sorted((int(var_i), int(var_j)))))
        if clause_now not in clauses_list:
            clauses_list.append(clause_now)


    return clauses_list


def get_weights_dictionary_from_weights_matrix(weights_matrix):

    weights_dictionary = {}


    number_of_qubits = weights_matrix.shape[0]

    for i in range(number_of_qubits):
        for j in range(i, number_of_qubits):
            if weights_matrix[i, j] != 0:
                if i == j:
                    weights_dictionary[(i,)] = weights_matrix[i, i]
                elif j > i:
                    weights_dictionary[(i, j)] = weights_matrix[i, j]

    return weights_dictionary

def get_2SAT_hamiltonian_from_clauses(clauses_list,
                                      number_of_qubits):


    weights_matrix = np.zeros((number_of_qubits, number_of_qubits))

    for i in range(number_of_qubits):
        for j in range(i, number_of_qubits):
            if i == j:
                weights_matrix[i,i]+=__get_local_field_1q(qubit_index=i,
                                                          clauses_list=clauses_list)


            else:
                weights_matrix[i,j]+= __get_local_field_2q(qubits_indices=[i, j],
                                                           clauses_list=clauses_list)

    weights_dictionary = get_weights_dictionary_from_weights_matrix(weights_matrix=weights_matrix)

    return weights_matrix, weights_dictionary





def generate_random_2SAT_hamiltonian(number_of_qubits:int,
                                     clause_density:float):


    random_clauses_list = get_random_2SAT_clauses(number_of_qubits=number_of_qubits,
                                                  clause_density=clause_density)


    weights_matrix, weights_dictionary = get_2SAT_hamiltonian_from_clauses(clauses_list=random_clauses_list,
                                                                           number_of_qubits=number_of_qubits)


    dictionary_to_return = {'clauses_list':random_clauses_list,
                            'number_of_qubits':number_of_qubits,
                            'weights_matrix':weights_matrix,
                            'weights_dictionary':weights_dictionary,
                            'clause_density':clause_density,
                            'name_of_hamiltonian':'2SAT'}

    return dictionary_to_return


def __change_representation_to_pysat(clause):
    return [clause[0][0]*(clause[1][0]+1),clause[0][1]*(clause[1][1]+1)]

def __get_state_from_other_format(solution):
    state = []
    for k in range(len(solution)):
        if np.sign(solution[k])==1:
            state.append('0')
        else:
            state.append('1')

    return state

def get_energy_from_SAT_solution(SATS, number_of_clauses):

    energy= 4*(-SATS+number_of_clauses)-number_of_clauses


    return energy

def __solve_2SAT_Hamiltonian_pysat(
        # weights_matrix,
                                   clauses_list,
                                   weights_dictionary,
                                   number_of_qubits,
                                   clause_density,
                                   repetitions = 1):

    from pysat.formula import WCNF
    from pysat.examples.rc2 import RC2Stratified

    new_rep = [__change_representation_to_pysat(cl) for cl in clauses_list]
    wcnf_now = WCNF()
    for cl in new_rep:
        wcnf_now.append(cl, weight=1)


    states_sol_repets, costs_sol_repets = [], []
    for heh in range(repetitions):
        solver_rc2 = RC2Stratified(wcnf_now,
                                   solver='g4')
        solution_this = solver_rc2.compute()
        sol_state_now = list(solver_rc2.model)
        sol_cost_now = solver_rc2.cost

        states_sol_repets.append(sol_state_now)
        costs_sol_repets.append(sol_cost_now)

    argmin_sol = np.argmin(costs_sol_repets)
    sol_cost_now = costs_sol_repets[argmin_sol]
    sol_state_now = states_sol_repets[argmin_sol]
    sat2_solution_rc2 = number_of_qubits * clause_density - sol_cost_now

    binary_state_rc2 = __get_state_from_other_format(sol_state_now)
    preferred_ground_state = ''.join([ex for ex in binary_state_rc2])
    # hamiltonian_elements = np.count_nonzero(weights_matrix)
    #
    # qubits, H1q, pairs, pairs_weights = fp.get_qubits_from_weights_matrix(weights_matrix)
    number_of_clauses = len(clauses_list)

    maxsat_solution, lowest_binary_state = sol_cost_now, preferred_ground_state

    # minimal_energy = get_energy_from_sat(maxsat_solution, number_of_clauses)
    min_energy = get_energy_from_SAT_solution(sat2_solution_rc2, number_of_clauses)

    list_state_binary = list(preferred_ground_state)
    ground_state_energy = fda.get_energy_from_bitstring_diagonal_local(bitstring=list_state_binary,
                                                                       weights_dictionary=weights_dictionary)\
        # .get_energy_from_string(list_state_binary, weights_dictionary)

    if abs(ground_state_energy - min_energy) >= 10 ** (-10):
        raise ValueError('something is wrong with ground state')

    dictionary_to_return = {
        # 'single_qubits': qubits,
        # 'pairs_of_qubits': pairs,
        # 'weighted_1q': H1q,
        # 'weighted_2q': pairs_weights,
        # 'weights_matrix': Wij,
        # 'weights_dictionary': weights_dict,
        # 'number_of_hamiltonian_elements': hamiltonian_elements,
        'ground_state': list_state_binary,
        'ground_state_energy': ground_state_energy,
        # 'clauses': clauses_list,
        'maxsat_solution': maxsat_solution
    }

    return dictionary_to_return





def solve_2SAT_hamiltonian(hamiltonian_data,
                           solving_method = 'pysat'
                           ):


    clauses_list = hamiltonian_data['clauses_list']
    weights_dictionary = hamiltonian_data['weights_dictionary']
    number_of_qubits = hamiltonian_data['number_of_qubits']
    clause_density = hamiltonian_data['clause_density']

    if solving_method.lower() in ['pysat']:
        solution_data = __solve_2SAT_Hamiltonian_pysat(clauses_list=clauses_list,
                                       weights_dictionary=weights_dictionary,
                                       number_of_qubits=number_of_qubits,
                                       clause_density=clause_density)

    return {**hamiltonian_data, **solution_data}