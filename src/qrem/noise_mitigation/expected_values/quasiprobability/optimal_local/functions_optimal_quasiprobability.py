"""
@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""


from typing import List, Dict, Tuple
import numpy as np
import picos
from qrem.functions_qrem import ancillary_functions as anf

from qrem.common.printer import qprint_array


def __do_sanity_checks_stochastic_maps(positive_matrix,
                                       negative_matrix):
    dimension = positive_matrix.shape[0]
    for i in range(dimension):
        for j in range(dimension):
            el_ij_positive = positive_matrix[i, j]
            el_ij_negative = negative_matrix[i, j]

            if np.sign(el_ij_positive) == -1:
                if abs(el_ij_positive) >= 10**(-8):
                    qprint_array(positive_matrix)
                    raise ValueError(
                        f"Weird solution! Element '{(i, j)}' of positive matrix has value : {el_ij_positive}")
                positive_matrix[i, j] = abs(0.)

            if np.sign(el_ij_negative) == -1:
                if abs(el_ij_negative) >= 10**(-8):
                    qprint_array(negative_matrix)
                    raise ValueError(
                        f"Weird solution! Element '{(i, j)}' of negative matrix has value : {el_ij_negative}")
                negative_matrix[i, j] = abs(0.)

        sum_column_positive = sum(positive_matrix[:, i])
        if abs(sum_column_positive - 1) > 0.001:
            qprint_array(positive_matrix)
            raise ValueError(
                f"Weird solution! Column '{i}' of positive matrix sums to : {sum_column_positive}")
        positive_matrix[:, i] /= sum_column_positive

        sum_column_negative = sum(negative_matrix[:, i])
        if abs(sum_column_negative - 1) > 0.001:
            qprint_array(negative_matrix)
            raise ValueError(
                f"Weird solution! Column '{i}' of negative matrix sums to : {sum_column_negative}")
        negative_matrix[:, i] /= sum_column_negative

        # print(sum_column_negative,sum(negative_matrix[:,i]), sum_column_positive,sum(positive_matrix[:,i]))
        # print(positive_matrix)
        # print(negative_matrix)

    return positive_matrix, negative_matrix

def find_optimal_quasiprobability_decomposition_postprocessing(stochastic_map:np.ndarray,
                                                               ):

    dimension = stochastic_map.shape[0]


    problem = picos.Problem()
    positive, negative = picos.RealVariable(name='positive',
                                            shape=(dimension,dimension),
                                            lower=0
                                            ),\
                         picos.RealVariable(name='negative',
                                            shape=(dimension,dimension),
                                            upper=0
                                            )

    a = picos.RealVariable(name='a',
                           shape=1)
    b = 1-a

    for column_index in range(dimension):
        problem.add_constraint(picos.sum(positive[:,column_index])==a)
        problem.add_constraint(picos.sum(negative[:,column_index])==b)

    problem.add_constraint(np.linalg.inv(stochastic_map)==positive+negative)

    cost_function = a

    problem.set_objective(direction='min',
                          expression=cost_function
                          )


    problem.solve(solver='mosek')


    a_optimal = a.value
    norm_optimal = a_optimal+abs(1-a_optimal)

    positive_matrix = np.array(positive.value_as_matrix/a_optimal)
    negative_matrix = np.array(negative.value_as_matrix / (1 - a_optimal))

    positive_matrix, negative_matrix = __do_sanity_checks_stochastic_maps(positive_matrix=positive_matrix,
                                                                          negative_matrix=negative_matrix)


    return [a_optimal,1-a_optimal], [positive_matrix, negative_matrix], norm_optimal


def get_probability_distribution_from_quasiprobability(quasiprobability):

    norm = np.linalg.norm(quasiprobability,ord=1)

    return [abs(qa)/norm for qa in quasiprobability]






def get_quasiprobability_dictionary_from_noise_matrices(local_noise_matrices:Dict[Tuple[int],np.ndarray]):

    quasiprobability_dictionary = {}
    for cluster, local_noise_matrix in local_noise_matrices.items():
        quasiprobability_distribution, stochastic_maps, quasi_norm = find_optimal_quasiprobability_decomposition_postprocessing(
            local_noise_matrix)

        probability_distribution = get_probability_distribution_from_quasiprobability(
            quasiprobability=quasiprobability_distribution)

        quasiprobability_dictionary[cluster] = (probability_distribution,stochastic_maps,quasi_norm)

    return quasiprobability_dictionary



