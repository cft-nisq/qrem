"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""
from typing import Optional, Dict, Tuple

import numpy as np
import picos
from tqdm import tqdm

from qrem.functions_qrem import ancillary_functions as anf
from qrem.noise_mitigation.expected_values.CRS.scipy_optimizers_wrappers import \
    wrapper_scipy_optimize as scop_wrapper

import qrem.common.math as qrem_math
from qrem.common.printer import qprint, qprint_array

def get_symmetrization_matrix_1q(stochastic_matrix: np.ndarray):
    p10, p01 = stochastic_matrix[1, 0], stochastic_matrix[0, 1]

    p, q = p10, p01

    delta = q - p

    b = np.max([0, -delta / (1 - delta)])

    dp, dm = 1 + delta, 1 - delta

    a = (b * dm + delta) / dp

    Ai = np.array([[1 - a, b], [a, 1 - b]])

    if qrem_math.is_matrix_stochastic(Ai):
        return Ai
    else:
        raise ValueError('Something is wrong, symmetrization matrix not stochastic.')


def get_equalization_matrix_1q(symmetrized_matrix,
                               max_probability):
    pi = symmetrized_matrix[0, 1]
    bi = (max_probability - pi) / (1 - 2 * pi)

    Bi = np.array([[1 - bi, bi], [bi, 1 - bi]])

    if qrem_math.is_matrix_stochastic(Bi):
        return Bi
    else:
        qprint_array(Bi)
        raise ValueError('Something is wrong, equalization matrix not stochastic.')


def find_general_symmetrizer(stochastic_matrix,
                             equalized_sucess_probability=None):
    dimension = stochastic_matrix.shape[0]

    problem = picos.Problem()
    symmetrizer = picos.RealVariable(name='S',
                                     shape=(dimension, dimension))
    if equalized_sucess_probability is None:
        p_succ = picos.RealVariable(name='psucc')
        maximization = True

    else:
        p_succ = equalized_sucess_probability
        maximization = False

    symmetrized_matrix = symmetrizer * stochastic_matrix

    for i in range(dimension):
        problem.add_constraint(picos.sum(symmetrizer[:, i]) == 1)

    for i in range(dimension):
        for j in range(dimension):
            problem.add_constraint(symmetrizer[i, j] >= 0)
    for i in range(dimension):
        problem.add_constraint(symmetrized_matrix[i, i] == p_succ)

    for i in range(dimension):
        for j in range(i + 1, dimension):
            problem.add_constraint(symmetrized_matrix[i, j] == (1 - p_succ) / (dimension - 1))
            problem.add_constraint(symmetrized_matrix[j, i] == (1 - p_succ) / (dimension - 1))

    if maximization:
        problem.set_objective('max',
                              expression=p_succ)
    else:
        problem.set_objective('find')
    problem.solve(solver='mosek')

    symmetrizer = np.array(symmetrizer.value_as_matrix)
    return symmetrizer


def find_stochastic_correction(stochastic_matrix_1q):
    dimension = 2

    problem = picos.Problem()
    improver = picos.RealVariable(name='S',
                                  shape=(dimension, dimension))

    p_succ = picos.RealVariable(name='psucc')
    improved_matrix = improver * stochastic_matrix_1q

    for i in range(dimension):
        problem.add_constraint(picos.sum(improver[:, i]) == 1)

    for i in range(dimension):
        for j in range(dimension):
            problem.add_constraint(improver[i, j] >= 0)
    for i in range(dimension):
        problem.add_constraint(improved_matrix[i, i] == p_succ)

    for i in range(dimension):
        for j in range(i + 1, dimension):
            problem.add_constraint(improved_matrix[i, j] == (1 - p_succ) / (dimension - 1))
            problem.add_constraint(improved_matrix[j, i] == (1 - p_succ) / (dimension - 1))

    problem.set_objective('max',
                          expression=p_succ)
    problem.solve(solver='mosek')

    improver = np.array(improver.value_as_matrix)

    return improver


def get_max_error_probability(symmetrized_noise_matrices):
    return np.max([noise_matrix[0, 1] for noise_matrix in symmetrized_noise_matrices])


def get_1q_lambda(p, q=None):
    if q is None:
        q = p

    return np.array([[1 - p, q],
                     [p, 1 - q]])


def get_tensored_basis_birkhoff(number_of_qubits):
    basis_1q = [np.eye(2, dtype=float),
                np.array([[0, 1], [1, 0]])]

    from qrem.functions_qrem.PyMaLi import GeneralTensorCalculator # old version: from PyMaLi import GeneralTensorCalculator 

    gtc = GeneralTensorCalculator()
    basis_birkhoff = gtc.calculate_tensor_to_increasing_list(
        objects=[basis_1q for _ in range(number_of_qubits)])

    return basis_birkhoff


def move_matrix_to_tensored_birkhoff_polytope_with_postprocessing(
        initial_stochastic_map,
        return_mover=True):
    dimension = initial_stochastic_map.shape[0]
    number_of_qubits = int(np.log2(dimension))

    problem = picos.Problem()
    mover_to_Birkhoff = picos.RealVariable(name='mover_to_Birkhoff',
                                           shape=(dimension, dimension),
                                           lower=0, upper=1)
    product = mover_to_Birkhoff * initial_stochastic_map

    for column_index in range(dimension):
        problem.add_constraint(picos.sum(mover_to_Birkhoff[:, column_index]) == 1)

    tensored_basis = get_tensored_basis_birkhoff(number_of_qubits=number_of_qubits)
    product_coeffs = picos.RealVariable('product_coeffs',
                                        shape=(dimension),
                                        lower=0,
                                        upper=1)
    problem.add_constraint(picos.sum(product_coeffs) == 1)

    tensored_birkhoff_element = 0
    for i in range(dimension):
        tensored_birkhoff_element += product_coeffs[i] * tensored_basis[i]

    problem.add_constraint(tensored_birkhoff_element == product)

    cost_function = product_coeffs[0]
    problem.set_objective(direction='max',
                          expression=cost_function
                          )
    problem.solve(solver='mosek')
    if return_mover:
        return np.array(mover_to_Birkhoff.value_as_matrix), np.array(product.value_as_matrix)
    else:
        return cost_function.coherences_values


def move_matrix_to_tensored_birkhoff_polytope_with_preprocessing(
        initial_stochastic_map,
        return_mover=True):
    dimension = initial_stochastic_map.shape[0]
    number_of_qubits = int(np.log2(dimension))

    tensored_basis = get_tensored_basis_birkhoff(number_of_qubits=number_of_qubits)

    problem = picos.Problem()

    movers_to_Birkhoff = [
        picos.RealVariable(name='mover_to_Birkhoff_%s' % i,
                           shape=(dimension, dimension),
                           lower=0, upper=1) for i in range(dimension)]

    rs = [picos.RealVariable(name='r_%s' % i, shape=1, lower=0, upper=1) for i in range(dimension)]

    problem.add_constraint(picos.sum(rs) == 1)
    product = 0
    for i in range(dimension):
        Lami, ri = movers_to_Birkhoff[i], rs[i]
        permuted_initital_lambda = initial_stochastic_map @ tensored_basis[i]
        product += Lami * permuted_initital_lambda
        for k in range(dimension):
            problem.add_constraint(picos.sum(Lami[:, k]) == ri)

    product_coeffs = picos.RealVariable('product_coeffs',
                                        shape=(dimension),
                                        lower=0,
                                        upper=1)
    problem.add_constraint(picos.sum(product_coeffs) == 1)

    tensored_birkhoff_element = 0
    for i in range(dimension):
        tensored_birkhoff_element += product_coeffs[i] * tensored_basis[i]

    problem.add_constraint(tensored_birkhoff_element == product)

    cost_function = product_coeffs[0]
    problem.set_objective(direction='max',
                          expression=cost_function
                          )
    problem.solve(solver='mosek')

    if return_mover:
        return [np.array(mover_to_Birkhoff.value_as_matrix) for mover_to_Birkhoff in
                movers_to_Birkhoff], [ri.value for ri in rs], np.array(product.value_as_matrix)
    else:
        return cost_function.coherences_values


def find_separable_decomposition_vectorized(params,
                                            element_of_tensored_birkhoff_polytope,
                                            minimize=True,
                                            tensored_vector=None,
                                            return_decoupler=False):
    dimension = element_of_tensored_birkhoff_polytope.shape[0]
    if tensored_vector is None:
        tensored_vector = 1
        for p in params:
            tensored_vector = np.kron(tensored_vector, np.array([1 - p, p]))

    problem = picos.Problem()
    coeffs_vector = picos.RealVariable(name='coeffs_vector', shape=(dimension, 1))
    product = element_of_tensored_birkhoff_polytope * coeffs_vector

    for column_index in range(dimension):
        problem.add_constraint(coeffs_vector[column_index] >= 0)
    problem.add_constraint(picos.sum(coeffs_vector) == 1)

    if minimize:
        t = picos.Norm(product - tensored_vector, p=2, q=2)
        problem.set_objective(direction='min',
                              expression=t
                              )
        problem.solve(solver='mosek')

        if return_decoupler:
            number_of_qubits = int(np.log2(dimension))
            tensored_basis = get_tensored_basis_birkhoff(number_of_qubits=number_of_qubits)
            decoupler = 0
            for i in range(dimension):
                decoupler += tensored_basis[i] * coeffs_vector[i].coherences_values

            return decoupler
        else:
            return t.value


    else:
        problem.add_constraint(product == tensored_vector)
        problem.set_objective(direction='find',
                              )

        try:
            problem.solve(solver='mosek')
            if problem.status == 'optimal':
                if return_decoupler:
                    number_of_qubits = int(np.log2(dimension))
                    tensored_basis = get_tensored_basis_birkhoff(number_of_qubits=number_of_qubits)
                    decoupler = 0
                    for i in range(dimension):
                        decoupler += tensored_basis[i] * coeffs_vector[i].coherences_values

                    return decoupler
                else:
                    return True
            else:
                return False
        except:
            return False


def find_separable_decomposition_not_vectorized_with_postprocessing(params,
                                                                    element_of_tensored_birkhoff_polytope,
                                                                    tensored_matrix=None,
                                                                    return_decoupler=False,
                                                                    symmetric=True):
    dimension = element_of_tensored_birkhoff_polytope.shape[0]
    number_of_qubits = int(np.log2(dimension))
    tensored_basis = get_tensored_basis_birkhoff(number_of_qubits=number_of_qubits)

    if tensored_matrix is None:
        tensored_matrix = 1

        if symmetric:
            for p in params:
                tensored_matrix = np.kron(tensored_matrix, np.array([[1 - p, p],
                                                                     [p, 1 - p]]))

        else:
            for i in range(int(len(params) / 2)):
                tensored_matrix = np.kron(tensored_matrix,
                                          np.array([[1 - params[2 * i], params[2 * i + 1]],
                                                    [params[2 * i], 1 - params[2 * i + 1]]]))

    problem = picos.Problem()

    decoupler = picos.RealVariable(name='decoupler', shape=(dimension, dimension),
                                   lower=0, upper=1)

    product = decoupler * element_of_tensored_birkhoff_polytope
    for i in range(dimension):
        problem.add_constraint(picos.sum(decoupler[:, i]) == 1)

    coeffs_vector = picos.RealVariable(name='coeffs_vector', shape=(dimension, 1), lower=0, upper=1)
    problem.add_constraint(picos.sum(coeffs_vector) == 1)
    problem.add_constraint(
        product == picos.sum([coeffs_vector[i] * tensored_basis[i] for i in range(dimension)]))

    t = picos.Norm(product - tensored_matrix, p=2, q=2)
    problem.set_objective(direction='min',
                          expression=t
                          )
    problem.solve(solver='mosek')

    if return_decoupler:
        return np.array(decoupler.value_as_matrix), np.array(product.value_as_matrix)

    else:
        return t.value


def find_separable_decomposition_not_vectorized_with_preprocessing(params,
                                                                   element_of_tensored_birkhoff_polytope,
                                                                   tensored_matrix=None,
                                                                   return_decoupler=False,
                                                                   symmetric=True):
    dimension = element_of_tensored_birkhoff_polytope.shape[0]
    number_of_qubits = int(np.log2(dimension))
    tensored_basis = get_tensored_basis_birkhoff(number_of_qubits=number_of_qubits)

    if tensored_matrix is None:
        tensored_matrix = 1
        if symmetric:
            for p in params:
                tensored_matrix = np.kron(tensored_matrix, np.array([[1 - p, p],
                                                                     [p, 1 - p]]))

        else:
            for i in range(int(len(params) / 2)):
                tensored_matrix = np.kron(tensored_matrix,
                                          np.array([[1 - params[2 * i], params[2 * i + 1]],
                                                    [params[2 * i], 1 - params[2 * i + 1]]]))

    problem = picos.Problem()

    pre_post = [
        picos.RealVariable(name='PP%s' % i, shape=(dimension, dimension),
                           lower=0, upper=1) for i in range(dimension)]
    rs = [picos.RealVariable(name='r_%s' % i, shape=1, lower=0, upper=1) for i in range(dimension)]

    problem.add_constraint(picos.sum(rs) == 1)
    product = 0
    for i in range(dimension):
        Lami, ri = pre_post[i], rs[i]
        permuted_lambda = element_of_tensored_birkhoff_polytope @ tensored_basis[i]
        product += Lami * permuted_lambda
        for k in range(dimension):
            problem.add_constraint(picos.sum(Lami[:, k]) == ri)

    # ps = [picos.RealVariable(name='p_%s' % i, shape=1, lower=0, upper=1) for i in range(dimension)]
    coeffs_vector = picos.RealVariable(name='coeffs_vector', shape=(dimension, 1), lower=0, upper=1)
    problem.add_constraint(picos.sum(coeffs_vector) == 1)
    problem.add_constraint(
        product == picos.sum([coeffs_vector[i] * tensored_basis[i] for i in range(dimension)]))
    # tensored_matrix

    t = picos.Norm(product - tensored_matrix, p=2, q=2)
    problem.set_objective(direction='min',
                          expression=t
                          )
    problem.solve(solver='mosek')

    if return_decoupler:
        postprocessing_matirces = [np.array(PP.value_as_matrix) for PP in
                                   pre_post]
        preprocessing_distribution = [ri.value for ri in rs]

        decoupled_matrix = np.array(product.value_as_matrix)

        return postprocessing_matirces, preprocessing_distribution, decoupled_matrix

    else:
        return t.value


def find_separable_decomposition_general(params,
                                         element_of_tensored_birkhoff_polytope,
                                         only_postprocessing: bool,
                                         symmetric: bool,
                                         tensored_matrix=None,
                                         return_decoupler=False,
                                         ):
    if only_postprocessing:
        return find_separable_decomposition_not_vectorized_with_postprocessing(params=params,
                                                                               element_of_tensored_birkhoff_polytope=element_of_tensored_birkhoff_polytope,
                                                                               tensored_matrix=tensored_matrix,
                                                                               return_decoupler=return_decoupler,
                                                                               symmetric=symmetric)
    else:
        return find_separable_decomposition_not_vectorized_with_preprocessing(params=params,
                                                                              element_of_tensored_birkhoff_polytope=element_of_tensored_birkhoff_polytope,
                                                                              tensored_matrix=tensored_matrix,
                                                                              return_decoupler=return_decoupler,
                                                                              symmetric=symmetric)


def cost_function_vectorized(params,
                             element_of_tensored_birkhoff_polytope,
                             tensored_vector=None
                             ):
    distance = find_separable_decomposition_vectorized(params=params,
                                                       element_of_tensored_birkhoff_polytope=element_of_tensored_birkhoff_polytope,
                                                       minimize=True,
                                                       tensored_vector=tensored_vector)

    cost_function = 0
    cost_function += np.sum(params)

    # for i in range(len(params)):
    #     for j in range(i+1,len(params)):
    #         pi, pj = params[i], params[j]
    #         cost_function+=1/(1-2*pi)*1/(1-2*pj)

    cost_function += 10 ** 6 * distance

    return cost_function


def cost_function_not_vectorized(params,
                                 element_of_tensored_birkhoff_polytope,
                                 symmetric,
                                 only_postprocessing,
                                 tensored_matrix=None,
                                 ):
    distance = find_separable_decomposition_general(params=params,
                                                    element_of_tensored_birkhoff_polytope=element_of_tensored_birkhoff_polytope,
                                                    only_postprocessing=only_postprocessing,
                                                    tensored_matrix=tensored_matrix,
                                                    symmetric=symmetric)
    #
    # distance = find_separable_decomposition_not_vectorized_with_preprocessing(params=params,
    #                                                                           element_of_tensored_birkhoff_polytope=element_of_tensored_birkhoff_polytope,
    #                                                                           tensored_matrix=tensored_matrix,
    #                                                                           symmetric=symmetric)

    cost_function = 0
    cost_function += np.sum(params)

    # for i in range(len(params)):
    #     for j in range(i+1,len(params)):
    #         pi, pj = params[i], params[j]
    #         cost_function+=1/(1-2*pi)*1/(1-2*pj)

    cost_function += 10 ** 7 * distance

    return cost_function


def find_optimal_tensored_decomposition_blackbox(element_of_tensored_birkhoff_polytope,
                                                 symmetric,
                                                 only_postprocessing,
                                                 printing=True,
                                                 max_error_probability=0.45,
                                                 ):
    # print('hejka')
    dimension = element_of_tensored_birkhoff_polytope.shape[0]
    number_of_qubits = int(np.log2(dimension))

    if symmetric:
        number_of_parameters = number_of_qubits
    else:
        number_of_parameters = 2 * number_of_qubits

    bounds_list = [(0, max_error_probability) for _ in range(number_of_parameters)]

    optimizer_inital = 'differential_evolution'
    optimizer_final = 'COBYLA'

    maxiter_initial = 10 ** 2
    maxiter_final = 10 ** 2
    # printing = True

    basinhopping_options_initial = {'niter': 10,
                                    'T': 0.1,
                                    'stepsize': 0.1,
                                    'disp': printing}
    params_starting = [max_error_probability for _ in
                       range(number_of_parameters)]

    basinhopping_options_final = {'niter': 10,
                                  'T': 0.01,
                                  'stepsize': 0.01,
                                  'disp': printing,
                                  'interval':3}

    differential_evolution_options_initial = {'maxiter': maxiter_initial,
                                              'popsize': 48,
                                              'workers': -1,
                                              'updating': 'deferred',
                                              'disp': printing}

    initial_kwargs = {'basinhopping_kwargs': basinhopping_options_initial,
                      'differential_evolution_kwargs': differential_evolution_options_initial}

    if optimizer_inital.upper() in ['DIFFERENTIAL_EVOLUTION']:
        basinhopping = False
    else:
        basinhopping = True

    if only_postprocessing:
        if symmetric:
            cost_function_now = cost_function_vectorized
            additional_arguments = (element_of_tensored_birkhoff_polytope,)
        else:
            cost_function_now = cost_function_not_vectorized
            additional_arguments = (element_of_tensored_birkhoff_polytope,
                                    symmetric,
                                    only_postprocessing
                                    )
    else:
        cost_function_now = cost_function_not_vectorized
        additional_arguments = (element_of_tensored_birkhoff_polytope,
                                symmetric,
                                only_postprocessing
                                )

    res = scop_wrapper(initial_parameters=params_starting,
                       target_function=cost_function_now,
                       optimizer_name=optimizer_inital,
                       basinhopping=basinhopping,
                       additional_arguments=additional_arguments,
                       bounds_list=bounds_list,
                       options={'maxiter': maxiter_initial},
                       **initial_kwargs
                       )



    best_xs = res.x
    best_funopt = res.fun
    if printing:
        qprint("initial optimal parameters:", best_xs)
        qprint("initial optimal function value:", best_funopt)

    # print(params_starting_next)
    res = scop_wrapper(initial_parameters=best_xs,
                       target_function=cost_function_now,
                       optimizer_name=optimizer_final,
                       basinhopping=True,
                       additional_arguments=additional_arguments,
                       bounds_list=bounds_list,
                       options={'maxiter': maxiter_final},
                       basinhopping_kwargs=basinhopping_options_final
                       )

    xopt = res.x
    funopt = res.fun

    if funopt < best_funopt:
        best_xs = xopt
        best_funopt = funopt

    if printing:
        print()
        qprint("final optimal parameters:", best_xs)
        qprint("final optimal function value:", best_funopt)
        print()

    return best_xs


def find_optimal_decoupler_postprocessing_symmetric(stochastic_matrix: np.ndarray,
                                                    printing: Optional[bool] = False,
                                                    ):
    mover, moved_matrix = move_matrix_to_tensored_birkhoff_polytope_with_postprocessing(
        initial_stochastic_map=stochastic_matrix,
        return_mover=True)
    # print('done')
    #
    error_probabilities = find_optimal_tensored_decomposition_blackbox(
        element_of_tensored_birkhoff_polytope=moved_matrix,
        printing=printing,
        symmetric=True,
        only_postprocessing=True)
    # print('done2')

    decoupler = find_separable_decomposition_vectorized(params=error_probabilities,
                                                        element_of_tensored_birkhoff_polytope=moved_matrix,
                                                        minimize=True,
                                                        return_decoupler=True
                                                        )

    symmetrizer = decoupler @ mover


    # symm = symmetrizer@stochastic_matrix
    # kronk = 1
    #
    # for qi in error_probabilities:
    #     kronk = np.kron(kronk,get_1q_lambda(qi))
    #
    # qprint_array(symm)
    # qprint_array(kronk)
    #
    #
    # raise KeyboardInterrupt

    decoupled_matrix = symmetrizer@stochastic_matrix

    return symmetrizer, error_probabilities, decoupled_matrix


def find_optimal_decoupler_general(stochastic_matrix: np.ndarray,
                                   symmetric: bool,
                                   only_postprocessing: bool,
                                   printing: Optional[bool] = False,
                                   ):
    # if symmetric and only_postprocessing:
    #

    if only_postprocessing:
        if symmetric:
            decoupler, error_probabilities, decoupled_matrix = find_optimal_decoupler_postprocessing_symmetric(stochastic_matrix=stochastic_matrix,
                                                                   printing=printing)

            return decoupler, error_probabilities, decoupled_matrix


        else:
            mover, moved_matrix = move_matrix_to_tensored_birkhoff_polytope_with_postprocessing(
                initial_stochastic_map=stochastic_matrix,
                return_mover=True)

    else:
        postprocessing_matrices, preprocessing_distribution, moved_matrix = move_matrix_to_tensored_birkhoff_polytope_with_preprocessing(
            initial_stochastic_map=stochastic_matrix,
            return_mover=True)

    error_probabilities = find_optimal_tensored_decomposition_blackbox(
        element_of_tensored_birkhoff_polytope=moved_matrix,
        printing=printing,
        symmetric=symmetric,
        only_postprocessing=only_postprocessing)

    if only_postprocessing:
        decoupler, decoupled_matrix = find_separable_decomposition_not_vectorized_with_postprocessing(
            params=error_probabilities,
            element_of_tensored_birkhoff_polytope=moved_matrix,
            return_decoupler=True,
            symmetric=symmetric)
        #TODO FBM: add error probabilities
        return decoupler@mover, None, decoupled_matrix

    else:
        postprocessing_matrices, preprocessing_distribution, decoupled_matrix = find_separable_decomposition_not_vectorized_with_preprocessing(
            params=error_probabilities,
            element_of_tensored_birkhoff_polytope=moved_matrix,
            return_decoupler=True,
            symmetric=symmetric)

        # print(preprocessing_distribution)
        return preprocessing_distribution, postprocessing_matrices, error_probabilities, decoupled_matrix






def get_CRS_dictionaries_from_noise_matrices_only_preprocessing(local_noise_matrices:Dict[Tuple[int], np.ndarray],
                                                                printing:Optional[bool]=False,
                                                                print_progress_bar:Optional[bool]=True):
    error_probabilitities_dictionary = {}
    symmetrizers_dictionary = {}

    for cluster, local_map in tqdm(local_noise_matrices.items(), disable= not print_progress_bar):
        # qprint_array(local_map)
        if len(cluster) == 1:
            equalizer = find_general_symmetrizer(stochastic_matrix=local_map
                                                         )
            equalized_matrix = equalizer @ local_map
            error_probability = equalized_matrix[0, 1]

            # errors_probabilities = [error_probability]
            error_probabilitities_dictionary[cluster] = error_probability
        else:
            equalizer, errors_probabilities = find_optimal_decoupler_general(
                stochastic_matrix=local_map,
                symmetric=True,
            only_postprocessing=True,
            printing=printing)
            for index_qubit in range(len(errors_probabilities)):
                error_probabilitities_dictionary[(cluster[index_qubit],)] = errors_probabilities[
                    index_qubit]

        symmetrizers_dictionary[cluster] = equalizer



    return symmetrizers_dictionary, error_probabilitities_dictionary

def get_CRS_dictionaries_from_noise_matrices(local_noise_matrices:Dict[Tuple[int], np.ndarray],
                                                                printing:Optional[bool]=False,
                                                                print_progress_bar:Optional[bool]=True):
    error_probabilitities_dictionary = {}
    symmetrizers_dictionary = {}

    for cluster, local_map in tqdm(local_noise_matrices.items(), disable= not print_progress_bar):
        # qprint_array(local_map)
        if len(cluster) == 1:
            equalizer = find_general_symmetrizer(stochastic_matrix=local_map
                                                         )
            equalized_matrix = equalizer @ local_map
            error_probability = equalized_matrix[0, 1]

            # errors_probabilities = [error_probability]
            error_probabilitities_dictionary[cluster] = error_probability
        else:
            equalizer, errors_probabilities = find_optimal_decoupler_general(
                stochastic_matrix=local_map,
                symmetric=True,
            only_postprocessing=False,
            printing=printing)
            for index_qubit in range(len(errors_probabilities)):
                error_probabilitities_dictionary[(cluster[index_qubit],)] = errors_probabilities[
                    index_qubit]

        symmetrizers_dictionary[cluster] = equalizer



    return symmetrizers_dictionary, error_probabilitities_dictionary




def get_error_mitigation_multipliers_2_local(error_probabilities_dictionary,
                                             number_of_qubits):
    multipliers_mitigation = {}
    for i in range(number_of_qubits):
        for j in range(i + 1, number_of_qubits):
            multiplier = 1
            for qi in (i, j):
                multiplier /= (1 - 2 * error_probabilities_dictionary[(qi,)])
            multipliers_mitigation[(i, j)] = multiplier
    for i in range(number_of_qubits):
        multipliers_mitigation[(i,)] = 1 / (1 - 2 * error_probabilities_dictionary[(i,)])
    return multipliers_mitigation