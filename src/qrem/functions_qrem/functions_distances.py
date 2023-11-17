"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""

import copy
import itertools

import numpy as np
import picos
from scipy import optimize as scopt


from qrem.functions_qrem import ancillary_functions as anf
from qrem.functions_qrem import povmtools
from qrem.noise_mitigation.expected_values.CRS.scipy_optimizers_wrappers import \
    wrapper_scipy_optimize as scop_wrapper
# from qiskit.quantum_info import partial_trace as qiskit_partial_trace
# from qiskit.quantum_info import DensityMatrix as qiskit_DM

from qrem.common.math import Constants as Const
# from qrem.common.printer import qprint, qprint_array

__worst_case_distance_names__ = ['operational', 'worst', 'worst_case', 'worst-case', 'wc']
__average_case_distance_names__ = ['average', 'average_case', 'average-case', 'ac']


def operational_distance_POVMs(POVM1,
                               POVM2=None,
                               method='direct',
                               classical=False):

    #JT: computation of classical operational distance is performed

    if classical:
        noise_map_M, noise_map_P = povmtools.get_stochastic_map_from_povm(
            POVM1), povmtools.get_stochastic_map_from_povm(POVM2)
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
        J = povmtools.get_choi_from_POVM(difference)

        #JT: old comment below:
        # this calculates completely bounded trace norm of the channel which is the upper bound for operational distance
        cbt_norm = povmtools.get_CBT_norm(J, n, m)

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

        noise_map_M, noise_map_P = povmtools.get_stochastic_map_from_povm(
            POVM1), povmtools.get_stochastic_map_from_povm(POVM2)

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

    # print(cost_function)
    problem.set_objective(direction='max',
                          expression=cost_function)

    # print(problem)
    problem.solve(solver='mosek')

    # print(cost_function.value)

    # print('indirect:')
    # best_local_difference = np.array(local_difference.value_as_matrix)
    # qprint_array(best_local_difference)
    # print(np.round(np.linalg.eigvals(best_local_difference),3))

    dav = 1 / 2 * np.sqrt(
        t00.coherences_values ** 2 + t01.coherences_values ** 2 + t11.coherences_values ** 2 + t_add.coherences_values ** 2)

    # qprint_array(np.array(local_difference.value_as_matrix))
    return dav.real


def calculate_correlations_coefficients_average_case(povm_2q):
    full_register = anf.get_classical_register_bitstrings(qubit_indices=range(4),
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


# from povms_qi import povmtools as pt
# povm_2q = pt.random_POVM(4,2)
# sign=1


def __construct_local_difference(parameters):
    nx, ny, nz = parameters

    local_difference = nx * Const.pauli_sigmas()['X'] + ny * Const.pauli_sigmas()['Y'] + nz * Const.pauli_sigmas()[
        'Z']
    return local_difference


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

    # c = np.trace((M01+M10) @ local_difference)
    # d = 0

    trace_part = a + b

    norm_squared = a ** 2 + b ** 2 + c ** 2 + d ** 2 + trace_part ** 2
    norm_squared = norm_squared.real
    # # print(norm_squared)
    # a00 = np.trace(local_difference @ M00).real
    # a01 = np.trace(local_difference @ M01)
    # a10 = np.conj(a01)
    # a10 = np.trace(local_difference @ M10)
    #     # np.trace(local_difference @ (M01 + M10)).real
    # a11 = np.trace(local_difference@M11).real
    # #
    # #
    # reduced_effect = np.array([[a00,a01],
    #                              [a10,a11]])
    #
    #
    # norm_squared_true = np.trace(reduced_effect@reduced_effect).real+(np.trace(reduced_effect).real)**2
    #
    # print(norm_squared,norm_squared_true)
    # # print(np.allclose(norm_squared, norm_squared_true))
    # #
    # # print(norm_squared,np.linalg.norm(reduced_effect,ord='fro')**2)
    # #
    # # Madded = np.zeros((4,4),dtype=complex)
    # #
    # # Madded[0:2, 0:2] = M00[:,:]
    # # Madded[0:2, 2:4] = M01[:,:]
    # # Madded[2:4, 0:2] = M10[:,:]
    # # Madded[2:4, 2:4] = M11[:,:]
    # #
    # #
    # # # qprint_array(M)
    # # test = qiskit_DM(data=Madded, dims=4)
    # #
    # # product_test = Madded@np.kron(np.eye(2),local_difference)
    # #
    # #
    # # reduced_effect_test = qiskit_partial_trace(product_test,qargs=[0]).data
    # # #
    # # #
    # #
    # # print(np.allclose(reduced_effect,reduced_effect_test))
    #
    # # qprint_array(reduced_effect)
    # # qprint_array(reduced_effect_test)
    #
    # raise KeyError

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
    #
    # qprint("BEST VAL DIFF:",best_funopt)
    #
    # # raise KeyError("UYAY")
    # res = scop_wrapper(initial_parameters=best_xs,
    #                    target_function=cost_function_now,
    #                    optimizer_name=optimizer_final,
    #                    basinhopping=True,
    #                    additional_arguments=additional_arguments,
    #                    bounds_list=bounds_list,
    #                    options={'maxiter': maxiter_final},
    #                    basinhopping_kwargs=basinhopping_options_final
    #                    )
    #
    # xopt = res.x
    # funopt = res.fun
    #
    # qprint("BEST VAL 2:",funopt)

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
        stochastic_map_2q_or_povm = povmtools.get_stochastic_map_from_povm(stochastic_map_2q_or_povm)

    noise_maps = [get_reduced_noise_matrix_2q(stochastic_map2q=stochastic_map_2q_or_povm,
                                              target_qubit=target_qubit,
                                              spectator_state=i) for i in range(2)]

    povm_0, povm_1 = tuple(
        [povmtools.get_povm_from_stochastic_map(stochastic_map=stochastic_map) for stochastic_map
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
