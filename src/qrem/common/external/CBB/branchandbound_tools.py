"""
Created on Wed Nov 22 23:50:46 2017

@author: Flavio Baccari
"""
from __future__ import print_function, division

import copy as cp
from operator import itemgetter
from time import time

import numpy as np
from qrem.common.external.ncpol2sdpa import flatten, SdpRelaxation, get_monomials
from qrem.common.external.ncpol2sdpa.nc_utils import get_support
from numpy import dot


def get_triang_ineqs(spins):
    """Generate the list of triangle inequalities involving the given spin
    variables

    :param spins: list of spin variables
    :type spins: list of sympy.core.symbol.Symbol
    :returns: list of sympy.core.add.Add
    """

    ineq = []

    # Run over all the triples
    for i in range(len(spins) - 2):
        for j in range(i + 1, len(spins) - 1):
            for k in range(j + 1, len(spins)):
                # Add all the different sign choices
                ineq.append(spins[i] * spins[j] + spins[i] * spins[k] +
                            spins[j] * spins[k] + 1)
                ineq.append(spins[i] * spins[j] - spins[i] * spins[k] -
                            spins[j] * spins[k] + 1)
                ineq.append(-spins[i] * spins[j] - spins[i] * spins[k] +
                            spins[j] * spins[k] + 1)
                ineq.append(-spins[i] * spins[j] + spins[i] * spins[k] -
                            spins[j] * spins[k] + 1)
    return ineq


def get_det_guess(s_variables, cliques, Mom, one, two):
    """Obtains the closest spin configuration from the moment matrix resulting from
    the lower bound SDP

    :param s_variables: list of spin variables
    :type s_variables: list of sympy.core.symbol.Symbol
    :param cliques: list of the cliques of the chordal extension
    :type cliques: list of lists of sympy.core.symbol.Symbol
    :param Mom: the block moment matrix at level 1
    :type Mom: list of numpy.ndarray
    :param one: coefficients of the one-body terms in the hamiltonian
    :type one: list of float
    :param two: coefficients of the two-body terms in the hamiltonian as a matrix
    :type one: list of list of float


    :returns: the deterministic configuration final_config as list of floats
    """

    # taking a Choleski decomposition of each block in the moment matrix
    Red = []
    configs = []

    for c, M in enumerate(Mom):

        B = np.linalg.cholesky(M + (10 ** (-6)) * np.identity(len(M)))
        BT = B.T.conj()
        # Get the smallest size of the vectors

        diff = np.linalg.norm(M - np.dot(B, BT))
        l = len(flatten(s_variables))

        while diff < 10 ** (-5):
            l -= 1
            tmp = B[::, 0:l]
            diff = np.linalg.norm(M - np.dot(tmp, tmp.T.conj()))

        d = l + 1
        Red.append(B[::, 0:d])

        configs.append([np.sign(dot(B[0, 0:d], el)) for el in B[1::, 0:d]])

    # Extract the overall configuration
    sub_conf = {}

    for b, block in enumerate(cliques):
        for e, el in enumerate(block):
            sub_conf[el] = configs[b][e]

    config = []

    for el in flatten(s_variables):
        config.append(sub_conf[el])

    bound = get_bound_from_det(flatten(config), one, two)

    final_config = cp.copy(config)
    # Check if the bound can be improved by flipping a single spin

    for i in range(len(config)):

        flip_config = cp.deepcopy(config)
        flip_config[i] = cp.deepcopy(-flip_config[i])

        bound_tmp = get_bound_from_det(flatten(flip_config), one, two)

        if bound_tmp < bound:
            bound = cp.deepcopy(bound_tmp)
            final_config = cp.deepcopy(flip_config)

    return final_config


def get_bound_from_det(config, one, two):
    """Gets the energy of the corresponding configuration

    :param config: the spin configuration
    :type config: list of float
    :param one: coefficients of the one-body terms in the hamiltonian
    :type one: list of float
    :param two: coefficients of the two-body terms in the hamiltonian as a matrix
    :type one: list of list of float


    :returns: the upper bound to the ground state energy as float
    """
    # Config has to be flattened

    bound = 0

    # One-body
    for i, el in enumerate(config):
        bound += one[i] * el

    # Two-body

    for i, el1 in enumerate(config):
        for j, el2 in enumerate(config[i + 1::], i + 1):
            bound += two[i][j] * el1 * el2

    return bound


def get_strengthen_low(s_variables, substitutions, hamiltonian, cliques,
                       threshold, solverparameters=None, verbose=0):
    """Runs the relaxation for the first time and determines how many triangle inequalities
    to add to get a reasonably good lower bound. Outputs the corresponding lower bound
    as well

    :param s_variables: list of spin variables
    :type s_variables: list of sympy.core.symbol.Symbol
    :param substitutions: substitution to be applied in the generation of the moment matrix at the level of operators
    :type substitutions: dict of items of polynomials of sympy.core.symbol.Symbol
    :param hamiltonian: hamiltonian of the problem as a symbolic polynomial of the spins
    :type hamiltonian: sympy.core.add.Add
    :param cliques: list of the cliques of the chordal extension
    :type cliques: list of lists of sympy.core.symbol.Symbol
    :param threshold: threshold clique size n_t under which generats blocks at level 2
    :type threshold: int
    :param solverparameters: parameters for the SDP solver
    :type solverparameters: dict
    :param verbose: verbosity level
    :type verbose: int

    :returns: sdp as ncpol2sdpa sdp class, list of triangle inequalities to add as
    list of sympy.core.add.Add
    """
    N = len(flatten(s_variables))
    # n_ineqs = N
    new_ineqs = []
    ineqs = []
    # Generate the mixed level with blocks al level 2 only if the are below
    # the threshold size
    monomial_blocks = []

    for el in cliques:

        if len(el) <= threshold:

            monomial_blocks.append(get_monomials(el, 2))
        else:
            monomial_blocks.append(get_monomials(el, 1))

    # Solve the first SDP

    sdp = SdpRelaxation(flatten(s_variables), verbose=verbose)
    sdp.get_relaxation(
        -1,
        substitutions=substitutions,
        objective=hamiltonian,
        momentinequalities=flatten(ineqs),
        extramonomials=monomial_blocks)

    sdp.solve(solver="mosek", solverparameters=solverparameters)

    bound_init = cp.deepcopy(sdp.primal)
    bound_new = cp.deepcopy(sdp.primal)

    # Get the moment matrix without the identity and computing the triangular
    # inequalities

    lengths = [len(el) for el in cliques]
    improv = 1

    while (improv > 0.005):

        bound_old = cp.deepcopy(sdp.primal)
        ineqs_values = []

        for c in range(len(cliques)):

            M = cp.deepcopy(sdp.x_mat[c][1:lengths[c] + 1, 1:lengths[c] + 1])

            for i in range(lengths[c] - 2):
                for j in range(i + 1, lengths[c] - 1):
                    for k in range(j + 1, lengths[c]):
                        tmp = []

                        tmp.append(M[i, j] + M[j, k] + M[i, k] + 1)
                        tmp.append(M[i, j] - M[j, k] - M[i, k] + 1)
                        tmp.append(-M[i, j] + M[j, k] - M[i, k] + 1)
                        tmp.append(-M[i, j] - M[j, k] + M[i, k] + 1)

                        ineqs_values.append([[c, i, j, k], min(tmp)])

        # Select the first N most violated ineqs and add the corresponding 4
        # constraints to the problem
        # print(ineqs_values)
        # print([el[1] for el in ineqs_values])
        # print()

        n_ineqs = min([len(ineqs_values), N])

        for _ in range(n_ineqs):
            b = [el[1] for el in ineqs_values]
            # print(ineqs_values)
            index = min(enumerate(b), key=itemgetter(1))[0]
            [c, i, j, k] = ineqs_values[index][0]
            ineqs_values.remove(ineqs_values[index])

            variables = [
                flatten(cliques[c])[i],
                flatten(cliques[c])[j],
                flatten(cliques[c])[k]
            ]
            new_ineqs.append(get_triang_ineqs(variables))

        # Solve the new sdp

        time0 = time()

        sdp = SdpRelaxation(flatten(s_variables), verbose=verbose)
        sdp.get_relaxation(
            -1,
            substitutions=substitutions,
            objective=hamiltonian,
            momentinequalities=flatten(new_ineqs),
            extramonomials=monomial_blocks)

        sdp.solve(solver="mosek", solverparameters=solverparameters)

        bound_new = cp.deepcopy(sdp.primal)
        improv = abs(bound_new - bound_old) / abs(bound_init)

        # if the relative improvement is good enough add the new inequalities

        if (improv > 0.005):
            ineqs = cp.deepcopy(new_ineqs)

    return sdp, ineqs


def get_Mom(sdp, cliques):
    """Extract the level 1 moment matrix from the solved sdp problem

    :param spd: SDP problem generated by ncpol2sdpa
    :type sdp: ncpol2sdpa.sdp_relaxation.SdpRelaxation
    :param cliques: list of the cliques of the chordal extension
    :type cliques: list of lists of sympy.core.symbol.Symbol


    :returns: the level 1 the block moment matrix Mom as list of numpy.ndarray,
    the average spin values as list of numpy.ndarray
    """

    # Extract the moment matrix for the cliques

    lengths = [len(el) for el in cliques]
    Mom = []
    spins = []

    for c in range(len(cliques)):
        M = cp.copy(sdp.x_mat[c][0:lengths[c] + 1, 0:lengths[c] + 1])
        Mom.append(sdp.x_mat[c][0:lengths[c] + 1, 0:lengths[c] + 1])

        spins.append(M[1:, 0])

    return Mom, spins


def get_up_and_low(s_variables, substitutions, ineqs, hamiltonian, one, two,
                   cliques, threshold, solverparameters=None, verbose=0):
    """Gets upper and lower bound for the given instance of branching

    :param s_variables: list of spin variables
    :type s_variables: list of sympy.core.symbol.Symbol
    :param substitutions: substitution to be applied in the generation of the moment matrix at the level of operators
    :type substitutions: dict of items of polynomials of sympy.core.symbol.Symbol
    :param ineqs: triangle inequalities to be added to improve the lower bound
    :type ineqs: list of sympy.core.add.Add
    :param hamiltonian: hamiltonian of the problem as a symbolic polynomial of the spins
    :type hamiltonian: sympy.core.add.Add
    :param one: coefficients for the one-body terms in the hamiltonian
    :type one: list of floats
    :param two: coefficients for the two-body terms in the hamiltonian
    :type two: list of floats
    :param cliques: list of the cliques of the chordal extension
    :type cliques: list of lists of sympy.core.symbol.Symbol
    :param threshold: threshold clique size n_t under which generats blocks at level 2
    :type threshold: int
    :param solverparameters: parameters for the SDP solver
    :type solverparameters: dict
    :param verbose: verbosity level
    :type verbose: int

    :returns: lower bound as float, upper bound as float, spin expectation values
    extracted from the moment matrix for each clique as list of array of floats,
    the deterministic configuration for each spin as list of floats
    """

    time0 = time()

    monomial_blocks = []

    for el in cliques:

        if len(el) <= threshold:

            monomial_blocks.append(get_monomials(el, 2))
        else:
            monomial_blocks.append(get_monomials(el, 1))

    # solving the first sdp

    sdp = SdpRelaxation(flatten(s_variables), verbose=verbose)
    sdp.get_relaxation(
        -1,
        substitutions=substitutions,
        objective=hamiltonian,
        momentinequalities=flatten(ineqs),
        extramonomials=monomial_blocks)

    sdp.solve(solver="mosek", solverparameters=solverparameters)

    print("The lower bound is " + str(sdp.primal))
    print("Time taken for lower is " + str(time() - time0))

    time0 = time()
    [Mom, spins] = get_Mom(sdp, cliques)

    config = get_det_guess(s_variables, cliques, Mom, one, two)
    bound = get_bound_from_det(flatten(config), one, two)
    print("Time taken for upper is " + str(time() - time0))

    return sdp.primal, bound, spins, config


def get_branching_node(spins, s_variables, cliques, eqs):
    """Gets the spin on which to perform the next branching. Works with an "easy-first"
    choice

    :param spins: the spin average values extracted from the moment matrix
    :type spins: list of floats
    :param s_variables: list of spin variables
    :type s_variables: list of sympy.core.symbol.Symbol
    :param cliques: list of the cliques of the chordal extension
    :type cliques: list of lists of sympy.core.symbol.Symbol
    :param eqs: list of already implemented branching choices
    :type eqs: list of sympy.core.add.Add

    :returns: the spin variable of the branching node as sympy.core.symbol.Symbol
    """

    det = 1
    sup = get_sup(s_variables, eqs)

    for c, block in enumerate(spins):
        for i, el in enumerate(block):

            if el not in sup:

                if abs(el) < det:
                    det = cp.deepcopy(abs(el))
                    branch_node = cp.deepcopy(cliques[c][i])

    return branch_node


def get_as_coeffs(hamiltonian, s_variables):
    """Extracts the one and two-body coefficients from the symbolic hamiltonian

    :param hamiltonian: hamiltonian of the problem as a symbolic polynomial of the spins
    :type hamiltonian: sympy.core.add.Add
    :param s_variables: list of spin variables
    :type s_variables: list of sympy.core.symbol.Symbol

    :returns: the one-body coeffs as list of float, the two-body coeffs as list
    of list of float
    """

    one = []
    two = []
    pol = hamiltonian.as_poly()
    mon = flatten(s_variables)

    # print(pol)

    for i, el in enumerate(mon):

        one.append(pol.coeff_monomial(el))
        tmp = []
        for j, el2 in enumerate(mon):
            if j < i:
                tmp.append(0)
            else:

                # print(el, el2)
                tmp.append(pol.coeff_monomial(el * el2))
        two.append(tmp)

    return one, two


def get_sup(s_variables, eqs):
    """Gets the spin variables on which the branching has alreadu been performed

    :param s_variables: list of spin variables
    :type s_variables: list of sympy.core.symbol.Symbol
    :param eqs: list of already implemented branching choices
    :type eqs: list of sympy.core.add.Add

    :returns: support of the branching equations as list of sympy.core.symbol.Symbol
    """

    sup = []

    for el in eqs:
        s = get_support(flatten(s_variables), el)
        sup.append(dot(s[1], flatten(s_variables)))

    return sup



def get_groundBandB(s_variables,
                    substitutions,
                    hamiltonian,
                    cliques,
                    threshold=3,
                    verbose=0,
                    # solverparameters=None
                    ):
    """Runs the whole branch&bound algorithm for the given hamiltonian

    :param s_variables: list of spin variables
    :type s_variables: list of sympy.core.symbol.Symbol
    :param substitutions: substitution to be applied in the generation of the moment matrix at the level of operators
    :type substitutions: dict of items of polynomials of sympy.core.symbol.Symbol
    :param hamiltonian: hamiltonian of the problem as a symbolic polynomial of the spins
    :type hamiltonian: sympy.core.add.Add
    :param cliques: list of the cliques of the chordal extension
    :type cliques: list of lists of sympy.core.symbol.Symbol
    :param threshold: threshold clique size n_t under which generats blocks at level 2
    :type threshold: int
    :param solverparameters: parameters for the SDP solver
    :type solverparameters: dict
    :param verbose: verbosity level
    :type verbose: int

    :returns z_low: final lower bound as float
    :returns z_up: final upper bound as float
    :returns final_config: final ground state configuration as list of floats
    """

    [one, two] = get_as_coeffs(hamiltonian, s_variables)
    # First root

    eqs = []
    tree = []
    print("Starting initial optimisation")

    # Get the first lower bound and choosing the list of triangle ineqs to add

    [sdp, ineqs] = get_strengthen_low(
        s_variables,
        substitutions,
        hamiltonian,
        cliques,
        threshold,
        # solverparameters,
        verbose=0)

    # Extract the corresponding upper bound and spin configuration

    [Mom, spins] = get_Mom(sdp, cliques)
    config = get_det_guess(s_variables, cliques, Mom, one, two)
    final_config = cp.deepcopy(config)

    z_up = get_bound_from_det(flatten(config), one, two)
    z_low = sdp.primal

    print("The lower bound is " + str(z_low))
    print("The upper bound is " + str(z_up))

    # Go for the branching with easy first

    if ((z_low - z_up) / abs(z_up) < -0.00001):
        node = get_branching_node(spins, s_variables, cliques, eqs)
        tree = [[z_low, eqs, substitutions, node]]

    while (len(tree) > 0):

        if abs(z_up-z_low)<10**(-6):
            print('bounds converged, returning solution')
            break

        lis = [el[0] for el in tree]
        index = min(enumerate(lis), key=itemgetter(1))[0]
        [z_low, eqs, subs, node] = tree[index]
        tree.remove(tree[index])

        # Branch +1

        eqs_new = cp.deepcopy(eqs)
        eqs_new.append(node - 1)
        subs_new = cp.deepcopy(subs)
        subs_new[node] = 1

        [z_low, tmp, spins, config] = get_up_and_low(
            s_variables,
            subs_new,
            ineqs,
            hamiltonian,
            one,
            two,
            cliques,
            threshold,
            # solverparameters,
            verbose=0)

        # Substitute the upper bound if better
        if tmp < z_up:
            z_up = cp.deepcopy(tmp)
            final_config = cp.deepcopy(config)

        # Include this node in the branching if the lower bound is meaningful

        if ((z_low - z_up) / abs(z_up) < -0.00001):
            node_new = get_branching_node(spins, s_variables, cliques, eqs_new)
            tree.append([z_low, eqs_new, subs_new, node_new])

        # branch -1

        eqs_new = cp.deepcopy(eqs)
        eqs_new.append(node + 1)
        subs_new = cp.deepcopy(subs)
        subs_new[node] = -1

        [z_low, tmp, spins, config] = get_up_and_low(
            s_variables,
            subs_new,
            ineqs,
            hamiltonian,
            one,
            two,
            cliques,
            threshold,
            # solverparameters,
            verbose=verbose)

        # Substitute the upper bound if better
        if tmp < z_up:
            z_up = cp.deepcopy(tmp)
            final_config = cp.deepcopy(config)

        # Include this node in the branching if the lower bound is meaningful
        print("The upper bound is " + str(z_up))
        if ((z_low - z_up) / abs(z_up) < -0.00001):
            node_new = get_branching_node(spins, s_variables, cliques, eqs_new)
            tree.append([z_low, eqs_new, subs_new, node_new])

        # Check all the previous nodes to see if we can eliminate some with
        # the new z_up

        tree_old = cp.deepcopy(tree)
        tree = []

        for el in tree_old:

            if ((el[0] - z_up) / abs(z_up) < -0.00001):
                tree.append(el)

    return z_low, z_up, final_config
