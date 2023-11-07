#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:21:03 2018

@author: flaviobaccari
"""

import numpy as np
from math import fmod
from qrem.ctmp.modeltools.ncpol2sdpa import generate_variables


def generate_commuting_variables(party, label):
    """Generates the list of symbolic variables representing the spins
    for a given site. The variables are treated as commuting.
    :param party: configuration indicating the configuration of number m
                  of measurements and outcomes d for each measurement. It is a
                  list with m integers, each of them representing the number of
                  outcomes of the corresponding  measurement.
    :type party: list of int
    :param label: label to represent the given party
    :type label: str
    :returns: list of sympy.core.symbol.Symbol
    """

    variables = []
    for i in range(len(party)):
        variables.append(
            generate_variables(label + '%s' % i, party[i] - 1, hermitian=True))
    return variables


def get_square2D_spins(k, l, configuration=None):
    """Genrates the list of symbolic variables representing the spins
    arranged in a 2D square lattice of size kxl.
    The variables are treated as commuting.

    :param k: horizontal size of the lattice
    :type k: int
    :param l: vertical size of the lattice
    :type l: int
    :param coniguration: number of spins per site with corresponding levels [d...d]
    :type configuration: list of int

    :returns: list of sympy.core.symbol.Symbol
    """


    if configuration is None:
        configuration = [2]

    s_variables = [[
        generate_commuting_variables(configuration, chr(65 + j * l + i))
        for i in range(l)
    ] for j in range(k)]

    # s_variables = [
    #     generate_commuting_variables(configuration, chr(65 + j * l + i))
    #     for i in range(l)
    # for j in range(k)]

    return s_variables


def get_2Dsquare_localdisorder(k, l, sigma):
    """Generates random local magnetic field values for a 2D square lattice
    distributed according to a Gaussian with zero mean and variance sigma

    :param k: horizontal size of the lattice
    :type k: int
    :param l: vertical size of the lattice
    :type l: int
    :param sigma: variance of the gaussian distribution
    :type sigma: float

    :returns: the magnetic field values h as list of float
    """

    local = []

    for i in range(k):

        row = []
        for j in range(l):

            # Add the local magnetic term

            magnetic = np.random.normal(0, sigma)

            row.append(magnetic)

        local.append(row)

    return local


def get_2Dsquare_ferromagneticdisorder_hamiltonian(k, l, local, s_variables):
    """Generates the hamiltonian of a ferromagnetic 2D Ising model and local
    mangetic fields as a polynomial in the spin variables

    :param k: horizontal size of the lattice
    :type k: int
    :param l: vertical size of the lattice
    :type l: int
    :param local: local magnetic field values
    :type local: list of float
    :param s_variables: list of spin variables
    :type s_variables: list of sympy.core.symbol.Symbol

    :returns: the hamiltonian as sympy.core.add.Add
    """
    hamiltonian = 0
    for i in range(k):

        for j in range(l):

            # Put the link on horizontal line unless I am at l
            if l > 1:
                if j is not (l - 1):

                    hamiltonian += -1.0 * s_variables[i][j][0][
                        0] * s_variables[i][int(fmod(j + 1, l))][0][0]

            # Now the link on the vertical line
            if k > 1:
                if i is not (k - 1):

                    hamiltonian += -1.0 * s_variables[i][j][0][
                        0] * s_variables[int(fmod(i + 1, k))][j][0][0]

            # Add the local magnetic term

            magnetic = local[i][j]
            hamiltonian += magnetic * s_variables[i][j][0][0]

    return hamiltonian


def get_lattice_spins(lattice, configuration):
    """Generates a dictionary of symbolic variables representing the
    spin in the given lattice

    :param lattice: the lattice structure
    :type lattice: networkx.classes.graph.Graph
    :param coniguration: number of spins per site with corresponding levels [d...d]
    :type configuration: list of int

    :returns: the spin variables as dict with tuple as keys representing the nodes
    and sympy.core.symbol.Symbol as values representing the spins
    """
    s_variables = {}
    for i, node in enumerate(lattice.nodes()):

        s_variables[node] = generate_commuting_variables(
            configuration, chr(65 + i))

    return s_variables


def get_lattice_localdisorder(lattice, sigma):
    """Generates the random magneitc field on each node of the lattice, according to
    a gaussian distribution with zero mean and variance sigma

    :param lattice: the lattice structure
    :type lattice: networkx.classes.graph.Graph
    :param sigma: variance of the gaussian distribution
    :type sigma: float

    :returns: the local field as dictionary with tuple as keys representing the nodes
    and float as values
    """

    h = {}

    for node in lattice.nodes():

        h[node] = np.random.normal(0, sigma)

    return h


def get_lattice_ferromangetic(lattice):
    """Generates the ferromagnetic couplings for each edge in the lattice

    :param lattice: the lattice structure
    :type lattice: networkx.classes.graph.Graph

    :returns: the couplings as dictionary with tuple as keys representing the edges
    and float as values
    """

    J = {}

    for edge in lattice.edges():

        J[edge] = -1

    return J


def get_lattice_hamiltonian(lattice, h, J, s_variables):
    """Generates the hamiltonian for the given lattice with the given local fields h
    and couplings J as a polynomial in the spin variables

    :param lattice: the lattice structure
    :type lattice: networkx.classes.graph.Graph
    :param h: the local fields h_i
    :type h: dict
    :param J: the couplings J_{ij}
    :type J: dict
    :param s_variables: list of spin variables
    :type s_variables: list of sympy.core.symbol.Symbol

    :returns: the hamiltonian as sympy.core.add.Add
    """

    hamiltonian = 0

    for node in lattice.nodes():
        hamiltonian += h[node] * s_variables[node][0][0]

    for edge in lattice.edges():
        hamiltonian += J[edge] * s_variables[edge[0]][0][0] * s_variables[edge[
            1]][0][0]

    return hamiltonian
