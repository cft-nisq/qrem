"""
QREM (Quantum Research Environment Manager) Common Utilities
============================================================

This subpackage `qrem.common` provides a suite of utility modules specifically tailored for the QREM project. 
It encompasses a range of functionalities from configuration loading to mathematical computations and file operations 
in the context of Quantum Error Mitigation (QEM).

Modules
-------
config
    Provides functionality for parsing configuration files and command-line arguments 
    in the Quantum Research Environment Manager (QREM) project. It's designed to streamline 
    the setup and customization of the QREM environment.

constants
    This module maintains all the mathematical constants, threshold values, and other 
    constant parameters required across the `qrem` package. It serves as a central repository 
    for constants to ensure consistency and ease of maintenance.

convert
    Contains helper functions for converting between various formats used to describe 
    circuit labels, bitstrings, quantum registers, etc. This module aids in interoperability 
    between different quantum computing representations.

experiment
    subpackage contains functions and classes related to quantum tomography 
    and creation of random set of cicrcuits.
    
io
    Provides utility functions for handling dates and file operations in a QREM context. 
    Includes functionality to format current date and time, prepare output file paths, and 
    perform file operations using pickle serialization, tailored for Quantum Error Mitigation.

math
    Encompasses all mathematical functions that are useful and utilized throughout the qrem package. 
    This module aims to provide a comprehensive collection of mathematical tools specific to quantum 
    research needs.

povmtools
    This module contains functions operating on Positive Operator-Valued Measures (POVMs), 
    an essential aspect of quantum measurements. It provides tools for POVM analysis and manipulations.

printer
    Contains helpful functions for logging and console printouts, facilitating better user 
    interaction and debugging support within the QREM project.

probability
    Dedicated to mathematical functions dealing with probability distributions, including 
    the calculation of marginals, and other probability-related operations. This module 
    supports probabilistic analyses within quantum computing contexts.

registers
    #TODO

utils
    A collection of general utility functions used throughout the QREM project. This module 
    serves as a toolbox for common tasks and operations, enhancing code reuse and efficiency.

Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""