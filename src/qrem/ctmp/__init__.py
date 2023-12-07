"""
Subpackage: **Continuous-Time Markov Process (CTMP) Model for Quantum Error Correction**
====================================================================================

This subpackage implements the Continuous-Time Markov Process (CTMP) model as detailed in Bravyi et al. 
It is designed to analyze and mitigate errors in quantum computations, particularly those involving qubits. 
The CTMP model represents quantum errors and their rates in a quantum system and includes tools for 
calibrating these models to experimental results, simulating quantum states under the influence of these 
errors, and mitigating the resultant errors.

Submodules
----------
model.py
    Contains functionalities for calibrating the CTMP model to experimental results. It includes methods
    for computing the rates of quantum errors based on observed experimental outcomes.

simulation.py
    Provides tools for simulating the evolution of quantum states under the CTMP model. It includes
    methods for generating quantum state samples based on the stochastic matrices derived from the CTMP model.

mitigation.py
    Offers methods for mitigating errors in quantum computations. It includes procedures for
    calculating the mitigated expected values of observables and for mitigating errors in marginal probability distributions.

    
Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""