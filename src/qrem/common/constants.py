"""
Constants Module for QREM
=========================

Constants Module in the QREM (Quantum Readout Error Mitigation) package defines various 
constants and configurations used throughout the project. These constants are central to maintaining 
consistency and accuracy in quantum computations and data handling.

Attributes
----------
SIGNIFICANT_DIGITS : int
    The default number of significant digits to be used in numerical computations within QREM. 
    Currently set to 9, this value determines the precision level for general numerical outputs.

SIGNIFICANT_DIGITS_POVM : int
    Defines the number of significant digits specifically for computations involving Positive 
    Operator-Valued Measures (POVMs). A precision of 7 significant digits is used for these 
    specialized calculations.

EXPERIMENT_TYPE_SYMBOLS : dict
    A dictionary mapping different types of quantum experiments to specific numerical codes. 
    Each experiment type, such as 'qdot', 'ddot', etc. , is associated with a unique integer. 
    These codes are used for categorizing and identifying experiments within the QREM system.

CIRCUIT_DATA_TYPE : numpy.dtype
    Specifies the data type for storing circuit data, using NumPy's `dtype` for efficient and 
    consistent data handling. Set to `np.uint8`, it optimizes memory usage for circuit representation.

CIRC_EFFECT_DISTANCE : float
    A constant defining a threshold or reference distance value used in circuit total count estimations. 
    This parameter plays a role in calculations related to the spatial aspects of circuit layouts or effects.

CIRC_PROBABILITY_OF_FAIL : float
    Represents the estimated probability of failure for a given circuit. Set to 0.5, this constant is 
    used in various probabilistic analyses and estimations regarding circuit performance and reliability.

Notes
-----
- The choice of significant digits, both general and POVM-specific, reflects a balance between 
  computational efficiency and the precision required for quantum computations.
- The `EXPERIMENT_TYPE_SYMBOLS` dictionary can be expanded as new types of quantum experiments are 
  incorporated into the QREM project.
- The `CIRCUIT_DATA_TYPE` is aligned with the typical data requirements for quantum circuits, ensuring 
  that the data structures used are both memory-efficient and sufficient for the intended computations.
- The `CIRC_EFFECT_DISTANCE` and `CIRC_PROBABILITY_OF_FAIL` constants are crucial for simulations and 
  estimations that factor in circuit layout and reliability within quantum experiments.

    
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""
#===================================================
# Constants
#===================================================
import numpy as np
#Why 9 is the number as of now? 
SIGNIFICANT_DIGITS = 9
SIGNIFICANT_DIGITS_POVM = 7

EXPERIMENT_TYPE_SYMBLOS = {"qdot": 6, "ddot": 2, "coh": 2}
CIRCUIT_DATA_TYPE = np.dtype(np.uint8) 


#For Circuits total count estimation:
CIRC_EFFECT_DISTANCE =0.1
CIRC_PROBABILITY_OF_FAIL = 0.5