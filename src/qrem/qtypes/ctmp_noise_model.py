"""
This module defines the `CTMPModelData` class which encapsulates data relevant
to the Continuous-Time Markov Process (CTMP) model used for quantum error correction.

The CTMP model, as detailed in Bravyi et al., involves representing quantum errors
and their respective rates in a quantum system, specifically tailored for qubits.
This model is useful in analyzing and mitigating errors in quantum computations.

Classes
-------
CTMPModelData : DataStructureBase
    Represents the CTMP model with associated quantum error rates and other parameters.


Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""

import random
from qrem.qtypes.datastructure_base import DataStructureBase
from typing import Dict, Tuple, List

import numpy as np
from qrem.ctmp.model import rates_from_exp_results
import qrem.ctmp.modeltools.ground_state_estimation as gsa

class CTMPModelData(DataStructureBase):
    """
    Represents the CTMP model for quantum error correction as described in Bravyi et al.

    This class stores information about quantum errors and their rates in a quantum system,
    especially focusing on qubits. It is part of the larger framework for managing quantum
    error correction and mitigation.

    Parameters
    ----------
    rates : List[Tuple], optional
        A list of tuples representing error types. Each error is a tuple of the form
        (i, j, in, out, rate), where 'i' and 'j' are qubits the error acts on (i=j for
        one qubit error), 'in' and 'out' are input and output bits of the error, and
        'rate' is the rate of the given error. For example, (0, 2, "00", "11", 0.6)
        represents an error flipping 0_0... to 1_1..., and (1, 1, "0", "1", 0.6) represents
        an error flipping _0_... to _1_....
    n : int, optional
        The number of qubits in the quantum system.
    gamma : float, optional
        The CTMP noise strength as in eq. (13) from Bravyi et al. If None, it will be estimated
        heuristically based on the provided rates.

    Attributes
    ----------
    rates : List[Tuple]
        Stores the error rates and types in the quantum system.
    n : int
        The number of qubits in the quantum system.
    gamma : float
        The noise strength in the CTMP model.

    Methods
    -------
    _estimate_gamma() -> float
        Estimates the gamma value heuristically if not provided.
    """

    def __init__(self, rates: List[Tuple]=[], n: int=0, gamma: float=None):
        """
        Initializes the CTMPModelData instance with given parameters.

        Parameters
        ----------
        rates : List[Tuple], optional
            Error types and rates. Default is an empty list.
        n : int, optional
            Number of qubits. Default is 0.
        gamma : float, optional
            CTMP noise strength. Default is None, which triggers heuristic estimation.
        """        
        super().__init__()
        self.rates = rates
        self.n = n
        if gamma is None and len(rates)>0:
            self.gamma = self._estimate_gamma()
        else:
            self.gamma = gamma

    def _estimate_gamma(self) -> float:
        """
        Estimates the gamma value heuristically based on the provided error rates.

        This is a temporary solution and uses a simple sum of error rates. It is expected
        to be replaced with a more sophisticated estimation method.

        Returns
        -------
        float
            The estimated gamma value for the CTMP model.
        """        
        # temporary solution until solver issue is taken care of
        return sum([error[-1] for error in self.rates])
        # return gsa.estimate_gamma(self.n, self.rates)

