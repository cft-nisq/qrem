import random
from qrem.types import DataStructureBase
from typing import Dict, Tuple, List

import numpy as np
from qrem.ctmp.model import rates_from_exp_results
import qrem.ctmp.modeltools.ground_state_estimation as gsa

class CTMPModelData(DataStructureBase):
    """The CTMP model as described in Bravyi et al.

    Parameters
    ----------
    n: int
        the number of qubits
    rates: list
        the list of error types, where each error = (i, j in, out, rate), with:
        i, j - qubits on which the error acts (i=j - one qubit error)
        in, out - input and output bits of the error
        rate - the rate of the given error
        e.g. (0, 2, "00", "11", 0.6) is the error flipping 0_0... --> 1_1...
        (1, 1, "0", "1", 0.6) is the error flipping _0_... -> _1_...
    gamma: float
        CTMP noise strength as in eq. (13); if None, it will be estimated heuristically

    """

    def __init__(self, rates: List[Tuple]=[], n: int=0, gamma: float=None):
        super().__init__()
        self.rates = rates
        self.n = n
        if gamma is None and len(rates)>0:
            self.gamma = self._estimate_gamma()
        else:
            self.gamma = gamma

    def _estimate_gamma(self) -> float:
        # temporary solution until solver issue is taken care of
        return sum([error[-1] for error in self.rates])
        # return gsa.estimate_gamma(self.n, self.rates)

