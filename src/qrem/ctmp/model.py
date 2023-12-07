"""
model.py: Error Rate Calibration in CTMP Model

This module is part of the CTMP subpackage and focuses on calibrating the error rates of a quantum system 
based on experimental results. It employs combinations of qubit states and experimental data to estimate
error rates as per the CTMP model.

Functions
---------
rates_from_exp_results(exp_results: Dict[str, Dict[str, int]], n: int) -> List[Tuple]
    Calibrates and returns error rates based on experimental results. It processes the experimental outcomes
    for pairs of qubits to estimate error transitions and rates.


Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""

import numpy as np
import scipy
from itertools import combinations
from typing import Dict, Tuple, List

from qrem.ctmp.modeltools.qubit_utils import outeq


# order of basis elements
basis_dict = {0 : "00", 1 : "11", 2: "01", 3 : "10"}
# reverse basis dictionary
rbd = {v: k for k, v in basis_dict.items()}

def rates_from_exp_results(exp_results: Dict[str, Dict[str, int]], n: int) -> List[Tuple]:
    """Calibrates the error rates to experimental results, as described in the paper. Return a list of model
    rates in the format as in CTMPModelData class.

    exp_results - dictionary of experimental results in the form {input string: dict of output counts}
    e.g. {"001" : {"000" : 56, "001" : 333, ...}}
    n - number of qubits

    The function assumes that experimental results are complete, i.e. suffice to determine the model rates.
    """
    # todo: check if the input set is complete
    pairs = list(combinations(range(0, n), 2))
    rates = []
    G_dict = {}
    for j, k in pairs:
        A = np.zeros((4,4))
        # not optimized for efficiency
        for v in range(0, 4):
            vs = basis_dict[v]
            v_results = {x:value for x, value in exp_results.items() if x[j]+x[k]==vs }
            filtered_dict = {}
            for x, value_x in v_results.items():
                filtered_values = {y:value_y for y, value_y in value_x.items() if outeq(x, y, j, k) }
                filtered_dict[x] = filtered_values
            total_count = sum({k:sum(val.values()) for k, val in filtered_dict.items()}.values())
            for w in range(0, 4):
                ws = basis_dict[w]
                final_count = 0
                for x, value_x in filtered_dict.items():
                    final_sum = sum([value_y for y, value_y in value_x.items() if y[j]+y[k]==ws])
                    final_count += final_sum
                if total_count > 0:
                    A[w, v] = final_count / total_count
                else:
                    raise Exception("Total count equal zero for j = ", j, "k = ", k, "v = ", vs)
        G = scipy.linalg.logm(A)
        if np.iscomplex(G).any():
            G = np.zeros_like(A)
            print("Warning: matrix log has complex entries, setting rates to 0.")
        G_prime = G.clip(min=0)
        G_dict[(j,k)] = G_prime
        rates.append((j, k, "00","11", G_prime[rbd["11"], rbd["00"]]))
        rates.append((j, k, "11","00", G_prime[rbd["00"], rbd["11"]]))
        rates.append((j, k, "01","10", G_prime[rbd["01"], rbd["10"]]))
        rates.append((j, k, "10","01", G_prime[rbd["10"], rbd["01"]]))
    for j in range(n):
        d = {k:v for k, v in G_dict.items() if j in k}
        # formula is different than in the paper since here always j \leq k
        r01 = 1/(2*(n-1)) *sum([G[rbd["10"], rbd["00"]] + G[rbd["01"], rbd["00"]] +
                                G[rbd["11"], rbd["01"]] + G[rbd["11"], rbd["10"]]
                                for G in d.values()])
        r10 = 1/(2*(n-1)) *sum([G[rbd["00"], rbd["10"]] + G[rbd["00"], rbd["01"]] +
                                G[rbd["01"], rbd["11"]] + G[rbd["10"], rbd["11"]]
                                for G in d.values()])
        rates.append((j, j, "0", "1", r01))
        rates.append((j, j, "1", "0", r10))
    return rates
