import numpy as np
from qrem.ctmp.mitigtools.mitig_utils import _mitigate
from qrem.types import CTMPModelData
from typing import Dict, Tuple, List



def mitigate_expected_value(observable: List[List],
                            results_dictionary: Dict[str, Dict[str, int]],
                            noise_model: CTMPModelData,
                            T: int) -> float:
    """Mitigation procedure as described in Bravyi et al. Returns mitigated expectation value of 
    the observable.

    model - the CTMP model instance
    exp_results - dictionary of experimental results (each result = string of 0s and 1s)
    observable - observable O whose expected value is to be mitigated, given as a list of coefficients
    in strings of Z, e.g. [ [(), 1], [(0,1,3), -2]] denotes O = I - 2 * ZZIZ
    T - number of timesteps of the algorithm"""
    result = {}
    for k, v in results_dictionary.items():
        result[k] = _mitigate(noise_model, v, observable, T, "obs")
    return result



def mitigate_marginals(marginals_list: List[Tuple],
                       results_dictionary: Dict[str, Dict[str, int]],
                       noise_model: CTMPModelData,
                       T: int,
                       ensure_proper_probability_distribution: bool=False)->Dict[str, Dict[Tuple[int], np.array]]:
    """Mitigation procedure as described in Bravyi et al. Returns list of mitigated marginals.

    model - the CTMP model instance
    exp_results - dictionary of experimental results 
    marginals - list of marginals to be mitigated e.g. [(0,1), (2,3,4)]
    T - number of timesteps of the algorithm
    proj - boolean variable indicating whether resulting marginals should be projected onto physical ones
    """
    result = {}
    for k, v in results_dictionary.items():
        marginal_results = {}
        for marginal in marginals_list:
            marginal_results[marginal] = _mitigate(noise_model, v, marginal, T, "marginal", ensure_proper_probability_distribution)
        result[k] = marginal_results
    return result
