import numpy as np
from qrem.ctmp.mitigtools.observables_utils import convert, random_element, sample, evaluate, delta
from qrem.functions_qrem import povmtools
from qrem.types import CTMPModelData
from typing import Dict, Tuple, List, Union
from qrem.common import probability




def _mitigate(model: CTMPModelData,
              output_counts: Dict[str, int],
              input_obs,
              T: int,
              mode: str,
              proj: bool=False) -> np.array:
    """ Mitigate either an observable or marginal, defined by mode. 

    Parameters
    ----------
    Input_obs: Union[List,Tuple]
         Input_obs = observable given as list of coefficients  if mode=obs or a tuple of qubits if mode=marginal
    """
    # todo: check input correctness
    # check statistical correctness
    # M = sum(output_counts.values())
    c1 = np.exp(2*model.gamma)
    if mode == "obs":
        xi = np.zeros(T)
        # convert observable description into an array
        obs = convert(input_obs, model.n)
    if mode == "marginal":
        xi = np.zeros((T, 2**len(input_obs)))
    # Algorithm 1 from the paper
    for t in range(T):
        alpha = np.random.poisson(model.gamma)
        s = random_element(output_counts)
        #i = np.random.randint(0, M)
        x = sample(model, alpha, s) #string of 0s and 1s representing output sample
        if mode == "obs":
            x = np.array([int(v) for v in x]) # maybe change this later
            xi[t] = (-1)**alpha * evaluate(obs, x)
        if mode == "marginal":
            xi[t] = (-1)**alpha * delta(x, input_obs)
    result = xi.mean(axis=0) * c1
    if mode == "marginal" and proj:
        result = result / result.sum() # normalize to sum 1
        if not probability.is_valid_probability_vector(result):
            result = povmtools.find_closest_prob_vector_l2(result).flatten()
    return result
