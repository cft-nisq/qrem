import numpy as np
import random
from typing import Dict, Tuple, List
from qrem.qtypes import CTMPModelData
from qrem.ctmp.simtools.sampling_utils import check_transition, apply_transition



"""
Converts a description of an observable obs given by coefficients in Z's to a numpy array; 
    last column = coefficient.

E.g. [ [(), 1], [(0,1,3), -2]] denotes O = I - 2 * ZZIZ and is converted into:
0, 0, 0, 0, 1
1, 1, 0, 1, -2

Z - list of coefficients in the above format
n - number of qubits
"""
def convert(obs: List[List], n: int) -> np.array:
    result = np.zeros((len(obs), n+1))
    for k in range(len(obs)):
        result[k, obs[k][0]] = 1
        result[k, -1] = obs[k][1]
    return result

"""
Computes obs(x), where obs is a numpy array describing an observable as above and x is an array of 0 and 1
"""
def evaluate(obs: List[List], x: np.array) -> float:
    parities = np.einsum("ij,j->i", obs[:,:-1], x).astype(int) % 2
    return ((-1)**parities * obs[:,-1]).sum()


"""
Return a sample x (array of 0s and 1s) from the distribution <x| B^alpha |s>, where B = I + gamma^{-1}G
input_state - |s>, string of 0s and 1s
"""
def sample(model: CTMPModelData, alpha: int, input_state: str) -> str:
    gamma = model.gamma
    current_state = input_state
    for t in range(alpha):
        transitions = list(filter(lambda x: check_transition(current_state, x), model.rates))
        if len(transitions) > 0:
            transition_rates = [error[4] for error in transitions]
            total_rate = sum(transition_rates)
            p = 1 - total_rate / gamma # probability of staying in place according to B
            if p < 0:
                raise Exception("Invalid gamma parameter - transition matrix not stochastic")
            if np.random.uniform() > p:
                # if not staying in place, pick random transition
                k = random.choices(range(len(transition_rates)), [x/total_rate for x in transition_rates])[0]
                current_state = apply_transition(current_state, transitions[k])
    return current_state

"""Arguments:
x - bit string of size n
indices - tuple representing a set of indices S

Returns a 2^|S| vector with 1 at index corresponding to position y such that x restricted to S equals y
and 0 otherwise.
"""
def delta(x: str, indices: Tuple[int]) -> np.array:
    result = np.zeros(2**len(indices))
    pos = int(''.join([x[i] for i in indices]), 2)
    result[pos] = 1
    return result

"""Returns a random element from probability distribution given by counts, where e.g.
counts = {"000": 70, "001": 123, ...}
"""
def random_element(counts: Dict[str, int]) -> str:
    s = sum(counts.values())
    d = {k: v/float(s) for k, v in counts.items()}
    return np.random.choice(list(d.keys()), 1, p=list(d.values()))[0]
