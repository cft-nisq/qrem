from qrem.ctmp.simtools.sampling_utils import check_transition, apply_transition
from qrem.types import CTMPModelData
from typing import Dict, Tuple, List
import numpy as np
import random


def generate_sample(CTMP_model: CTMPModelData,
                    input_state: str,
                    t: float=1) -> str:
    """Generates a sample according to the stochastic matrix exp(Gt) of the  CTMP model 
    acting on a classical input state by using Gillespie's algorithm. Default t=1.
    """
    if len(input_state) != CTMP_model.n:
        raise Exception("Input state size and model size do not match")
    time = 0
    next_state = input_state
    while time < t:
        current_state = next_state
        # check which transitions are available in the current state
        transitions = list(filter(lambda x: check_transition(current_state, x), CTMP_model.rates))
        if len(transitions) == 0:
            # stuck in a state where no transitions are available
            time = t
        else:
            # Gillespie step
            transition_rates = [error[4] for error in transitions]
            total_rate = sum(transition_rates)
            jump = np.random.exponential(1 / total_rate)
            time += jump
            # pick random transition
            k = random.choices(range(len(transition_rates)), [x/total_rate for x in transition_rates])[0]
            next_state = apply_transition(current_state, transitions[k])
    return current_state
