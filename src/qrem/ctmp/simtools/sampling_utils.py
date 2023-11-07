from typing import Dict, Tuple, List

# checks if transition is available for the given input state
def check_transition(input_state: str, transition: Tuple) -> bool:
    i, j, error_in, _, _ = transition
    if i == j:
        return input_state[i] == error_in
    else:
        return input_state[i] + input_state[j] == error_in

def apply_transition(input_state: str, transition: Tuple) -> str:
    i, j, _, error_out, _ = transition
    state = list(input_state)
    if i == j:
        state[i] = error_out
    else:
        state[i], state[j] = error_out
    return ''.join(state)
