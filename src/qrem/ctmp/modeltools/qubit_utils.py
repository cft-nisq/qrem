from typing import Dict, Tuple, List



def outeq(x: str, y: str, j: int, k:int) -> bool:
    """x, y - bit strings
    j, k - indices

    Returns True if x=y outside indices {j, k} and False otherwise.
    """
    # todo: what if n=2 ?
    return x[:j] + x[j+1:k] + x[k+1:] == y[:j] + y[j+1:k] + y[k+1:]
