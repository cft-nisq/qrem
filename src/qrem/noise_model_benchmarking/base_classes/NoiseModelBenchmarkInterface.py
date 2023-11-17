"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""
import abc
from typing import List, Dict, Tuple, Union


class MarginalsAnalyzerInterface(abc.ABC):
    """
    This is interface for classes that will analyze marginal probability distributions.
    It requires those child classes to have basic functionalities that should be included.
    for analyzing marginals_dictionary.
    """

    @property
    @abc.abstractmethod
    def benchmarking_results(self) -> Dict[str, Dict[str, float]]:
        # dictionary of experimental results
        raise NotImplementedError

    @benchmarking_results.setter
    @abc.abstractmethod
    def benchmarking_results(self, benchmarking_results: Dict[str, Dict[str, float]]) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def noisy_marginals_dictionary(self) -> dict:
        # Dictionary storing marginal probability distributions
        raise NotImplementedError


    @noisy_marginals_dictionary.setter
    @abc.abstractmethod
    def noisy_marginals_dictionary(self, marginals_dictionary: dict) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_noisy_marginals(self,
                          experiment_key: str,
                          subsets: List[List[int]]) -> dict:
        """Computes marginals_dictionary for input subsets_list of qubits"""
        raise NotImplementedError

    def compute_hamiltonian_energy_on_classical_state(self,
                                                      weights_dictionary: Dict[Tuple[int], float],
                                                      input_state=Union[str, List[str]]) -> float:
        """Computes energy of diagonal Hamiltonian for given classical input state"""
        raise NotImplementedError
