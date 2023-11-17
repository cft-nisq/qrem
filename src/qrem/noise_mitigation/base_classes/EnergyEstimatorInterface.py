"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""

import abc


class EnergyEstimatorInterface(abc.ABC):
    """
    This is interface for classes that will perform error mitigation for energy estimation.
    """

    @property
    @abc.abstractmethod
    def results_dictionary(self) -> dict:
        # dictionary of experimental results
        raise NotImplementedError

    @results_dictionary.setter
    @abc.abstractmethod
    def results_dictionary(self, results_dictionary: dict) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def hamiltonian_dictionaries(self) -> dict:
        # Dictionary storing information about Hamiltonians weights
        raise NotImplementedError

    @hamiltonian_dictionaries.setter
    @abc.abstractmethod
    def hamiltonian_dictionaries(self, hamiltonian_dictionaries: dict) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def energies_dictionary_raw(self) -> dict:
        # Dictionary storing information about estimated energies (raw)
        raise NotImplementedError

    @energies_dictionary_raw.setter
    @abc.abstractmethod
    def energies_dictionary_raw(self,energies_dictionary_raw:dict) -> dict:
        # Dictionary storing information about estimated energies (raw)
        raise NotImplementedError


    # @property
    # @abc.abstractmethod
    # def energies_dictionary_corrected(self) -> dict:
    #     # Dictionary storing information about estimated energies (corrected)
    #     raise NotImplementedError
    #
    # @energies_dictionary_corrected.setter
    # @abc.abstractmethod
    # def energies_dictionary_corrected(self,energies_dictionary_corrected:dict) -> dict:
    #     # Dictionary storing information about estimated energies (corrected)
    #     raise NotImplementedError

    @abc.abstractmethod
    def estimate_energy(self,
                        experiment_key: str) -> float:
        raise NotImplementedError


    #
    # @abc.abstractmethod
    # def estimate_corrected_energy(self,
    #                               experiment_key: str
    #                               ) -> float:
    #     raise NotImplementedError
