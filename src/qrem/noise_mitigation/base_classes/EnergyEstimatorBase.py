"""
@author: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
@contact: filip.b.maciejewski@gmail.com
"""

from typing import Optional, Dict, Tuple, List
import abc
import numpy as np
from qrem.functions_qrem import functions_data_analysis as fda
from qrem.noise_characterization.base_classes.marginals_analyzer_base import MarginalsAnalyzerBase # used only to compute marginals
from qrem.noise_mitigation.base_classes.EnergyEstimatorInterface import EnergyEstimatorInterface
from tqdm import tqdm


class EnergyEstimatorBase(EnergyEstimatorInterface):
    """
    This is base class for estimating corrected energy of some Hamiltonian.

    In this class and its children, we use the following convention for:

     1. Generic experimental results:
    :param results_dictionary: Nested dictionary with following structure:

    results_dictionary[label_of_experiment][bitstring_outcome] = number_of_occurrences

    where:
        -label_of_experiment is arbitrary label for particular experiment,
        -bitstring_outcome is label for measurement outcome,
        -number_of_occurrences is number of times that bitstring_outcome was recorded

        Hence top-level key labels particular experiment
        (one can think about quantum circuit implementation)
        and its value is another dictionary with results of given experiment in the form
        of dictionary of measurement outcomes


    2. hamiltonian_dictionaries:
        :param hamiltonian_dictionaries: Nested dictionary with the following structure:

        hamiltonian_dictionaries[label_of_experiment][label_of_subset] = weight in decomposition
                                                                    of the Hamiltonian into Z-Paulis

        where:
            -label_of_experiment is the same as in results_dictionary and it labels results from which
            marginal distributions were calculated
            -label_of_subset is a label for qubits subset for which marginals_dictionary were calculated.
            We use convention that such label if of the form "q5q8q12..." etc., hence it is bitstring of
            qubits labels starting from "q".
            -marginal_probability_vector marginal distribution stored as vector

    """

    def __init__(self,
                 results_dictionary: Dict[str, Dict[str, int]],
                 hamiltonian_dictionaries: Dict[str, Dict[Tuple[int], float]],
                 energies_dictionary_raw: Optional[Dict[str, float]] = None
                 ) -> None:

        """
        :param results_dictionary: see class description
        :param hamiltonian_dictionaries: see class description
        """

        # set initial values of class properties
        self._results_dictionary = results_dictionary
        self._hamiltonian_dictionaries = hamiltonian_dictionaries

        if energies_dictionary_raw is None:
            energies_dictionary_raw = {}

        self._energies_dictionary_raw = energies_dictionary_raw

        self._energies_dictionary_corrected = {}


        results_of_exemplary_experiment = list(results_dictionary.values())[0]
        self._number_of_qubits = len(list(results_of_exemplary_experiment.keys())[0])

    @property
    def results_dictionary(self) -> Dict[str, Dict[str, int]]:

        return self._results_dictionary

    @results_dictionary.setter
    def results_dictionary(self, results_dictionary: Dict[str, Dict[str, int]]) -> None:
        self._results_dictionary = results_dictionary

    @property
    def hamiltonian_dictionaries(self) -> Dict[str, Dict[Tuple[int], float]]:
        return self._hamiltonian_dictionaries

    @hamiltonian_dictionaries.setter
    def hamiltonian_dictionaries(self,
                                 hamiltonian_dictionaries: Dict[
                                     str, Dict[Tuple[int], np.ndarray]]) -> None:
        self._hamiltonian_dictionaries = hamiltonian_dictionaries

    @property
    def energies_dictionary_raw(self) -> Dict[str, float]:

        return self._energies_dictionary_raw

    @property
    def energies_dictionary_corrected(self) -> Dict[str, float]:
        return self._energies_dictionary_raw

    def estimate_energy(self,
                        experiment_key: str,
                        method: Optional[str] = 'direct') -> float:

        experimental_results = self._results_dictionary[experiment_key]
        hamiltonian_weights = self._hamiltonian_dictionaries[experiment_key]

        if method.upper() == 'DIRECT':
            estimated_energy = fda.estimate_energy_from_counts_dictionary(
                counts_dictionary=experimental_results,
                weights_dictionary=hamiltonian_weights)

        elif method.upper() == 'MARGINALS':       # MOcomm loopy usage of marginal analyzer base
            marginals_analyzer = MarginalsAnalyzerBase(results_dictionary=self._results_dictionary,
                                                       bitstrings_right_to_left=False)
            estimated_marginals = marginals_analyzer.get_marginals(experiment_key=experiment_key,
                                                                   subsets_list=list(
                                                                       hamiltonian_weights.keys()))

            estimated_energy = fda.estimate_energy_from_marginals(
                weights_dictionary=hamiltonian_weights,
                marginals_dictionary=estimated_marginals)
        else:
            raise ValueError(f"Method name '{method}' unrecognized.\n "
                             f"Please choose either 'direct' or 'marginals' method.")

        self._energies_dictionary_raw[experiment_key] = estimated_energy

        return estimated_energy

    def estimate_multiple_energies(self,
                                   experiment_keys: List[str],
                                   method: Optional[str] = 'direct'):

        for experiment_label in tqdm(experiment_keys):
            self.estimate_energy(experiment_key=experiment_label,
                                 method=method)
        return self._energies_dictionary_raw
