"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com

REFERENCES:
[0] Filip B. Maciejewski, Zoltán Zimborás, Michał Oszmaniec,
"Mitigation of readout noise in near-term quantum devices
by classical post-processing based on detector tomography",
Quantum 4, 257 (2020)

[0.5] Filip B. Maciejewski, Flavio Baccari, Zoltán Zimborás, Michał Oszmaniec,
"Modeling and mitigation of cross-talk effects in readout noise 
with applications to the Quantum Approximate Optimization Algorithm",
Quantum 5, 464 (2021).

"""

from typing import Optional, List, Dict, Union

import numpy as np
 
from qrem.noise_characterization.tomography_design.overlapping import overlapping_tomography_functions as otf

class LabelsBaseDDOT:
    def __init__(self,
                 number_of_qubits: int,
                 subsets: Optional[Union[Dict[str, List[int]], List[List[int]]]] = None,
                 maximal_circuits_amount: Optional[int] = 1500,
                 show_progress_bars: Optional[bool] = True,
                 ):
        self._number_of_qubits = number_of_qubits
        self._subsets = subsets
        self._maximal_circuits_amount = maximal_circuits_amount
        self._circuits_list = []

        self._show_progress_bars = show_progress_bars



    @property
    def subsets(self) -> Union[Dict[str, List[int]], List[List[int]]]:
        return self._subsets

    @subsets.setter
    def subsets(self, subsets: Union[Dict[str, List[int]], List[List[int]]]) -> None:
        self._subsets = subsets

    @property
    def circuits_list(self) -> List[List[int]]:
        return self._circuits_list

    @circuits_list.setter
    def circuits_list(self, circuits_list: List[List[int]]) -> None:
        self._circuits_list = circuits_list

    def get_random_circuit(self,
                           number_of_qubits:int=None)->List[int]:

        if number_of_qubits is None:
            number_of_qubits = self._number_of_qubits
        return [np.random.randint(0, 2) for _ in
                range(number_of_qubits)]

    def get_random_circuit_with_fixed_state_of_some_qubits(self,
                                                           fixed_states: Dict[int, int])->List[int]:
        """
        :param fixed_states: dictionary where each KEY is index of qubit,
                            and VALUE denotes qubit's state (0 or 1)
        :return:
        """

        fixed_qubits = fixed_states.keys()

        circuit = []
        for qubit_index in range(self._number_of_qubits):
            if qubit_index in fixed_qubits:
                circuit.append(fixed_states[qubit_index])
            else:
                circuit.append(np.random.randint(0, 2))

        return circuit

    def add_circuits(self,
                     circuits: List[List[str]]) -> None:
        for circuit in circuits:
            self._circuits_list.append(circuit)

    def get_random_circuits_list(self,
                                 number_of_circuits: int,
                                 fixed_states: Optional[Dict[int, int]] = None)->List[List[int]]:
        if fixed_states is None:
            return [self.get_random_circuit() for _ in range(number_of_circuits)]
        else:
            return [self.get_random_circuit_with_fixed_state_of_some_qubits(fixed_states) for _ in
                    range(number_of_circuits)]

    def add_random_circuits(self,
                            number_of_circuits: int,
                            fixed_states: Optional[Dict[int, int]] = None):

        self.add_circuits(self.get_random_circuits_list(number_of_circuits, fixed_states))

    def get_random_collection(self,
                              number_of_circuits: int,
                              locality: int,
                              desired_accuracy: float = 0.005):
        # FBM finish

        1

        # suggested_number_of_circuits.

    def get_parallel_tomography_on_subsets(self,
                                           number_of_circuits: int,
                                           non_overlapping_subsets:Optional[List[List[int]]]=None,
                                           ) -> List[List[int]]:
        """

        :param number_of_circuits: should be power of 2
        :param subsets:
        :param number_of_symbols:
        :return:
        :rtype:
        """

        if non_overlapping_subsets is None:
            non_overlapping_subsets = self._subsets

        number_of_symbols = 2

        return otf.get_parallel_tomography_on_non_overlapping_subsets(number_of_circuits=number_of_circuits,
                                                                      number_of_symbols=number_of_symbols,
                                                                      non_overlapping_subsets=non_overlapping_subsets)