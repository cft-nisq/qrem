
from typing import Optional, List, Dict, Union
import numpy as np

from qrem.functions_qrem import ancillary_functions as anf

#ORGANISATIION:  At this moment this class is used only as a parent class to OverlappingTomographyBase.
#  In principle it owuld be good to have a it for futher generalisations (MO)
#ORGANISATION: function file does not start with capital letters (MO)

#TODO_MO (PP) This class is more about gennerating circuits, We should
# - create Circuit class. Circuits will contain circuit type definnition QCircuit =NDArray, conversions, and creation/ranndomisation functions, i/o
# - channge name to CharacterisationCircuits - as this seesm to be the merit of this class
# Further: OverlappingTomogtaphyBase should be changed into OverlappingTomography
# OverlappingTomography should get CharacterisationCircuits as an input, not as an inheritance




class OverlappingEstimationBase:
    def __init__(self,
                 number_of_qubits: int,
                 number_of_symbols: int,
                 subsets: Union[Dict[str, List[int]], List[List[int]]],
                 # maximal_circuits_amount: Optional[int] = 1500,       #THIS can potentially be restored
                 show_progress_bars: Optional[bool] = True,
                 ):

        self._number_of_symbols = number_of_symbols
        self._number_of_qubits = number_of_qubits
        self._subsets = subsets
       # self._maximal_circuits_amount = maximal_circuits_amount  #THIS can potentially be restored

        # a list that will store lists of integers that label characterisation cir
        self._circuits_list = []

        self._show_progress_bars = show_progress_bars


    # properties for restoring "inputs" of the constructor - on high level redundant

    #ALG: can we be faster when we use numpy lists here


    #------------------
    @property
    def number_of_symbols(self) -> int:
        return self._number_of_symbols

    @number_of_symbols.setter
    def number_of_symbols(self, number_of_symbols: int) -> None:
        self._number_of_symbols = number_of_symbols

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

    # MOquestion - what about random seed?
    def get_random_circuit(self):
        return [np.random.randint(0, self._number_of_symbols) for _ in
                range(self._number_of_qubits)]
    # Gives at output a random list of integers (from colleection of size self._number_of_symbols), where leangth correspond to number of qubits

    def get_random_circuit_with_fixed_states_of_some_qubits(self,
                                                           fixed_states: Dict[int, int]):
        """
        :param fixed_states: dictionary where each KEY is index of qubit,
                            and VALUE denotes symbols' state
        :return:
        """
# Generates fixes state on some subset of qubits, random (from   colection of size self._number_of_symbols) and appends it to internal list of circuit
# MOcomm- update documentation
        fixed_qubits = fixed_states.keys()

        circuit = []

        for qubit_index in range(self._number_of_qubits):
            if qubit_index in fixed_qubits:
                circuit.append(fixed_states[qubit_index])
            else:
                circuit.append(np.random.randint(0, self._number_of_symbols))

        return circuit

    def add_circuits(self,
                     circuits: List[List[Union[str,int]]]):
        for circuit in circuits:
            self._circuits_list.append(circuit)

    # appends a list of random  circuits of leangth number_of_circuits to internal circuit list
    def get_random_circuits_list(self,
                                 number_of_circuits: int,
                                 fixed_states: Optional[Dict[int, int]] = None):
        if fixed_states is None:
            # MOcomm - not sure about efficiency of this procedure - using numpy maybe faster
            return [self.get_random_circuit() for _ in range(number_of_circuits)]
        else:
            # MOcomm - not sure about efficiency of this procedure - using numpy maybe faster
            return [self.get_random_circuit_with_fixed_states_of_some_qubits(fixed_states) for _ in
                    range(number_of_circuits)]

    # very similar name to get_random_circuit function defined previously
    def add_random_circuits(self,
                            number_of_circuits: int,
                            fixed_states: Optional[Dict[int, int]] = None):

        self.add_circuits(self.get_random_circuits_list(number_of_circuits, fixed_states))

    # adds random circuits (encoded by integer list) to list of circuits
#MOcomm - improve documentation
