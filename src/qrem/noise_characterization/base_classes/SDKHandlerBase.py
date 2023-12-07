from collections import Counter
from typing import Optional, List, Dict, Union


#ORGANIZE - this has to be globally accessible list (MO)
__valid_SDK_names__ = ['QISKIT',
                       # 'PYQUIL',
                       'PYQUIL-FOR-AZURE-QUANTUM']


class SDKHandlerBase:
    def __init__(self,
                 SDK_name: str,
                 qubit_indices: List[int],
                 descriptions_of_circuits: Union[Dict[str, int], List[List[int]]],
                 experiment_name: str,
                 number_of_repetitions: Optional[int] = 1,
                 quantum_register_size: Optional[int] = None,  # MOcomm this variable does not seam to be used in relevant way - refactoring needed
                 classical_register_size: Optional[int] = None,  # MOcomm this variable does not seam to be used in relevant way - refactoring needed
                 add_barriers=True,
                 pyquil_compilation_method='parametric-native',
                 temporary_helper_argument=True,
                 ):

        if SDK_name.upper() not in __valid_SDK_names__:
            raise ValueError('Backend: ' + SDK_name + ' is not supported yet.')

        self._SDK_name = SDK_name
        self._experiment_name = experiment_name
        self._qubit_indices = qubit_indices

        self.pyquil_compilation_method = pyquil_compilation_method

        # FBM: very temporary
        if isinstance(descriptions_of_circuits, list) and temporary_helper_argument:
            self._circuits_labels_list = descriptions_of_circuits
            circuits_labels_strings_list = [''.join([str(symbol) for symbol in symbols_list]) for
                                            symbols_list
                                            in
                                            descriptions_of_circuits]
            descriptions_of_circuits = dict(zip(Counter(circuits_labels_strings_list).keys(),
                                       Counter(circuits_labels_strings_list).values()))
            if number_of_repetitions > 1:
                for key in descriptions_of_circuits.keys():
                    descriptions_of_circuits[key] *= number_of_repetitions

        self._circuit_labels = descriptions_of_circuits
        self._quantum_register_size = quantum_register_size
        self._classical_register_size = classical_register_size
        self._add_barriers = add_barriers

    @staticmethod
    def _add_measurements_qiskit(
            circuit_object,
            qreg,
            creg,
            qubit_indices):
        for qubit_index in range(len(qubit_indices)):
            # print(qubit_indices)
            # print(qreg)
            # print(creg)
            circuit_object.measure(qreg[qubit_indices[qubit_index]], creg[qubit_index])

        return circuit_object

    @staticmethod
    def _add_measurements_pyquil(
            quantum_program,
            qubit_indices,
            classical_register):
        from pyquil.gates import MEASURE
        for qubit_index in range(len(qubit_indices)):
            quantum_program += MEASURE(qubit_indices[qubit_index],
                                       classical_register[qubit_index])
        return quantum_program

    def _add_measurements(self,
                          quantum_circuit,
                          qubit_indices,
                          classical_register,
                          quantum_register=None):

        1
        # FBM: finish this

#
    def __create_circuit_qiskit(self,
                                quantum_register_size: int,
                                classical_register_size: int,
                                # qubit_indices: List[int],
                                # circuit_label_list: List[Union[int, str]],
                                circuit_name: str,
                                # add_measurements: bool
                                ):
#MOcomm -imports next to evaluation, not elegant!
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        # create registers
        qreg, creg = QuantumRegister(quantum_register_size), \
                     ClassicalRegister(classical_register_size)
        # print(qreg)
        circuit_object = QuantumCircuit(qreg,
                                        creg,
                                        name=circuit_name)

        return circuit_object, qreg, creg

#MOcomm - what is classical register size below?

    def __create_circuit_pyquil(self,
                                # quantum_register_size: int,
                                classical_register_size: int,
                                qubit_indices: List[int], # not used inside the method, but returned by it
                                # pyquil_compilation_method:str='parametric-native'
                                # circuit_label_list: List[Union[int, str]],
                                # circuit_name: str,
                                # add_measurements: bool
                                ):

        from pyquil import Program

        circuit_object = Program()
        classical_register = circuit_object.declare('ro',
                                                    'BIT',
                                                    classical_register_size)

        return circuit_object, qubit_indices, classical_register

    def __create_circuit_braket(self,
                                # quantum_register_size: int,
                                classical_register_size: int,
                                qubit_indices: List[int],
                                qubi
                                ):
        from braket.circuits import Circuit
        circuit_object = Circuit()
        return circuit_object


    def _create_circuit(self,
                        classical_register_size,
                        qubit_indices=None,
                        circuit_name=None,
                        quantum_register_size=None):
        # TODO JM, FBM: add braket

        if self._SDK_name.upper() in ['QISKIT']:
            if quantum_register_size is None:
                raise ValueError("Please provide quantum register size for qiskit!")
            if circuit_name is None:
                raise ValueError("Please provide circuit name for qiskit!")
            circuit_object, qreg, creg = self.__create_circuit_qiskit(quantum_register_size=quantum_register_size,
                                                                      classical_register_size=classical_register_size,
                                                                      # qubit_indices=qubit_indices,
                                                                      circuit_name=circuit_name)

        elif self._SDK_name.upper() in ['PYQUIL', 'PYQUIL-FOR-AZURE-QUANTUM']:
            if qubit_indices is None:
                raise ValueError("Please provide qubits indices for pyquil!")
            circuit_object, qreg, creg = self.__create_circuit_pyquil(classical_register_size=classical_register_size,
                                                                      qubit_indices=qubit_indices)

        elif self._SDK_name.upper() in ['BRAKET']:
            circuit_object = self.__create_circuit_braket()
            qreg = None
            creg = None

        return circuit_object, qreg, creg

#
#
