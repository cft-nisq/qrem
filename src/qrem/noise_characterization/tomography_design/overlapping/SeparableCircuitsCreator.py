from typing import Optional, List, Dict, Union

from qrem.noise_characterization.base_classes.SDKHandlerBase import SDKHandlerBase

# TODO: check what do names other than ddot and qdot mean?
__valid_experiments_names__ = ['ddot',
                               'ddt',
                               'qdot',
                               'qdt',
                               'classical_states',
                               'separable_classical_states',
                               'separable_states',
                               'separable_quantum_states']



#(PP) This actually connnnects with the external SDKs: converts format to qiskit or pyqil and prepares circuits
class SeparableCircuitsCreator(SDKHandlerBase):
    """
    This is general class for creating quantum circuits implementing separable states on a quantum
    device. This is done based on provided list of circuits labels and chosen "interpreter" of those
    labels (see descriptions below).

    NOTE: please note that this class' is used only to create QuantumCircuit objects that can be passed
    to execution on some backend. It is assumed that specific labels describing circuits (see below)
    already have been generated.

    MAIN PARAMETERS:
    :param SDK_name:
    :param qubit_indices: one always needs to provide list of qubit indices on which experiments
                          should be performed (this does not need to be the whole device)

    :param descriptions_of_circuits: main argument to the class, this is list of symbolic circuits description

    Two formats are accepted:
    """
    # ORGANIZATION - in my opinion it is confucing that we have two arguments for qubit indices
    """
    1. List[List[int]] - integer indicates gate to be implemented on
                         corresponding qubit.
                         For example: circuit_labels = [[0,1,5],
                                                        [2,0,1]]

                         will implement circuits:
                         gate "0" on qubit_indices[0],
                         gate "1" on qubit_indices[1],
                         gate "5" on qubit_indices[2]

                         gate "2" on qubit_indices[0],
                         gate "0" on qubit_indices[1],
                         gate "1" on qubit_indices[2]

    2. Dict[str,int] - dictionary where each KEY is string of integers describing
                      a circuit (in manner analogous to described above),
                      and VALUE is the number of times that this circuit should be implemented

    Now the meaning of the 'gate i' depends on the type of experiment.
    For example, if one wishes to perform Diagonal Detector Overlapping Tomography (DDOT) [0.5],
    then there will be only two gates - idle and X gate, labeled by 0 and 1, respectively.

    In general, interpretation of symbols in descriptions_of_circuits parameter is done via something which
    we call here "circuits interpreter" (one of classes' properties).

    If one wishes to perform one of the available experiments (see __valid_experiments_names__ list),
    there is no need to specify such interpreter.

    One can add custom circuits interpreter using method "add_custom_circuits_interpreter".
    Alternatively, one can provide interpreter by passing suitable argument to method
    "__initialize_circuits_interpreter" (see description of the method).


    :param experiment_name: string describing type of experiment. It is used to initialize existing
                            circuits interpreters.

    possible values:

    1. 'DDOT' - implements Diagonal Detector Overlapping Tomography
    2. 'QDOT' - implements Quantum Detector Overlapping Tomography



    """

    def __init__(self,
                 SDK_name: str,
                 experiment_name: str,
                 qubit_indices: List[int],
                 descriptions_of_circuits: Union[Dict[str, int], List[List[int]]],
                 number_of_repetitions: Optional[int] = 1,
                 quantum_register_size: Optional[int] = None,
                 # MOcomm this variable does not seam to be used in relevant way - refactoring needed
                 classical_register_size: Optional[int] = None,
                 # MOcomm this variable does not seam to be used in relevant way - refactoring needed
                 add_barriers=True,
                 pyquil_compilation_method='parametric-native'):

        """
        :param SDK_name: see parent class' description
        :param experiment_name: see class' description
        #TODO FBM: check how important is that
        :param qubit_indices: see class' description
        :param descriptions_of_circuits: see class' description
        :param number_of_repetitions: each circuit in circuit_labels will be implemented
                                      this number of times
                                      NOTE: if there are doubles in the list, they will be counted as
                                        multiple circuits.
                                        For example: circuit_labels = [ [0,1,0], [0,1,0] ]
                                        with number_of_repetitions = 3
                                        will implement 6 circuits [0,1,0].

        #TODO FBM: check if this is used
        :param quantum_register_size: is used by method _create_circuit in parent class
        :param classical_register_size: is used by method _create_circuit in parent class
        """
        # ORGANIZEATION I checked this class. It does noy use quantum/ classical register size, its parent does
        # still this class is porely documented
        """

        :param add_barriers:
        """

        super().__init__(SDK_name=SDK_name,
                         qubit_indices=qubit_indices,
                         descriptions_of_circuits=descriptions_of_circuits,
                         experiment_name=experiment_name,
                         number_of_repetitions=number_of_repetitions,
                         quantum_register_size=quantum_register_size,
                         classical_register_size=classical_register_size,
                         add_barriers=add_barriers,
                         pyquil_compilation_method=pyquil_compilation_method)

        # TODO FBM, JT: maybe modify to include more than Pauli
        if self._experiment_name.lower() in ['ddot',
                                             'ddt',
                                             'classical_states',
                                             'separable_classical_states']:
            self._circuits_interpreter = self.__circuits_interpreter_DDOT
        elif self._experiment_name.lower() in ['qdot',
                                               'qdt',
                                               'separable_quantum_states']:
            self._circuits_interpreter = self.__circuits_interpreter_QDOT
        else:

            raise ValueError(
                f"Experiments of type: '{self._experiment_name}' is not supported yet.")
        # print(self._circuits_interpreter)
        self._experiment_name = experiment_name

    @property
    def circuits_interpreter(self):
        return self._circuits_interpreter

    @circuits_interpreter.setter
    def circuits_interpreter(self, circuits_interpreter):
        self._circuits_interpreter = circuits_interpreter

    @property
    def circuit_labels_dictionary(self) -> Dict[str, int]:
        return self._circuit_labels

    # TODO FBM: opisac to lepiej i dodac ze to konkretnie dla Pauliego
    def __circuits_interpreter_DDOT(self,
                                    circuit_label_list,
                                    circuit_object,
                                    qubit_indices):
        """
        This method creates DDOT circuits in format native to the given SDK (specified by class attribute SDK_name),
        based on the passed circuit_label_list and qubit_indices, specifying what gates should be applied to what qubits.
        """

        # TODO FBM: make sure qreg does not need to be passed
        # [X] This loop goes through all passed circuits and translates them to qiskit format, into variable circuit_object
        if self._SDK_name.upper() in ['QISKIT']:
            for qubit_index in range(len(circuit_label_list)):
                qubit_now = qubit_indices[qubit_index]
                label_now = circuit_label_list[qubit_index]

                if label_now in [0, '0']:
                    circuit_object.id(qubit_now)
                elif label_now in [1, '1']:
                    circuit_object.x(qubit_now)
                else:
                    raise ValueError('Wrong circuit label: ', circuit_label_list)

        # [X] This loop goes through all passed circuits and translates them to pyquil format, into variable circuit_object
        # This loop is almost identical to the one above, they could be merged into one and separation into SDKs
        # could be done inside the loop
        elif self._SDK_name.upper() in ['PYQUIL', 'PYQUIL-FOR-AZURE-QUANTUM']:
            if self.pyquil_compilation_method.lower() in ['high_level']:
                from pyquil.gates import I, X
                for qubit_index in range(len(circuit_label_list)):
                    qubit_now = qubit_indices[qubit_index]
                    label_now = circuit_label_list[qubit_index]

                    if int(label_now) == 0:
                        circuit_object += I(qubit_now)
                    elif int(label_now) == 1:
                        circuit_object += X(qubit_now)
                    else:
                        raise ValueError('Wrong circuit label: ', circuit_label_list)
            elif self.pyquil_compilation_method.lower() in ['parametric-native']:
                from pyquil.quilbase import Declare
                from pyquil.quilatom import MemoryReference
                from pyquil.gates import RX, RZ

                for qubit_index in qubit_indices:
                    circuit_object += Declare(name=f'rz-{0}_q-{qubit_index}',
                                              memory_type='REAL',
                                              memory_size=1)

                for qubit_index in qubit_indices:
                    circuit_object += RX(MemoryReference(f'rz-{0}_q-{qubit_index}', qubit_index))
        return circuit_object

    # TODO FBM: opisac to lepiej i dodac ze to konkretnie dla Pauliego
    def __circuits_interpreter_QDOT(self,
                                    circuit_label_list,
                                    circuit_object,
                                    qubit_indices):
        if self._SDK_name.upper() in ['QISKIT']:
            for qubit_index in range(len(circuit_label_list)):
                qubit_now = qubit_indices[qubit_index]
                label_now = circuit_label_list[qubit_index]

                # TODO FBM: make sure qreg does not need to be passed
                if int(label_now) == 0:
                    circuit_object.id(qubit_now)
                elif int(label_now) == 1:
                    circuit_object.x(qubit_now)
                elif int(label_now) == 2:
                    circuit_object.h(qubit_now)
                elif int(label_now) == 3:
                    circuit_object.x(qubit_now)
                    circuit_object.h(qubit_now)
                elif int(label_now) == 4:
                    circuit_object.h(qubit_now)
                    circuit_object.s(qubit_now)
                elif int(label_now) == 5:
                    circuit_object.x(qubit_now)
                    circuit_object.h(qubit_now)
                    circuit_object.s(qubit_now)
                else:
                    raise ValueError(
                        f"Wrong circuit label: '{label_now} from circuit: '{circuit_label_list}'")


        elif self._SDK_name.upper() in ['PYQUIL', 'PYQUIL-FOR-AZURE-QUANTUM']:

            if self.pyquil_compilation_method.lower() in ['high_level']:
                from pyquil.gates import I, X, H, S
                for qubit_index in range(len(circuit_label_list)):
                    qubit_now = qubit_indices[qubit_index]
                    label_now = circuit_label_list[qubit_index]

                    if int(label_now) == 0:
                        circuit_object += I(qubit_now)
                    elif int(label_now) == 1:
                        circuit_object += X(qubit_now)
                    elif int(label_now) == 2:
                        circuit_object += H(qubit_now)
                    elif int(label_now) == 3:
                        circuit_object += X(qubit_now)
                        circuit_object += H(qubit_now)
                    elif int(label_now) == 4:
                        circuit_object += H(qubit_now)
                        circuit_object += S(qubit_now)
                    elif int(label_now) == 5:
                        circuit_object += X(qubit_now)
                        circuit_object += H(qubit_now)
                        circuit_object += S(qubit_now)
                    else:
                        raise ValueError('Wrong circuit label: ', circuit_label_list)

            elif self.pyquil_compilation_method.lower() in ['parametric-native']:
                from pyquil.quilbase import Declare
                from pyquil.quilatom import MemoryReference
                from pyquil.gates import RX, RZ
        # ORGANIZATION: this part of code is commented - I suggest to removing it (MO)
        # ORGANIZATION: this part is supposed to give us full control of what gates are used, instead of relying on the
        # compiler - I suggest we discuss (JM)
        # for qubit_index in qubit_indices:
        #     for angle_name in [f'rz-{0}', f'rx-{0}', f'rz-{1}', f'rx-{1}', f'rz-{0}']:
        #         circuit_object += Declare(name=f'{angle_name}_q-{qubit_index}',
        #                                   memory_type='REAL',
        #                                   memory_size=1)
        #
        # for qubit_index in qubit_indices:
        #     for angle_name in [f'rz-{0}', f'rx-{0}', f'rz-{1}', f'rx-{1}', f'rz-{0}']:
        #         if angle_name[1]=='z':
        #             circuit_object += RZ(MemoryReference(f'{angle_name}_q-{qubit_index}',qubit_index))
        #         elif angle_name[1]=='x':
        #             circuit_object += RX(
        #                 MemoryReference(f'{angle_name}_q-{qubit_index}', qubit_index))
        #         else:
        #             raise ValueError(f"WRONG ANGLE NAME {angle_name}")

        return circuit_object

    # def __initialize_circuits_interpreter(self,circuits_interpreter):

    def get_circuits(self,
                     # TODO JT: break this function down so it gets single circuit
                     # circuit_object:Optional[] = None
                     add_measurements: Optional[bool] = True):  # MOcomm not sure what is the meaning of this variable
        # TODO JT: modify this to allow also for process tomography 
        # MOcomm - most likelly we will do it defferently
        """

        Returns quantum circuits as a list.

        Circuits are later identified by names for which we use the following convention:

        circuit_name = "experiment name" + "-" + "circuit label"+
        "no"+ "integer identifier for multiple implementations of the same circuit"

        for example the circuit can have name:
        "DDOT-010no3"

        which means that this experiment is Diagonal Detector Overlapping Tomography (DDOT),
        the circuit implements state "010" (i.e., gates iden, X, iden on qubits 0,1,2), and
        in the whole circuits sets this is the 4th (we start counting from 0) circuit that implements
        that particular state.

        :param add_measurements:
        :return:
        """

        qubit_indices = self._qubit_indices

        quantum_register_size, classical_register_size = self._quantum_register_size, \
            self._classical_register_size

        qubit_indices = sorted(qubit_indices)  # Sort to ensure, that results can easily be interpreted.

        if self._SDK_name.lower() in ['pyquil', 'pyquil-for-azure-quantum']:
            if self.pyquil_compilation_method.lower() in ['parametric-native',
                                                          'parametric',
                                                          'native-parametric']:
                from qrem.backends_support.pyquil import pyquil_utilities

                list_of_circuit_labels = self._circuits_labels_list

                memory_map = pyquil_utilities.create_memory_map_DOT(list_of_keys=list_of_circuit_labels,
                                                                    qubit_indices=qubit_indices)

                return memory_map

        # why to copy local variables to some extra variables
        descriptions_of_circuits = self._circuit_labels

        if quantum_register_size is None:
            quantum_register_size = max(qubit_indices) + 1

        if classical_register_size is None:
            classical_register_size = len(qubit_indices)

        all_circuits = []

        unique_circuit_labels = list(descriptions_of_circuits.keys())

        # outer loop is for copies of DDOT experiment
        # for repetition_index in range(number_of_repetitions):
        # inner loop goes through all circuits in QDT experiments and prepares them
        for circuit_label_string in unique_circuit_labels:
            # this loop goes over multiple instances of the same experiments (if they exist)
            circuit_label_list = list(circuit_label_string)

            for multiple_circuits_counter in range(descriptions_of_circuits[circuit_label_string]):

                # [X] create quantum circuit with nice name:
                circuit_name = self._experiment_name + "-" \
                               + circuit_label_string + 'no%s' % multiple_circuits_counter
                # [X] create "empty" circuit object for the given SDK:
                circuit_object, quantum_register, classical_register = self._create_circuit(
                    classical_register_size=classical_register_size,
                    quantum_register_size=quantum_register_size,
                    qubit_indices=qubit_indices,
                    circuit_name=circuit_name)
                # [X] add appropriate gates to the "empty" circuit:
                circuit_object = self._circuits_interpreter(circuit_label_list,
                                                            circuit_object,
                                                            qubit_indices)



                # TODO JM, FBM: add braket-qiskit and pyquil-azure implementation

                if self._SDK_name.upper() in ['QISKIT']:
                    if self._add_barriers:
                        circuit_object.barrier()
                    if add_measurements:
                        # TODO JM, FBM: add single abstract method for adding measurements?
                        circuit_object = self._add_measurements_qiskit(circuit_object=circuit_object,
                                                                       qreg=quantum_register,
                                                                       creg=classical_register,
                                                                       qubit_indices=qubit_indices)



                elif self._SDK_name.upper() in ['PYQUIL', 'PYQUIL-FOR-AZURE-QUANTUM']:
                    if add_measurements:
                        circuit_object = self._add_measurements_pyquil(quantum_program=circuit_object,
                                                                       qubit_indices=qubit_indices,
                                                                       classical_register=classical_register)

                all_circuits.append(circuit_object)

        return all_circuits
