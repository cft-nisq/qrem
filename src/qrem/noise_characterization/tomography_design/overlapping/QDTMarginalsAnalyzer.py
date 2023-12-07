from typing import Optional, Dict, List, Union, Tuple

from tqdm import tqdm
import numpy as np

from qrem.noise_characterization.data_analysis.InitialNoiseAnalyzer import InitialNoiseAnalyzer
 
from qrem.common import povmtools
from qrem.noise_characterization.tomography_design.standard.QuantumDetectorTomography import QDTCalibrationSetup, QDTFitter

from qrem.common.math import Constants as Const
from qrem.common.math import ketbra
"""
JT: This class is used to:
    - perform computations of marginals
    - compute reduced POVMs


Analysis    

"""


#FBM: this should probably descend from different class?
class QDTMarginalsAnalyzer(InitialNoiseAnalyzer):
    def __init__(self,
                 results_dictionary: Dict[str, Dict[str, int]],
                 experiment_name: str,
                 marginals_dictionary: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                 marginals_averaged: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                 POVM_dictionary: Optional[Dict[str, List[np.ndarray]]] = None,
                 convergence_treshold_QDT: Optional[float] = 10 ** (-6)) -> None:

        super().__init__(results_dictionary=results_dictionary,
                         marginals_dictionary=marginals_dictionary
                         )

        if marginals_averaged is None:
            marginals_averaged = {}
        if POVM_dictionary is None:
            POVM_dictionary = {}

        self._experiment_name = experiment_name

        self._labels_interpreter = None

        self._convergence_treshold_QDT = convergence_treshold_QDT



        self._marginals_averaged = marginals_averaged

    @property
    def POVM_dictionary(self) -> Dict[Tuple[int], List[np.ndarray]]:
        return self._POVM_dictionary

    @property
    def marginals_averaged(self) -> Dict[Tuple[int], Dict[Tuple[int], np.ndarray]]:
        return self._marginals_averaged
#JT: labels are used to interpret abstract symbols used to create tomographic circuits in terms of physical states used in tomography
#By Pauli an overcomplete bais of 6 eigenstates of Pauli eigenstates is ment
    @staticmethod
    def _labels_interpreter_paulis(label_gate):
        #FBM: TESTS
        _pauli_labels = ['z+', 'z-', 'x+', 'x-', 'y+', 'y-']
        # _pauli_labels = ['z+', 'z-', 'x+', 'y+', 'x-', 'y-']

        if label_gate not in ['%s' % i for i in range(6)] + list(range(6)):
            raise ValueError('Wrong label: ' + label_gate + ' for Pauli eigentket.')
        #JT: here a dictionary mapping integers to states labels is performed
        mapping = {i: _pauli_labels[i] for i in range(6)}
        #JT: This line extends the above dictionary so that it maps symbols to states and states to symbols so that we have 0: 'z+' and 'z+':0
        mapping = {**mapping, **{'%s' % i: _pauli_labels[i] for i in range(6)}}

        #JT: This loop in this place is redundant, for now I comment it
        """for i in range(6):
            mapping[i] = _pauli_labels[i]"""
        #The labels correspoding to Pauli eigenstates are interpreted as states, __pauli_eigenkets returns vectors corresponding to normalized eigenstates
        return Const.pauli_eigenkets()[mapping[label_gate]].reshape(2,1)

#JT: The same as above, the only difference is ordering of Pauli labels
    @staticmethod
    def __labels_interpreter_paulis_rigetti_debug(label_gate):
        _pauli_labels_rigetti_debug = ['z+', 'z-', 'x+', 'y+', 'x-', 'y-']

        if label_gate not in ['%s' % i for i in range(6)] + list(range(6)):
            raise ValueError('Wrong label: ' + label_gate + ' for Pauli eigentket.')

        mapping = {i: _pauli_labels_rigetti_debug[i] for i in range(6)}
        mapping = {**mapping, **{'%s' % i: _pauli_labels_rigetti_debug[i] for i in range(6)}}
        for i in range(6):
            mapping[i] = _pauli_labels_rigetti_debug[i]

        return Const.pauli_eigenkets()[mapping[label_gate]].reshape(2,1)

    """
    JT: initialize_labels_interpreter: sets interpreter of abstract symbols used in circutis creaation to the states realized in the experiments
    for PAULI we interpret symbols as 6 eigenstates of the Pauli operators
    For some reason for some older experiments on Rigetti the order of the eigenstates was different that that on IBM
    or AWS implemented by Asia. To take this into account there is an additional bool parameter rigetti_debug, which for
    analysis of Rigetii experimental data form March must be set to True, otherwise it should be false      
    """


    def initialize_labels_interpreter(self,
                                      interpreter,
                                      rigetti_debug=False):
        if isinstance(interpreter, str):
            if interpreter.upper() in ['PAULI', 'PAULIS']:
                #JT: here an interpreter is set, see above
                if not rigetti_debug:
                    interpreter = self._labels_interpreter_paulis
                else:
                    interpreter = self.__labels_interpreter_paulis_rigetti_debug
            else:
                raise ValueError('Basis: ' + interpreter + ' not yet supported. \
                    Please provide custom interpreter.')

        self._labels_interpreter = interpreter




#JT: a function used to create a multiqubit state out of a string of integers, uses interpreter. It is used in POVMs computations via ML reconstruction
    def _get_tensored_ket(self,
                          key_multiple):
        ket_now = 1
        for key_state in key_multiple:
            ket_now = np.kron(ket_now,self._labels_interpreter(key_state))
        return ket_now

#JT: a function used to create a multiqubit density matrix out of a string of integers, uses interpreter. It is used in POVMs computations via PLS reconstruction
#the terms 3 * projector onto a state and - identity matrix stem from formula for PLS reconstruction

    def _get_tensored_ket_LS(self,
                          key_multiple):
        ket_now = 1
        for key_state in key_multiple:
            ket_now = np.kron(ket_now,
                              3*ketbra(self._labels_interpreter(key_state))-np.eye(2,
                                                                           dtype=complex))
        return ket_now


    def _compute_averaged_marginals_subset(self,
                                           subset:Tuple[int]):
        # subset_key = 'q' + 'q'.join([str(s) for s in subset])

        #JT: uses a method from MarginalsAnalyzerBase class, comments to that method can be found there

        self._marginals_averaged[subset] = self.get_averaged_marginal_for_subset(subset)

    #JT: We use this internal method to perform POVMs computation

    def _compute_POVM_from_marginals(self,
                                     marginals_dictionary: Dict[str, np.ndarray],
                                     qubit_indices: Union[str, List[int]], # why do we have two ways of specifying subsets
                                     estimation_method='pls'):
        # print('printing')
        # print(marginals_dictionary)
        # print(qubit_indices)
        # raise KeyboardInterrupt
        number_of_qubits = len(qubit_indices)
        dimension = int(2 ** number_of_qubits)

        #FBM: fix this
        keys_list = sorted(list(marginals_dictionary.keys()))


         
        # keys_list = anf.register_names_qubits(qubit_indices=qubit_indices,
        #                                       )

        # print(marginals_dictionary)
        #First index - probe ket
        #Second index - outcome label
        frequencies_array = np.zeros((len(keys_list), dimension), dtype=complex)

        #FBM: debug
        # frequencies_array = np.full((len(keys_list), dimension), 10**(-6), dtype=complex)

        # print(keys_list)

         

        probe_kets = []

        #JT: A loop over all keys of marginals dictionary

        for index_state in range(len(keys_list)):
            key_now = keys_list[index_state]
            if estimation_method.lower() in ['maximal_likelihood', 'ml']:
                probe_kets.append(self._get_tensored_ket(key_now))

            #JT: for PLS here a PLS estimator is computed

            elif estimation_method.lower() in ['least_squares', 'ls', 'projected_least_squares', 'pls']:
                #TODO FIX PLS FOR SIMULATIONS
                probe_kets.append(self._get_tensored_ket_LS(key_now))



            try:
                frequencies_array[index_state, :] = marginals_dictionary[key_now][:]


            except(IndexError):
                frequencies_array[index_state, :] = marginals_dictionary[key_now][:, 0]

            #JT: This is a different error than that above, solution appers to be the same
            except(ValueError):
                frequencies_array[index_state, :] = marginals_dictionary[key_now][:, 0]
            except(KeyError):
                pass

        # qprint_array(frequencies_array)

        #JT the classes below are used to get physical quantities out of POVM estimators

        setup_QDT = QDTCalibrationSetup(qubits_number=number_of_qubits,
                                        probe_kets=probe_kets,
                                        frequencies_array=frequencies_array,
                                        states_labels=keys_list)

        fitter_QDT = QDTFitter()


        #JT: Here physical POVM is reconstructed

        POVM_now = fitter_QDT.get_POVM_estimator(calibration_setup=setup_QDT,
                                                 method=estimation_method)

        # if isinstance(qubit_indices, list):
        #     qubit_indices = convert.qubit_indices_to_keystring(qubit_indices)

        self._POVM_dictionary[qubit_indices] = POVM_now

    def _compute_subset_POVM(self,
                             subset,
                             estimation_method='pls'):
        # subset_key = 'q' + 'q'.join([str(s) for s in subset])

        #JT: the if below is redundant (at least for as we use it, as the same codition is checked in compute_subsets_POVMs_averaged)

        if subset not in self._marginals_averaged.keys():
            self._compute_averaged_marginals_subset(subset)

        averaged_marginals = self._marginals_averaged[subset]
        self._compute_POVM_from_marginals(averaged_marginals,
                                          qubit_indices=subset,
                                          estimation_method=estimation_method)

#JT This method is used to compute POVMs once marginals are computed
    def compute_subsets_POVMs_averaged(self,
                                       subsets_of_qubits: List[List[int]],
                                       estimation_method:str='pls',
                                       show_progress_bar: Optional[bool] = True) -> None:

        #Here number of subsets is established
        subsets_range = range(len(subsets_of_qubits))

        for subset in tqdm(subsets_of_qubits, disable = not show_progress_bar):
            #JT if subset is missing in averadged marginals dictionary then computation its performed

            if subset not in self._marginals_averaged.keys():
                self._compute_averaged_marginals_subset(subset)

        #JT: onec averadged marginals dictionary is ready computation of POVMs is done

        for subset_index in subsets_range:
            self._compute_subset_POVM(subsets_of_qubits[subset_index],
                                      estimation_method=estimation_method)

#JT: This method is used to get noise matrices from reconstructed POVMs, used e.g. after clustering to get noise matrices for computed clusters
# subsets list - list of subsets for which computation should be performed

    def compute_noise_matrices_from_POVMs(self,
                                       subsets_of_qubits: List[List[int]],
                                       show_progress_bar: Optional[bool] = True) -> None:

        for subset in tqdm(subsets_of_qubits,
                           disable = not show_progress_bar):

            #JT: computation of stochastic noise from a quantum POVM, unig a function from povmtools

            reduced_noise_matrix_now = povmtools.get_stochastic_map_from_povm(povm=self.POVM_dictionary[subset])

            #if subset in self.noise_matrices_dictionary[subset]:

            #JT: if key corresponding to a particular subset is present in noise_matrix_dictionary then internal dictionary is updated - (nested structure of disctionaries)
            # the key is subset and 'averaged'

            if subset in self.noise_matrices_dictionary.keys():
                self.noise_matrices_dictionary[subset]['averaged'] = reduced_noise_matrix_now

            #JT: If there is no entry in the dictionary, it is created with value corresponding to another dictionary

            else:
                self.noise_matrices_dictionary[subset]= {'averaged':reduced_noise_matrix_now}

