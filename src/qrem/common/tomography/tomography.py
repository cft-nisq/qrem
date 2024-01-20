from typing import Dict, Tuple, List, Union, Optional, Type
import numpy as np
import time
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from qrem.common import io, probability, printer,convert, math as qmath
import qrem.common.povmtools as povmtools
from qrem.common.printer import qprint 
from qrem.common.math import Constants as Const
from qrem.common.math import ketbra
import numpy as np 
from qrem.qtypes.qunatum_detector_tomography_data import QDTCalibrationSetup
from qrem.common.external import hyperplane_projections
import copy
import scipy as sc 
import qutip


class QDTFitter:
    """
        This class is meant to resemble qiskit's state tomography and process tomography fitters and to calculate the
        maximum likelihood povm estimator describing a detector basing
        on QDT job results and used probe states.
    """

    def __init__(self, algorithm_convergence_threshold=1e-6):
        self.algorithmConvergenceThreshold = algorithm_convergence_threshold


    @staticmethod
    def __get_r_operator(m_m: np.ndarray, index_of_povm_effect: int, frequencies_array: np.ndarray,
                         probe_states: List[np.ndarray]) -> np.ndarray:
        """
        Description:
            This method_name calculates R operator as defined in Ref. [1].
        Parameters:
            :param m_m: Effect for which R operator is calculated.
            :param index_of_povm_effect: Index of povm effect for which R is calculated.
            :param frequencies_array: frequencies_array - array with size (m x number_of_qubits), where m means number of probe states,
            number_of_qubits means number of POSSIBLE outcomes.
            :param probe_states: arrray_to_print list of probe states density matrices.
        Returns:
            The R operator as described in Ref. [1].
        """

        number_of_probe_states = frequencies_array.shape[0]


        d = probe_states[0].shape[0]

        m_r = np.zeros((d, d), dtype=complex)

        for probe_state_index in range(number_of_probe_states):
            expectation_value_on_probe_state = np.trace(m_m @ probe_states[probe_state_index])

            if expectation_value_on_probe_state == 0j:
                continue

            x = frequencies_array[probe_state_index, index_of_povm_effect] / expectation_value_on_probe_state
            m_r += x * probe_states[probe_state_index]

        if np.linalg.norm(m_r) == 0:
            m_r = np.zeros((d, d), dtype=complex)

        return m_r

    @staticmethod
    def __get_lagrange_matrix(r_matrices: List[np.ndarray], povms: List[np.ndarray]) -> np.ndarray:
        """
        Description:
            Calculates Lagrange matrix used in Lagrange multipliers optimization method_name.
        Parameters:
            :param r_matrices: arrray_to_print list of R matrices described in a method_name generating them.
            :param povms: arrray_to_print list of effects for which Lagrange matrix will be calculated.
        Returns:
           Lagrange matrix for given parameters.
        """
        number_of_outcomes = len(povms)
        dimension = povms[0].shape[0]
        second_power_of_lagrange_matrix = np.zeros((dimension, dimension), dtype=complex)

        for j in range(number_of_outcomes):
            second_power_of_lagrange_matrix += r_matrices[j] @ povms[j] @ r_matrices[j]

        lagrange_matrix = sc.linalg.sqrtm(second_power_of_lagrange_matrix)

        return lagrange_matrix

    @staticmethod
    def __calculate_symmetric_m(m_lagrange_matrix: np.ndarray, m_r: np.ndarray, m_m: np.ndarray) -> np.ndarray:
        """
        Description:
            arrray_to_print method_name used for calculating symmetric m matrix.
        Parameters:
            :param m_m: arrray_to_print matrix of which symmetric method_name will be calculated.
            :param m_r: Previously calculated R operator.
            :param m_lagrange_matrix:
        Returns:
            Symmetric m matrix.
        """
        try:
            # Try to perform inversion of lagrange matrix.
            m_inversed_lagrange_matrix = np.linalg.inv(m_lagrange_matrix)
        except np.linalg.LinAlgError:
            # In some special cases it may fail. Provide identity matrix in that scenario.
            m_inversed_lagrange_matrix = np.eye(np.shape(m_lagrange_matrix)[0])

        symmetric_m = m_inversed_lagrange_matrix @ m_r @ m_m @ m_r @ m_inversed_lagrange_matrix

        return symmetric_m




    def _get_maximum_likelihood_povm_estimator(self,
                                               calibration_setup: QDTCalibrationSetup) -> List[np.ndarray]:
        """
        Description:
            Given results of Quantum Detector Tomography experiments and list of probe states, return the Maximum
            Likelihood estimation of POVM describing a detector. Uses recursive method_name from [1]. See also [2].
        Parameters:
            :param calibration_setup: QDTCalibrationSetup object that consists data upon which maximum likelihood POVM
            estimator should be calculated.
        Returns
            Maximum likelihood estimator of POVM describing a detector.
        """

        number_of_probe_states = calibration_setup.frequencies_array.shape[0]
        number_of_outcomes = calibration_setup.frequencies_array.shape[1]

        dimension = calibration_setup.probe_states[0].shape[0]

         

        # for pi in calibration_setup.probe_states:
        #
        #     qprint_array(pi)
        #
        # raise KeyboardInterrupt
        initial_povm = calibration_setup.initial_ml_povm_guess

        if initial_povm is None:

            povm = []


            for j in range(number_of_outcomes):
                povm.append(np.identity(dimension) / number_of_outcomes)
        else:
            povm = copy.deepcopy(initial_povm)

        # print(np.round(sum(povm),3))
        # # print(calibration_setup.frequencies_array.shape)
        # raise KeyError
        # Threshold is dynamic, thus another variable
        threshold = self.algorithmConvergenceThreshold

        i = 0
        current_difference = 1

        t0 = time.time()

        try:
            while abs(current_difference) >= threshold:
                i += 1

                if i % 50 == 0:
                    last_step_povm = copy.copy(povm)

                r_matrices = [self.__get_r_operator(povm[j], j,
                                                    calibration_setup.frequencies_array,
                                                    calibration_setup.probe_states)
                              for j in range(number_of_outcomes)]
                lagrange_matrix = self.__get_lagrange_matrix(r_matrices, povm)
                povm = [self.__calculate_symmetric_m(lagrange_matrix, r_matrices[j], povm[j])
                        for j in range(number_of_outcomes)]

                if i % 100 == 0:  # calculate the convergence tests only sometimes to make the code faster
                    current_difference = sum([np.linalg.norm(povm[k] - last_step_povm[k], ord=2)
                                              for k in range(number_of_outcomes)])




                elif (time.time()-t0)/60 > 30:  # make sure it does not take too long, sometimes convergence might not be so good
                    threshold = 1e-3

                elif (time.time()-t0)/60 > 10:  # make sure it does not take too long, sometimes convergence might not be so good
                    threshold = 1e-4

                if i%5000==0 and (time.time()-t0)>60:
                    print('iteration:',i,', with current difference: ',current_difference)

                    # for Mi in povm:
                    # qprint_array(povm[0])

        except(KeyboardInterrupt):
            pass


        return povm




    def _get_least_squares_povm_estimator_pauli(self,
                                                calibration_setup: QDTCalibrationSetup):


        number_of_probe_states = calibration_setup.frequencies_array.shape[0]
        number_of_outcomes = calibration_setup.frequencies_array.shape[1]
        dimension = calibration_setup.probe_states[0].shape[0]

        number_of_qubits = int(np.log2(dimension))

        norm = 1/3**number_of_qubits

        #First index - probe ket
        #Second index - outcome label
        frequencies_array = calibration_setup.frequencies_array
        probe_states = calibration_setup.probe_states


        states_labels = calibration_setup.states_labels

        povm_choi = np.zeros((dimension**2,dimension**2),dtype=complex)




        povm = [np.zeros((dimension,dimension),dtype=complex)
                for _ in range(number_of_outcomes)]
        #
        #  
        # for pi in probe_states:
        #     qprint_array(pi)
        # raise KeyboardInterrupt("hejka")
        # print(number_of_outcomes,number_of_probe_states)
        for outcome_index in range(number_of_outcomes):
            for state_index in range(len(states_labels)):
                povm[outcome_index]+=probe_states[state_index]*frequencies_array[state_index,
                                                                                 outcome_index]

        povm = [Mi.T*norm for Mi in povm]

        return povm



    def __get_least_sqaures_povm_choi_estimator_pauli(self,
                                         calibration_setup):

        povm = self._get_least_squares_povm_estimator_pauli(calibration_setup=calibration_setup)
        return povmtools.get_choi_from_POVM(povm)

    def _get_projected_least_squares_povm_estimator(self,
                                                   calibration_setup):

        number_of_outcomes = calibration_setup.frequencies_array.shape[1]
        dimension = calibration_setup.probe_states[0].shape[0]

        number_of_qubits = int(np.log2(dimension))


        povm_LS = self._get_least_squares_povm_estimator_pauli(calibration_setup=calibration_setup)



        #JT a test is performed wheter POVM is a physical POVM, if this is true it is returned

        if povmtools.check_if_povm(povm=povm_LS):
            return povm_LS

        #JT: if this is not the case projection is performed

        else:

            #JT: here a Choi state of the corresponding POVM is performed

            choi_LS = povmtools.get_choi_from_POVM(POVM=povm_LS)

            #JT: Here Qutip is used, an object corresponding to POVM joi state is constructed

            test_choi = qutip.Qobj(choi_LS,superrep='choi',type='super')


            #JT: Now a test of complete positivity of the Choi state is performed

            if test_choi.iscp:

                #JT: If the state is completly positive no projection onto CP states is performed
                projected_choi_CP1 = choi_LS
            else:

                #JT: if the corresponding Choi state is not CP a projection is performed

                projected_choi_CP1 = hyperplane_projections.new_proj_CP_threshold(rho=choi_LS,
                                                           free_trace=False,
                                                           thres_least_ev=True,
                                                           full_output=False
                                                           )
            #JT: The second projection (onto trace preserving - TP maps) is performed
            #JT: Again qutip is used

            projected_choi_CP_test = qutip.Qobj(projected_choi_CP1,superrep='choi',type='super')

            #JT: If the state is CPTP no projection is performed

            if projected_choi_CP_test.iscptp:
                projected_choi_CPTP = projected_choi_CP1

            #JT: If state is CP but not TP the final mixing with a CPTP state is performed

            else:
                projected_choi_CPTP = hyperplane_projections.final_CPTP_by_mixing(hyperplane_projections.proj_TP(projected_choi_CP1))



            #JT: When a proper Choi state is obtained, the resulting povm is constructed out of this Choi state

            povm = povmtools.get_POVM_from_choi(povm_choi=projected_choi_CPTP,
                                                number_of_outcomes=number_of_outcomes,
                                                dimension = int(2**number_of_qubits))
        return povm


    def get_POVM_estimator(self,
                           calibration_setup,
                           method='pls'):

        if method.lower() in ['maximal_likelihood', 'ml']:
            return self._get_maximum_likelihood_povm_estimator(calibration_setup=calibration_setup)
        elif method.lower() in ['least_squares', 'ls']:
            return self._get_least_squares_povm_estimator_pauli(calibration_setup=calibration_setup)
        elif method.lower() in ['projected_least_squares', 'pls']:
            return self._get_projected_least_squares_povm_estimator(calibration_setup=calibration_setup)




    # def get_least_squared_povm_estimator(self):
        

def get_r_operator(m_m: np.ndarray, index_of_povm_effect: int, frequencies_array: np.ndarray,
                        probe_states: List[np.ndarray]) -> np.ndarray:
    """
    Description:
        This method_name calculates R operator as defined in Ref. [1].
    Parameters:
        :param m_m: Effect for which R operator is calculated.
        :param index_of_povm_effect: Index of povm effect for which R is calculated.
        :param frequencies_array: frequencies_array - array with size (m x number_of_qubits), where m means number of probe states,
        number_of_qubits means number of POSSIBLE outcomes.
        :param probe_states: arrray_to_print list of probe states density matrices.
    Returns:
        The R operator as described in Ref. [1].
    """

    number_of_probe_states = frequencies_array.shape[0]


    d = probe_states[0].shape[0]

    m_r = np.zeros((d, d), dtype=complex)

    for probe_state_index in range(number_of_probe_states):
        expectation_value_on_probe_state = np.trace(m_m @ probe_states[probe_state_index])

        if expectation_value_on_probe_state == 0j:
            continue

        x = frequencies_array[probe_state_index, index_of_povm_effect] / expectation_value_on_probe_state
        m_r += x * probe_states[probe_state_index]

    if np.linalg.norm(m_r) == 0:
        m_r = np.zeros((d, d), dtype=complex)

    return m_r

def get_lagrange_matrix(r_matrices: List[np.ndarray], povms: List[np.ndarray]) -> np.ndarray:
    """
    Description:
        Calculates Lagrange matrix used in Lagrange multipliers optimization method_name.
    Parameters:
        :param r_matrices: arrray_to_print list of R matrices described in a method_name generating them.
        :param povms: arrray_to_print list of effects for which Lagrange matrix will be calculated.
    Returns:
        Lagrange matrix for given parameters.
    """
    number_of_outcomes = len(povms)
    dimension = povms[0].shape[0]
    second_power_of_lagrange_matrix = np.zeros((dimension, dimension), dtype=complex)

    for j in range(number_of_outcomes):
        second_power_of_lagrange_matrix += r_matrices[j] @ povms[j] @ r_matrices[j]

    lagrange_matrix = sc.linalg.sqrtm(second_power_of_lagrange_matrix)

    return lagrange_matrix


def calculate_symmetric_m(m_lagrange_matrix: np.ndarray, m_r: np.ndarray, m_m: np.ndarray) -> np.ndarray:
    """
    Description:
        arrray_to_print method_name used for calculating symmetric m matrix.
    Parameters:
        :param m_m: arrray_to_print matrix of which symmetric method_name will be calculated.
        :param m_r: Previously calculated R operator.
        :param m_lagrange_matrix:
    Returns:
        Symmetric m matrix.
    """
    try:
        # Try to perform inversion of lagrange matrix.
        m_inversed_lagrange_matrix = np.linalg.inv(m_lagrange_matrix)
    except np.linalg.LinAlgError:
        # In some special cases it may fail. Provide identity matrix in that scenario.
        m_inversed_lagrange_matrix = np.eye(np.shape(m_lagrange_matrix)[0])

    symmetric_m = m_inversed_lagrange_matrix @ m_r @ m_m @ m_r @ m_inversed_lagrange_matrix

    return symmetric_m




def get_maximum_likelihood_povm_estimator(calibration_setup: QDTCalibrationSetup) -> List[np.ndarray]:
    """
    Description:
        Given results of Quantum Detector Tomography experiments and list of probe states, return the Maximum
        Likelihood estimation of POVM describing a detector. Uses recursive method_name from [1]. See also [2].
    Parameters:
        :param calibration_setup: QDTCalibrationSetup object that consists data upon which maximum likelihood POVM
        estimator should be calculated.
    Returns
        Maximum likelihood estimator of POVM describing a detector.
    """

    number_of_probe_states = calibration_setup.frequencies_array.shape[0]
    number_of_outcomes = calibration_setup.frequencies_array.shape[1]

    dimension = calibration_setup.probe_states[0].shape[0]

        

    # for pi in calibration_setup.probe_states:
    #
    #     qprint_array(pi)
    #
    # raise KeyboardInterrupt
    initial_povm = calibration_setup.initial_ml_povm_guess

    if initial_povm is None:

        povm = []


        for j in range(number_of_outcomes):
            povm.append(np.identity(dimension) / number_of_outcomes)
    else:
        povm = copy.deepcopy(initial_povm)


    # Threshold is dynamic, thus another variable
    threshold = algorithmConvergenceThreshold

    i = 0
    current_difference = 1

    t0 = time.time()

    try:
        while abs(current_difference) >= threshold:
            i += 1

            if i % 50 == 0:
                last_step_povm = copy.copy(povm)

            r_matrices = [get_r_operator(povm[j], j,
                                                calibration_setup.frequencies_array,
                                                calibration_setup.probe_states)
                            for j in range(number_of_outcomes)]
            lagrange_matrix = get_lagrange_matrix(r_matrices, povm)
            povm = [calculate_symmetric_m(lagrange_matrix, r_matrices[j], povm[j])
                    for j in range(number_of_outcomes)]

            if i % 100 == 0:  # calculate the convergence tests only sometimes to make the code faster
                current_difference = sum([np.linalg.norm(povm[k] - last_step_povm[k], ord=2)
                                            for k in range(number_of_outcomes)])




            elif (time.time()-t0)/60 > 30:  # make sure it does not take too long, sometimes convergence might not be so good
                threshold = 1e-3

            elif (time.time()-t0)/60 > 10:  # make sure it does not take too long, sometimes convergence might not be so good
                threshold = 1e-4

            if i%5000==0 and (time.time()-t0)>60:
                print('iteration:',i,', with current difference: ',current_difference)

                # for Mi in povm:
                # qprint_array(povm[0])

    except(KeyboardInterrupt):
        pass


    return povm




def get_least_squares_povm_estimator_pauli(calibration_setup: QDTCalibrationSetup):


    number_of_probe_states = calibration_setup.frequencies_array.shape[0]
    number_of_outcomes = calibration_setup.frequencies_array.shape[1]
    dimension = calibration_setup.probe_states[0].shape[0]

    number_of_qubits = int(np.log2(dimension))

    norm = 1/3**number_of_qubits

    #First index - probe ket
    #Second index - outcome label
    frequencies_array = calibration_setup.frequencies_array
    probe_states = calibration_setup.probe_states


    states_labels = calibration_setup.states_labels

    povm_choi = np.zeros((dimension**2,dimension**2),dtype=complex)




    povm = [np.zeros((dimension,dimension),dtype=complex)
            for _ in range(number_of_outcomes)]
    #
    #  
    # for pi in probe_states:
    #     qprint_array(pi)
    # raise KeyboardInterrupt("hejka")
    # print(number_of_outcomes,number_of_probe_states)
    for outcome_index in range(number_of_outcomes):
        for state_index in range(len(states_labels)):
            povm[outcome_index]+=probe_states[state_index]*frequencies_array[state_index,
                                                                                outcome_index]

    povm = [Mi.T*norm for Mi in povm]

    return povm



def get_least_sqaures_povm_choi_estimator_pauli(calibration_setup):

    povm = get_least_squares_povm_estimator_pauli(calibration_setup=calibration_setup)
    return povmtools.get_choi_from_POVM(povm)

def get_projected_least_squares_povm_estimator(calibration_setup):

    number_of_outcomes = calibration_setup.frequencies_array.shape[1]
    dimension = calibration_setup.probe_states[0].shape[0]

    number_of_qubits = int(np.log2(dimension))


    povm_LS = get_least_squares_povm_estimator_pauli(calibration_setup=calibration_setup)



    #JT a test is performed wheter POVM is a physical POVM, if this is true it is returned

    if povmtools.check_if_povm(povm=povm_LS):
        return povm_LS

    #JT: if this is not the case projection is performed

    else:

        #JT: here a Choi state of the corresponding POVM is performed

        choi_LS = povmtools.get_choi_from_POVM(POVM=povm_LS)

        #JT: Here Qutip is used, an object corresponding to POVM joi state is constructed

        #test_choi = qutip.Qobj(choi_LS,superrep='choi',type='super')

        #test_choi = copy.copy(choi_LS)

        #JT: Now a test of complete positivity of the Choi state is performed

        if qmath.check_complete_positivity_and_trace_preservation(density_operator=choi_LS,dim1=dimension,dim2=dimension): 

            #JT: If the state is completly positive no projection onto CP states is performed
            projected_choi_CP1 = choi_LS
        else:

            #JT: if the corresponding Choi state is not CP a projection is performed

            projected_choi_CP1 = hyperplane_projections.new_proj_CP_threshold(rho=choi_LS,
                                                        free_trace=False,
                                                        thres_least_ev=True,
                                                        full_output=False
                                                        )
        #JT: The second projection (onto trace preserving - TP maps) is performed
        #JT: Again qutip is used

        #projected_choi_CP_test = qutip.Qobj(projected_choi_CP1,superrep='choi',type='super')

        #JT: If the state is CPTP no projection is performed
        test_choi = qutip.Qobj(choi_LS,superrep='choi',type='super')


            #JT: Now a test of complete positivity of the Choi state is performed

        if test_choi.iscp:

            #JT: If the state is completly positive no projection onto CP states is performed
            projected_choi_CP1 = choi_LS

        if qmath.check_complete_positivity_and_trace_preservation(density_operator=projected_choi_CP1,dim1=dimension,dim2=dimension): 
            projected_choi_CPTP = projected_choi_CP1

        #JT: If state is CP but not TP the final mixing with a CPTP state is performed

        else:
            projected_choi_CPTP = hyperplane_projections.final_CPTP_by_mixing(hyperplane_projections.proj_TP(projected_choi_CP1))



        #JT: When a proper Choi state is obtained, the resulting povm is constructed out of this Choi state

        povm = povmtools.get_POVM_from_choi(povm_choi=projected_choi_CPTP,
                                            number_of_outcomes=number_of_outcomes,
                                            dimension = int(2**number_of_qubits))
    return povm


def get_POVM_estimator(calibration_setup,
                        method='pls'):

    if method.lower() in ['maximal_likelihood', 'ml']:
        return get_maximum_likelihood_povm_estimator(calibration_setup=calibration_setup)
    elif method.lower() in ['least_squares', 'ls']:
        return get_least_squares_povm_estimator_pauli(calibration_setup=calibration_setup)
    elif method.lower() in ['projected_least_squares', 'pls']:
        return get_projected_least_squares_povm_estimator(calibration_setup=calibration_setup)




# def get_least_squared_povm_estimator(self):

#JT: labels are used to interpret abstract symbols used to create tomographic circuits in terms of physical states used in tomography
#By Pauli an overcomplete bais of 6 eigenstates of Pauli eigenstates is ment
def labels_interpreter_paulis(label_gate):
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








#JT: a function used to create a multiqubit state out of a string of integers, uses interpreter. It is used in POVMs computations via ML reconstruction
def get_tensored_ket(self,
                        key_multiple):
    ket_now = 1
    for key_state in key_multiple:
        ket_now = np.kron(ket_now,self._labels_interpreter(key_state))
    return ket_now

#JT: a function used to create a multiqubit density matrix out of a string of integers, uses interpreter. It is used in POVMs computations via PLS reconstruction
#the terms 3 * projector onto a state and - identity matrix stem from formula for PLS reconstruction

def get_tensored_ket_LS(key_multiple,labels_interpreter ='PAULI'):
    
    if labels_interpreter == "PAULI": 
        ket_now = 1
        for key_state in key_multiple:
            ket_now = np.kron(ket_now,
                                3*ketbra(labels_interpreter_paulis(key_state))-np.eye(2,
                                                                            dtype=complex))
    else:
        qprint("AS FOR NOW THIS INTRPRETER IS NOT SUPPORTED, PLEASE ADD IT TO THE CODE")
    
    return ket_now


####USED FOR OLD (I.E EARLIER THAN MARCH 2021 RIGETTI EXPERIMENTS)
def labels_interpreter_paulis_rigetti_debug(label_gate):
    _pauli_labels_rigetti_debug = ['z+', 'z-', 'x+', 'y+', 'x-', 'y-']

    if label_gate not in ['%s' % i for i in range(6)] + list(range(6)):
        raise ValueError('Wrong label: ' + label_gate + ' for Pauli eigentket.')

    mapping = {i: _pauli_labels_rigetti_debug[i] for i in range(6)}
    mapping = {**mapping, **{'%s' % i: _pauli_labels_rigetti_debug[i] for i in range(6)}}
    for i in range(6):
        mapping[i] = _pauli_labels_rigetti_debug[i]

    return Const.pauli_eigenkets()[mapping[label_gate]].reshape(2,1)