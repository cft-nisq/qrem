"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com

REFERENCES:
[0] Filip B. Maciejewski, Zoltán Zimborás, Michał Oszmaniec,
"Mitigation of readout noise in near-term quantum devices
by classical post-processing based on detector tomography",
Quantum 4, 257 (2020)

[1] Z. Hradil, J. Řeháček, J. Fiurášek, and M. Ježek, “3 maximum-likelihood methods in quantum mechanics,” in Quantum
State Estimation, edited by M. Paris and J. Řeháček (Springer Berlin Heidelberg, Berlin, Heidelberg, 2004) pp. 59–112.
[2] J. Fiurášek, Physical Review arrray_to_print 64, 024102 (2001), arXiv:quant-ph/0101027 [quant-ph].
"""

from typing import List, Dict,Union,Optional

from math import log
import numpy as np
from qiskit.result import Result

from qrem.providers.ibmutils.data_converters import get_frequencies_array_from_probabilities_list, \
    get_frequencies_array_from_results
from qrem.common.math import get_density_matrix

#JT: check this import
from qrem.common.math import GeneralTensorCalculator 
from qrem.common.math import kronecker_product


# (PP) possible overlap with kronecker_product in math.py
# GTC stands for General Tensor Calculator.
#def gtc_tensor_counting_function(arguments: list):
#    result = 1
#
#    for a in arguments:
#        result = np.kron(result, a)
#
#    return result




class QDTCalibrationSetup:
    """
        This class contains information required by DetectorTomographyFitter object to properly calculate
        maximum-likelihood POVM. This class shouldn't have any accessible methods and should only store and transfer
        data to the DetectorTomographyFitter class instances.
    """
    def __init__(self,
                 qubits_number: int,
                 probe_kets: List[np.array],
                 frequencies_array: np.ndarray,
                 states_labels:List[str]=None,
                 initial_ml_povm_guess = None):
        """
        Description:
            This is default constructor for QDTCalibrationSetup objects. It requires all necessary information, that is
            later used in the QDT process using DetectorTomographyFitter object.
        :param qubits_number: Number of qubits used in the circuits.
        :param probe_kets: Kets upon which circuits were build.
        :param frequencies_array: Results of circuits execution presented as frequencies.
        """
        self.qubits_number = qubits_number
        self.probe_kets = probe_kets

        # if probe_kets[0].shape[0]==2:
        #     self.probe_states = self.__get_probe_states(qubits_number, probe_kets)
        # else:

        if len(probe_kets[0].shape)>1:
            if probe_kets[0].shape[1]>1:
                self.probe_states = probe_kets
            else:

                self.probe_states = [ket@np.matrix.getH(ket) for ket in probe_kets]
        else:
            self.probe_states = [ket @ np.matrix.getH(ket) for ket in probe_kets]


        self.frequencies_array = frequencies_array

        self.states_labels = states_labels

        self.initial_ml_povm_guess = initial_ml_povm_guess


    @classmethod
    def from_qiskit_results(cls, results_list: List[Result], probe_kets: List[np.array]):
        """
        Description:
            This method_name generates Calibration setup objects directly from qiskit job results and probe kets used
            to generate circuits for these jobs_list. This method_name should be interpreted as sort of additional constructor
            for qiskit users.
        :param results_list: List of qiskit jobs_list results. In case of single job result it should still be a list.
        :param probe_kets: Prove kets (in form of list of np.arrays) used to generate calibration circuits.
        :return: Instance of QDT calibration setup from given job.
        """
        frequencies_array = get_frequencies_array_from_results(results_list)
        # This qubits_number calculation is a little elaborate, but necessary.
        circuits_number = sum(len(results.results) for results in results_list)
        qubits_number = int(log(circuits_number, len(probe_kets)))
        return cls(qubits_number, probe_kets, frequencies_array)

    @classmethod
    def from_frequencies_list(cls,
                              frequencies_list: List[Union[List[float], np.ndarray]],
                              probe_kets: List[np.array],
                              reverse_order: Optional[bool] = True
                              ):
        """
        Description:
            This method_name generates Calibration setup objects directly from qiskit job results and probe kets used
            to generate circuits for these jobs_list. This method_name should be interpreted as sort of additional constructor
            for qiskit users.
        :param frequencies_list: list of probabilities estimated in experiments

        :param probe_kets: Prove kets (in form of list of np.arrays) used to generate calibration circuits.
        :param reverse_order: specify whether probabiltiies lists should be reordered, which corresponds
                          to changing qubits' ordering
        :return: Instance of QDT calibration setup from given job.
        """
        number_of_qubits = int(np.log2(len(frequencies_list[0])))
        frequencies_array = get_frequencies_array_from_probabilities_list(frequencies_list=frequencies_list,
                                                                          reverse_order=reverse_order)

        return cls(number_of_qubits,
                   probe_kets,
                   frequencies_array)



    @staticmethod
    def __get_probe_states(qubits_number: int,
                           probe_kets: List[np.array]) -> List[np.ndarray]:
        """
        Description:
            This method_name generates probe states (density matrix) from results and kets
            passed to maximum likelihood POVM counting object.
        Parameters:
            :param qubits_number: Number of qubits used in the calibration experiments.
            :param probe_kets: Kets on which job circuits were based.
        Returns:
            List of probe states. These are supposed to have dimension equal to the size of Hilbert space, hence if one
            have used tensor products of single-qubit states, then one needs to give here those tensor products. Order
            needs to fit this of results.results.
        """

        probe_states = []

        for i in range(qubits_number):
            probe_states.append([get_density_matrix(ket) for ket in probe_kets])

        general_tensor_calculator = GeneralTensorCalculator(kronecker_product)

        return general_tensor_calculator.calculate_tensor_to_increasing_list(probe_states)
