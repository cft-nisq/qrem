"""
qrem.qtypes.circuit_collection module contains CircuitCollection class, which serves as the base class that contains all information
about a set of circuits to be sent to a quantum machine for characterization, mitigation, or benchmarking purposes.
It inherits from DataStructureBase and is extended by ExperimentResults, a class that contains results of experiments performed on a given circuit collection.


Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""

from datetime import datetime
from qrem.qtypes.datastructure_base import DataStructureBase
from qrem.common.config import QremConfigLoader
from qrem.common.constants import CIRCUIT_DATA_TYPE as c_type
from typing import List, Union
import numpy as np
import numpy.typing as npt

class CircuitCollection(DataStructureBase):

    """
    CircuitCollection class contains data about circuits to be prepared on a certain device for a specified experiment.

    Parameters
    ----------
    device : str
        Name of the device on which the circuits will be run.
    experiment_type : str
        Type of experiment, DDOT (Diagonal Detector Overlapping Tomography) or QDOT (Quantum Detector Overlapping Tomography).
    circuits_list : list[str]
        List of circuits, where each circuit is a string of symbols (0-1 for DDOT, 0-5 for QDOT).
    qubit_indices : list[int]
        List of sorted indices of qubits on which the circuits are to be executed.
    gate_error_threshold : float
        Gate error on a qubit above which qubits are disregarded (circuits are not executed there).
    locality : int
        The degree of marginals we wish to characterize.
    no_shots : int
        How many times each circuit is executed.
    datetime_created_utc : datetime
        When the circuit collection was created.
    author : str
        Creator of the collection.

    Attributes
    ----------
    device : str
        Name of the device on which the circuits will be run.
    experiment_type : str
        Type of experiment, DDOT (Diagonal Detector Overlapping Tomography) or QDOT (Quantum Detector Overlapping Tomography).
    circuits_labels : list[str]
        List of circuits where each circuit is a string of symbols (0-1 for DDOT, 0-5 for QDOT).
    circuits : numpy.ndarray
        Matrix representing circuits (shape: number_of_qubits x number_of_circuits, dtype: uint8).
    job_IDs : list[str]
        After experiment is executed, it will be filled with relevant job_IDs (ibm) / task_ARNs (aws).
    qubit_indices : list[int]
        List of sorted indices of qubits on which the circuits are to be executed.
    random_circuits_count : int
        Number of random circuits to be generated.
    total_circuits_count : int
        Total number of circuits (random + predefined).
    gate_error_threshold : float
        Gate error on a qubit above which qubits are disregarded (circuits are not executed there).
    locality : int
        The degree of marginals we wish to characterize.
    no_shots : int
        How many times each circuit is executed.
    datetime_created_utc : datetime
        When the circuit collection was created.
    experiment_name : str
        Name of the experiment.
    author : str
        Creator of the collection.
    metadata : list
        Additional metadata information.
    """
    
    
    def __init__(self):
        # [X] default value definitions
        super().__init__()
        self.experiment_name = ''
        self.device = ''
        self.experiment_type = ''
        self.author = ''
        
        self.metadata = []
        
        self.circuits_labels : List(str)= []           # it is good to keep circuits still as a list of strings for results analysis
        self.circuits : List(npt.NDArray) = []                  # target format: np.ndarray((number_of_qubits,number_of_circuits),dtype=uint8)
        self.job_IDs = None

        self.qubit_indices = []             #needs to be sorted - valid qubits indices
        self.random_circuits_count = None #to be filled in compute_number_of_circuits()
        self.total_circuits_count = None    #to be filled in compute_number_of_circuits(); dividing it by random_circuits_count will give how many times a circuit needs to be run

        self.gate_error_threshold = None  # 0.01
        self.locality = None  # 2
        self.no_shots = None  # 10000
        self.datetime_created_utc = datetime.utcnow()

    def load_config(self, config: QremConfigLoader):
        """
        Load configuration settings into the CircuitCollection object.

        Parameters
        ----------
        config : QremConfigLoader
            Configuration settings to load.
        """
        self.experiment_name = config.experiment_name
        self.experiment_type = config.experiment_type
        self.locality = config.k_locality
        self.gate_error_threshold = config.gate_threshold
        self.author = config.author
        self.no_shots = config.shots_per_circuit

    def load_from_dict(self, dictionary):
        """
        Load data from a dictionary into the CircuitCollection object.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing data to load into the object.
        """
        for key in dictionary:
            if key in self.get_dict_format():
                setattr(self, key,
                        dictionary[key])  # TODO LOW: secure against bad data, e.g mismatched qubit and circuit len?
                
                if key =="circuits":
                    if type(dictionary[key]) is list:
                        self.circuits = np.array(dictionary[key],dtype=c_type)
                    elif type(dictionary[key]) is  np.ndarray:
                        if(type(dictionary[key][0][0])!=c_type):
                            print("WARNING, Type of ndarray element should be uint8, converting")
                            self.circuits = self.circuits.astype(c_type)
                        pass
                    else:
                        print("WARNING, Werid format of circuits saved in json, may fail on next step")
                        self.circuits = np.ndarray(dictionary[key],dtype=c_type)
        pass



# tests:
if __name__ == "__main__":
    _test_collection = CircuitCollection()
    print(_test_collection.get_dict_format())
    _dictionary_to_load = {'experiment_type': 'qdot',
                          'circuits': np.array([[1, 2, 3], [4, 5, 6]], dtype=c_type),
                          'qubit_indices': [0, 2, 5],
                          'gate_error_threshold': 0.005,
                          'no_shots': 1,
                          'datetime_created_utc': datetime.utcnow(),
                          'author': 'tester',
                          'metadata': {"note1": 'some string note',}}
    _test_collection.load_from_dict(_dictionary_to_load)
    print(_test_collection.circuits.shape)
    print('after loading dict:\n', _test_collection.get_dict_format())
    _json_dict_test = _test_collection.to_json()
    _test_collection.export_json('test_exported_json.json', overwrite=True)
    _test_collection.import_json('test_exported_json.json')
    print('after reading from file:\n', _test_collection.get_dict_format())
    # pickle_dict_test = test_collection.get_pickle()
    # test_collection.export_pickle('exported_pickle', overwrite=True)
    # test_collection.metadata["note2"]= 'changed note'
    # print('after change:\n', test_collection.get_dict_format())
    # test_collection.import_pickle('exported_pickle')
