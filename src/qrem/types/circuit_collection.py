"""qrem.types.circuit_collection module contains CircuitCollection class, which serves as the base class that contains all information
about set of circuits to be sent to quantum machine for characterisation/mitigation/benchmarking purposes.
It inherits from DataStructureBase, and is extended by ExperimentResults - class that contains results of experiments performed on a given circuit collection.
""" 

from datetime import datetime
from qrem.types.datastructure_base import DataStructureBase
from qrem.common.config import QremConfigLoader
from qrem.common.constants import CIRCUIT_DATA_TYPE as c_type

import numpy as np
import numpy.typing as npt

# TODO sort out repeating circuits already in CircuitCollection

class CircuitCollection(DataStructureBase):
    """
    The class contains data about circuits to be prepared on a certain device, for a specified experiment.
    Parameters
    ----------
    device (str): name of the device on which the circuits will be run
    experiment_type (str): DDOT or QDOT (Diagonal/Quantum Detector Overlapping Tomography)
    circuits_list (list[str]): list of circuits, where each circuit is a string of symbols (0-1 for DDOT, 0-5 for QDOT)
    qubit_indices (list[int]): list of sorted indices of qubits on which the circuits are to be executed
    gate_error_threshold (float): gate error on qubit above which qubits are disregarded (we don't execute circuits there)
    locality (int): what degree marginals we wish to characterize
    no_shots (int): how many times each circuit is executed
    datetime_created_utc: when the circuit collection was created
    author (str): creator of collection
    """
    def __init__(self, device):
        # [X] default value definitions
        super().__init__()
        self.device = device
        self.experiment_type = ''
        
        self.circuits_labels = []           # it is good to keep circuits still as a list of strings for results analysis
        self.circuits = []                  # target format: np.ndarray((number_of_qubits,number_of_circuits),dtype=uint8)
        #self.circuits_dictionary = []

        self.qubit_indices = []             #needs to be sorted - valid qubits indices
        self.random_circuits_count = None #to be filled in compute_number_of_circuits()
        self.total_circuits_count = None    #to be filled in compute_number_of_circuits(); dividing it by random_circuits_count will give how many times a circuit needs to be run

        self.gate_error_threshold = None  # 0.01
        self.locality = None  # 2
        self.no_shots = None  # 10000
        self.datetime_created_utc = datetime.utcnow()
        self.experiment_name = ''
        self.author = ''
        self.metadata = []

    def load_config(self, config: QremConfigLoader):
        self.experiment_name = config.experiment_name
        self.experiment_type = config.experiment_type
        self.locality = config.k_locality
        self.gate_error_threshold = config.gate_threshold
        self.author = config.author
        self.no_shots = config.shots_per_circuit

    def load_from_dict(self, dictionary):
        for key in dictionary:
            if key in self.get_dict_format():
                setattr(self, key,
                        dictionary[key])  # TODO: secure against bad data, e.g mismatched qubit and circuit len?
                
                if key =="circuits":
                    print(type(dictionary[key]) is list)
                    if type(dictionary[key]) is list:
                        self.circuits = np.array(dictionary[key],dtype=c_type)
                    elif type(dictionary[key]) is  np.ndarray:
                        if(type(dictionary[key][0][0])!=c_type):
                            print("WARNING, Type of ndarray element should be uint8, converting")
                            self.circuits = self.circuits.astype(c_type)
                        pass
                    else:
                        print("WARNIGN, Werid format of circuits saved in json, may fail on next step")
                        self.circuits = np.ndarray(dictionary[key],dtype=c_type)
        pass



# tests:
if __name__ == "__main__":
    test_collection = CircuitCollection('test_name')
    print(test_collection.get_dict_format())
    dictionary_to_load = {'experiment_type': 'qdot',
                          'circuits': np.array([[1, 2, 3], [4, 5, 6]], dtype=c_type),
                          'qubit_indices': [0, 2, 5],
                          'gate_error_threshold': 0.005,
                          'no_shots': 1,
                          'datetime_created_utc': datetime.utcnow(),
                          'author': 'tester',
                          'metadata': {"note1": 'some string note',}}
    test_collection.load_from_dict(dictionary_to_load)
    print(test_collection.circuits.shape)
    print('after loading dict:\n', test_collection.get_dict_format())
    json_dict_test = test_collection.to_json()
    test_collection.export_json('test_exported_json.json', overwrite=True)
    test_collection.import_json('test_exported_json.json')
    print('after reading from file:\n', test_collection.get_dict_format())
    # pickle_dict_test = test_collection.get_pickle()
    # test_collection.export_pickle('exported_pickle', overwrite=True)
    # test_collection.metadata["note2"]= 'changed note'
    # print('after change:\n', test_collection.get_dict_format())
    # test_collection.import_pickle('exported_pickle')
