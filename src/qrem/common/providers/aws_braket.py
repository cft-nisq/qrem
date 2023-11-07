import os, shutil
from datetime import datetime
from typing import List, Union, Optional,Tuple
from pathlib import Path
import pickle

import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit

# qrem imports
from qrem.common.constants import EXPERIMENT_TYPE_SYMBLOS
from qrem.common.printer import qprint, errprint, warprint
from qrem.common.utils import get_full_object_size
from qrem.types import CircuitCollection
from braket.aws import AwsQuantumJob


def add_ground_states(groundstates_file = ""):
    pass

#-----------------------
# PART 1: Get backend properties (most importantly - good qubits)
#-----------------------
def get_device(device_full_name = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2",
               verbose_log = False):
    
    device = AwsDevice(device_full_name)
    qubit_count = device.properties.paradigm.qubitCount
    device_cost = device.properties.service.deviceCost
    execution_windows = device.properties.service.executionWindows
    connectivity_graph = device.properties.paradigm.connectivity
    calibration = device.properties.provider.specs

    if verbose_log:
        print('The maximum number of qubits supported by this device is', qubit_count)
        print('Execution windows', execution_windows)
        print('The price of running tasks on this device:', device_cost)

    metadata = {
        "device_name": device_full_name,
        "qubit_count": qubit_count,
        "execution_windows": execution_windows,
        "connectivity_graph": connectivity_graph,
        "calibration_data": calibration,
        "device_cost": device_cost,
    }
    return device, metadata

def get_valid_qubits_properties(  device: AwsDevice, 
                            threshold: Optional[float] = 0,
                            verbose_log = False,
                            simultaneous_randomize_benchmark: bool=True ):

    one_qubit_properties = device.properties.standardized.oneQubitProperties
    quantum_register_size = device.properties.paradigm.qubitCount

    # - quantum register size
    test_quantum_reg_size = len(one_qubit_properties)
    if test_quantum_reg_size != quantum_register_size:
        print(f"ERROR: test_quantum_reg_size <{test_quantum_reg_size}> does not match oneQubitProperties schema size<{quantum_register_size}>. Check braket code for changes.")

    one_qubit_fidelity_dict = {}
    qbits_above_error_treshold = [] # randomized benchmark
    qbits_above_error_treshold_simultaneous = [] #simultanous randomized benchmark
    all_qubits = [int(qubit_key) for qubit_key in one_qubit_properties]
    # [2] get qbits above treshold
    # TODO change to get exactly qubit_key list from one_qubit_properties
    
    if(threshold!= None and threshold > 0):
        # [2.1] for each qubit in backend get properties from one_qubit_properties:
        # https://amazon-braket-schemas-python.readthedocs.io/en/latest/_apidoc/braket.device_schema.standardized_gate_model_qpu_device_properties_v1.html
        for qubit_key in one_qubit_properties:
            #all_qubits.append(qubit_key)
            one_qubit_fidelity_dict[qubit_key] = {}
            one_qubit_fidelity_dict[qubit_key]['rb'] = one_qubit_properties[qubit_key].oneQubitFidelity[0].fidelity
            one_qubit_fidelity_dict[qubit_key]['simultaneous_rb'] = one_qubit_properties[qubit_key].oneQubitFidelity[1].fidelity

            if 1 - one_qubit_fidelity_dict[qubit_key]['rb'] > threshold:
                qbits_above_error_treshold.append(int(qubit_key))
            if 1 - one_qubit_fidelity_dict[qubit_key]['simultaneous_rb'] > threshold:
                qbits_above_error_treshold_simultaneous.append(int(qubit_key))
    else:
        
        qbits_above_error_treshold = []
        qbits_above_error_treshold_simultaneous = []
   
    if simultaneous_randomize_benchmark:
        all_bad_qubits = qbits_above_error_treshold_simultaneous;
    else:
        all_bad_qubits = qbits_above_error_treshold;

    good_qubits_indices =  [q for q in all_qubits if q not in all_bad_qubits]
    # [3.1] Renmove double entries and sort indices to ensure, that results can easily be interpreted. (NOT NECESSARY HERE MOST LIKELY)
    good_qubits_indices = list(set(good_qubits_indices)) 
    # [3.2] Important - indicies should be always sorted
    good_qubits_indices = sorted(good_qubits_indices) 
    number_of_good_qubits = len(good_qubits_indices)

    if(verbose_log):
        print(f"All {number_of_good_qubits} good qubit indices: {good_qubits_indices}")

    return {"good_qubits_indices": good_qubits_indices,
            "number_of_good_qubits": number_of_good_qubits,
            "original_quantum_register_size": quantum_register_size,
            "faulty_qubits": all_bad_qubits
            }


#-----------------------
# PART 2: Translate circuits to IBM / QISKIT FORMAT
#-----------------------
def _circuits_translator(eigenstate_index: int,
                            quantum_circuit: Circuit,
                            qubit_index: int):
    """
    Helper method that creates both DDOT and QDOT circuits in format native to BRAKET and applies to existing (empty) braket_circuit,
     by applying Pauli eigenstates (formerly _apply_pauli_eigenstate).
    """
    # _pauli_labels = ['z+', 'z-', 'x+', 'x-', 'y+', 'y-']

    # Z+
    if eigenstate_index == 0:
        quantum_circuit = quantum_circuit.rz(qubit_index, 0)

    # Z-
    elif eigenstate_index == 1:
        # quantum_circuit = quantum_circuit.rx(qubit_index, np.pi)
        quantum_circuit = quantum_circuit.rz(qubit_index, -0.9199372448290238)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi)
        quantum_circuit = quantum_circuit.rx(qubit_index, - np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, 2.2216554087607694)

    # X+
    elif eigenstate_index == 2:
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)

    # X-
    elif eigenstate_index == 3:
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, -np.pi / 2)

    # Y+
    elif eigenstate_index == 4:
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi)

    # Y-
    elif eigenstate_index == 5:
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)

    else:
        raise ValueError(f"Incorrect eigenstate index: '{eigenstate_index}'!")

    return quantum_circuit

def translate_circuits_to_braket_format(circuit_collection:CircuitCollection,
                                        valid_qubit_indices:list) -> List[Circuit]:
    
    # TODO (PP) sort out repeating circuits already in CircuitCollection
    # prerequisite: no repeating circuits in the accepted collection
    braket_circuits = []
    
    for id in range(0,len(circuit_collection.circuits)):
        #[1] Make a label string from circuit labels - 
        #TODO (PP) won't work for machines with 1 million qubits, also now not used in AWS case
        base_label = np.array2string(circuit_collection.circuits[id], precision=0, separator='', max_line_width = 1000000)[1:-1]
        label = circuit_collection.experiment_name + "___" + base_label + "___" +str(id)

        #[2]  Make an empty circuit
        braket_circuit = Circuit()
        qrem_circuit=circuit_collection.circuits[id]

        for qubit_index in range(qrem_circuit.size):
            qubit_id = valid_qubit_indices[qubit_index]
            eigenstate_index = qrem_circuit[qubit_index]
            braket_circuit = _circuits_translator(eigenstate_index=eigenstate_index, 
                                                  quantum_circuit=braket_circuit,
                                                  qubit_index=qubit_id)

        braket_circuit_verbatim = Circuit().add_verbatim_box(braket_circuit)
        braket_circuits.append(braket_circuit)
    return braket_circuits



#-----------------------
# PART 3: Prepare braket circuits and all metadata to run adn output to job_dir
#-----------------------
def prepare_cricuits(braket_circuits:List[Circuit],
                     circuit_collection:CircuitCollection,
                     good_qubits_indices,
                     number_of_repetitions:int = 1024,
                     experiment_name:str = "",
                     metadata:dict = {},
                     job_tags:dict = {"QREM_JOB",},
                     job_dir:str = "",
                     pickle_submission = False,
                     number_of_task_retries:int = 3,
                     overwrite_output = False,
                     verbose_log:bool = False):
    
    # [0] default naming that always should be the same
    if(pickle_submission):
        circuits_file = 'braket_circuits_list.pkl'
    else:
        circuits_file = None
    qrem_circuits_file = 'qrem_circuits_list.txt'
    good_qubits_indices_file = 'good_qubits_indices.txt'
    metadata_file = 'metadata.pkl'

  
    # [1] Check and prepare output folder
    job_path= Path(job_dir)
    if not job_path.is_dir():
        job_path.mkdir(parents=True,exist_ok=False)
    elif any(job_path.iterdir()) == True and overwrite_output:
        try:
            shutil.rmtree(str(job_path))
        except OSError:
            os.remove(str(job_path))
        job_path.mkdir(parents=True,exist_ok=False)
    elif  any(job_path.iterdir()) == True and not overwrite_output:
        print(f"ERROR: non empty directory would be overwritten {str(job_path)}. Cancelling")
        return False
    else:
        print(f"ERROR: usnpecified. Cancelling")
        return False
    
    
    # [1.1] Check and prepare output fules
    circuits_file_path = job_path.joinpath(circuits_file)
    qrem_circuits_file_path = job_path.joinpath(qrem_circuits_file)
    good_qubits_indices_file_path = job_path.joinpath(good_qubits_indices_file)
    metadata_file_path = job_path.joinpath(metadata_file)

    # [2] write the circuit symbols to pickle file:
    if(pickle_submission):
        with open(circuits_file_path, 'wb') as fp:  # open a text file
            pickle.dump(braket_circuits, fp) # serialize the list

    # write the circuit symbols to txt file:
    with open(qrem_circuits_file_path, 'w') as fp:
        for circuit in circuit_collection.circuits:
            for label in circuit:
                fp.write("%s" % str(label))
            # write each item on a new line
            fp.write("\n")

    # [3] write the used qubits to txt file:
    with open(good_qubits_indices_file_path, 'w') as fp:
        for qubit_index in good_qubits_indices:
            fp.write("%s\n" % qubit_index)
            


    # [4] Prepare and save the metadata to pickle file:
    now = datetime.now()
    metadata["date"] = f"{now.year}-{now.month}-{now.day}"
    metadata["number_of_repetitions"] = number_of_repetitions
    metadata["max_task_retries"] = number_of_task_retries

    # [4.1] Prepare and save tthe tags to metadata:
    job_tags_dict = {tag:tag for tag in job_tags.items()}
    job_tags_dict["experiment"] = experiment_name
    job_tags_dict["date"] = metadata["date"]
    metadata["tags"] = job_tags_dict


    with open(metadata_file_path, 'wb') as fp:  # open a text file
        pickle.dump(metadata, fp) # serialize the metadata
        
    return True

#-----------------------
# PART 4: Execute circuits
#-----------------------
def execute_circuits(
    device_name:str="arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2",
    pickle_submission= False,
    job_dir:Path="job_dir",
    verbose_log = True):
    """
    Packs all data and script runner and executes on AWS machine
    """

    #[0] Define all path based on the job dir
    current_script_dir = Path( __file__ ).parent.absolute()

    if(pickle_submission):
        path_to_circuits = job_dir.joinpath('braket_circuits_list.pkl')
    qrem_path_to_circuits = job_dir.joinpath('qrem_circuits_list.txt')
    path_to_module = current_script_dir.joinpath("awsutils").joinpath("aws_braket_runner.py")
    runner_function = "aws_braket_runner:run"

    abspath_to_circuits = str(path_to_circuits.resolve())
    abspath_to_qrem_circuits = str(qrem_path_to_circuits.resolve())
    abspath_to_module = str(path_to_module.resolve())

    path_to_good_qubits_indices = job_dir.joinpath('good_qubits_indices.txt')
    abspath_to_good_qubits_indices = str(path_to_good_qubits_indices.resolve())

    path_to_metadata = job_dir.joinpath('metadata.pkl')
    abspath_to_metadata = str(path_to_metadata.resolve())


    #[1] Define input data
    input_data={'qrem_circuits_list': abspath_to_qrem_circuits,
                'good_qubits_indices': abspath_to_good_qubits_indices,
                'metadata': abspath_to_metadata}
    
    if(pickle_submission):
        input_data['braket_circuits_list'] = abspath_to_circuits
    
    #[2] Create a job
    job = AwsQuantumJob.create(
        device_name,
        source_module=abspath_to_module,
        input_data=input_data,
        entry_point=runner_function,
        wait_until_complete=False
    )
    if(verbose_log):
        print(job)


