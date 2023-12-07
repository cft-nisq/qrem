"""
AWS-Braket Provider Module
==========================

This module (`qrem.providers.aws_braket`) provides functionalities for running characterisation, 
mitigation, and benchmarking experiments on AWS Braket devices. It includes methods for retrieving 
device properties, translating circuit collections to Braket format, preparing circuits for execution, 
and executing these circuits on AWS Braket.

The module is designed to interact with AWS Braket services, utilizing their device and job handling 
capabilities.

Functions
---------
add_ground_states
    Function to add ground states from a file. [Functionality not implemented in provided code snippet.]
    Parameters:
    - groundstates_file (str, optional): Path to the file containing ground state information.

get_device
    Retrieves properties of a specified AWS Braket device.
    Parameters:
    - device_full_name (str, optional): The full ARN of the AWS Braket device. Defaults to Rigetti's Aspen-M-2.
    - verbose_log (bool, optional): If True, enables verbose logging of device properties.

get_valid_qubits_properties
    Extracts properties of valid qubits from an AWS Braket device.
    Parameters:
    - device (AwsDevice): The AWS Braket device from which to retrieve qubit properties.
    - threshold (Optional[float], optional): Error threshold for qualifying a qubit as valid. Defaults to 0 (no threshold).
    - verbose_log (bool, optional): Enables verbose logging if set to True. Defaults to False.
    - simultaneous_randomize_benchmark (bool, optional): If True, uses simultaneous randomize benchmark for error thresholding. Defaults to True.

translate_circuits_to_braket_format
    Translates a collection of quantum circuits into the format used by AWS Braket.
    Parameters:
    - circuit_collection (CircuitCollection): Collection of circuits to be translated.
    - valid_qubit_indices (List[int]): Indices of valid qubits to be used in the translation.

prepare_circuits
    Prepares and stores AWS Braket circuits and associated metadata in a specified directory.
    Parameters:
    - braket_circuits (List[Circuit]): List of quantum circuits in AWS Braket format.
    - circuit_collection (CircuitCollection): The original collection of circuits before translation.
    - good_qubits_indices: Indices of qubits deemed good for computation.
    - number_of_repetitions (int, optional): The number of repetitions for each circuit execution. Defaults to 1024.
    - experiment_name (str, optional): Name of the experiment. Defaults to an empty string.
    - metadata (dict, optional): Additional metadata related to the experiment. Defaults to an empty dictionary.
    - job_tags (dict, optional): Tags associated with the job. Defaults to {'QREM_JOB'}.
    - job_dir (str, optional): Directory to store prepared circuits and metadata. Defaults to an empty string.
    - pickle_submission (bool, optional): If True, circuits will be pickled for submission. Defaults to False.
    - number_of_task_retries (int, optional): Number of retries for each task. Defaults to 3.
    - overwrite_output (bool, optional): If True, overwrites existing data in the job directory. Defaults to False.
    - verbose_log (bool, optional): Enables verbose logging if set to True. Defaults to False.

execute_circuits
    Executes prepared quantum circuits on an AWS Braket device.
    Parameters:
    - device_name (str, optional): The full ARN of the AWS Braket device to use for execution. Defaults to Rigetti's Aspen-M-2.
    - pickle_submission (bool, optional): If True, uses pickled circuits for submission. Defaults to False.
    - job_dir (Union[Path, str], optional): Directory containing the circuits and metadata for the job. Defaults to 'job_dir'.
    - verbose_log (bool, optional): If True, enables verbose logging of the execution process. Defaults to True.
"""

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
from qrem.qtypes import CircuitCollection, ExperimentResults
from braket.aws import AwsQuantumJob, AwsQuantumTask

from qrem.common import convert
from qrem.providers.ibmutils import ibmutils 
from qrem.providers.awsutils import awsutils 
#TODO add  function add_ground_states
def add_ground_states(groundstates_file = ""):
    """
    [Placeholder for future implementation]
    Function to add ground states from a file.

    Parameters
    ----------
    groundstates_file : str, optional
        Path to the file containing ground state information.

    Notes
    -----
    This function is currently not implemented in the provided code snippet.
    """    
    pass

#-----------------------
# PART 1: Get backend properties (most importantly - good qubits)
#-----------------------
def get_device(device_full_name = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2",
               verbose_log = False):
    """
    Retrieves various properties of the specified AWS Braket device.

    Parameters
    ----------
    device_full_name : str, optional
        The full ARN of the AWS Braket device. Defaults to Rigetti's Aspen-M-2.
    verbose_log : bool, optional
        If True, enables verbose logging of device properties.

    Returns
    -------
    Tuple[AwsDevice, dict]
        Returns a tuple of the AwsDevice object and a dictionary containing metadata about the device.
    """

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
    """
    Analyzes and reports properties of valid qubits from an AWS Braket device, 
    considering a given error threshold and benchmarking method.

    Parameters
    ----------
    device : AwsDevice
        The AWS Braket device from which to retrieve qubit properties.
    threshold : float, optional
        Error threshold for qualifying a qubit as valid. Defaults to 0 (no threshold).
    verbose_log : bool, optional
        Enables verbose logging if set to True. Defaults to False.
    simultaneous_randomize_benchmark : bool, optional
        If True, uses simultaneous randomize benchmark for error thresholding. Defaults to True.

    Returns
    -------
    dict
        A dictionary containing indices of good qubits and other relevant information.
    """

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
    Internal helper method that translates a circuit representation into a specific
    eigenstate format native to BRAKET. It applies Pauli eigenstates to the circuit.

    (in deprecated codebase formerly named _apply_pauli_eigenstate).

    Parameters
    ----------
    eigenstate_index : int
        Index representing the eigenstate to be applied.
    quantum_circuit : Circuit
        The Braket quantum circuit to which the eigenstate is applied.
    qubit_index : int
        Index of the qubit in the quantum circuit.

    Returns
    -------
    Circuit
        The modified Braket quantum circuit with the applied eigenstate.

    Raises
    ------
    ValueError
        If an incorrect eigenstate index is provided.

    Notes
    -----
    This function is intended for internal use.
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
    """
    Translates a collection of quantum circuits into the format used by AWS Braket.

    Parameters
    ----------
    circuit_collection : CircuitCollection
        Collection of circuits to be translated.
    valid_qubit_indices : List[int]
        Indices of valid qubits to be used in the translation.

    Returns
    -------
    List[Circuit]
        A list of circuits in the AWS Braket format.
    """

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
                     pickle_submission = True,
                     number_of_task_retries:int = 3,
                     overwrite_output = False,
                     verbose_log:bool = False):
    """
    Prepares and stores AWS Braket circuits and associated metadata in a specified directory.

    Parameters
    ----------
    braket_circuits : List[Circuit]
        List of quantum circuits in AWS Braket format.
    circuit_collection : CircuitCollection
        The original collection of circuits before translation.
    good_qubits_indices : list
        Indices of qubits deemed good for computation.
    number_of_repetitions : int, optional
        The number of repetitions for each circuit execution. Defaults to 1024.
    experiment_name : str, optional
        Name of the experiment. Defaults to an empty string.
    metadata : dict, optional
        Additional metadata related to the experiment. Defaults to an empty dictionary.
    job_tags : dict, optional
        Tags associated with the job. Defaults to {'QREM_JOB'}.
    job_dir : str, optional
        Directory to store prepared circuits and metadata. Defaults to an empty string.
    pickle_submission : bool, optional
        If True, circuits will be pickled for submission. Defaults to True.
    number_of_task_retries : int, optional
        Number of retries for each task. Defaults to 3.
    overwrite_output : bool, optional
        If True, overwrites existing data in the job directory. Defaults to False.
    verbose_log : bool, optional
        Enables verbose logging if set to True. Defaults to False.

    Returns
    -------
    bool
        Returns True if preparation is successful, False otherwise.
    """

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
    pickle_submission= True,
    job_dir:Path="job_dir",
    verbose_log = True):
    """
    Packs all data and script runner, and executes prepared quantum circuits on an AWS Braket device.

    Parameters
    ----------
    device_name : str, optional
        The full ARN of the AWS Braket device to use for execution. Defaults to Rigetti's Aspen-M-2: arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-2.
    pickle_submission : bool, optional
        If True, uses pickled circuits for submission. Defaults to True.
    job_dir : Union[Path, str], optional
        Directory containing the circuits and metadata for the job. Defaults to 'job_dir'.
    verbose_log : bool, optional
        If True, enables verbose logging of the execution process. Defaults to True.
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


def retrieve_results(task_ARNs:List[str] =[],
                    original_circuits: Union[CircuitCollection,bool]= False,
                    original_circuits_path: Union[bool,str,Path] = False,
                    save_experiment_results: Union[bool,str,Path]=False,
                    backup_original_task_results: Union[bool,str,Path]=False,
                    from_tasks_backup: Union[bool,str,Path] = False,
                    overwrite = False,
                    verbose_log = False) -> ExperimentResults: 
    """
    Retrieves and processes results from an AWS Braket backend based on provided task ARNs.

    Parameters
    ----------
    provider : IBMProvider
        The IBMProvider instance used to retrieve results.
    task_ARNs : List[str]
        List of task ARNs for which results are to be retrieved.
    original_circuits : Union[CircuitCollection, bool], optional
        The original circuits used in the jobs. Set to False if not provided. Defaults to False.
    original_circuits_path : Union[bool, str, Path], optional
        Path to the file containing the original circuits. Defaults to False.
    config_file_path : Union[bool, str, Path], optional
        Path to the file containing the original submission config. Defaults to False.
    save_experiment_results : Union[bool, str, Path], optional
        Path to save the processed experiment results. Defaults to False.
    backup_original_task_results : Union[bool, str, Path], optional
        Path to backup the original results. Defaults to False.
    from_tasks_backup : Union[bool, str, Path], optional
        Read from pickled backed-up task files. Useful mostly for debugging.
    overwrite : bool, optional
        If True, overwrites existing files at the specified paths. Defaults to False.

    Returns
    -------
    ExperimentResults
        results of an experiment held in ExperimentResults.counts object of format {"<circuit_label>": tuple(<RESTULS_MATRIX>, <COUNTS_VECTOR>)}
    """     
   #[1] Download original results from provider
    original_results, circuit_labels = awsutils.download_results(task_ARNs = task_ARNs,
                                                                 backup =  backup_original_task_results,
                                                                 from_tasks_backup = from_tasks_backup,
                                                                 overwrite = overwrite);
    
    #[2] Retrieve, prepare circuits collection; look for discrepancies
    if original_circuits_path or original_circuits:
        #this will compare circuits and provide original circuits if they are not provided in original_circuits, but path is filled
        #same impoementation as in case of ibm. Possible change of location of compare_circuits function in future
        original_circuits = ibmutils.compare_circuits(circuit_labels, original_circuits_path, original_circuits );

    #[3] Retrieve, prepare circuits collection; look for discrepancies
    if not original_circuits:
        warprint("METADATA MISSING. No original circuits file provided. ExperimentResults will miss metadata. Always try to match experiment with original circuit collection.")
        if not circuit_labels:
            raise ValueError("ERROR: No circuit labels provided and no original circuits file referenced. Cannot create ExperimentResults object.")
        processed_circuits = CircuitCollection()
        processed_circuits.job_IDs = task_ARNs
        processed_circuits.circuits = [ convert.bitstring_to_ndarray(label) for label in circuit_labels ]
        processed_circuits.circuits_labels = circuit_labels

    else:
        processed_circuits = original_circuits
        if processed_circuits.job_IDs:
            warprint("INFO: CricuitCollection already contains job_IDs. Will update them for ExperimentResult object based on the given list of task ARNs")
        processed_circuits.job_IDs = task_ARNs 

    #[4] Prepare results object
    experiment_results = awsutils.aws_to_qrem_restults( qrem_circuits = processed_circuits, 
                                                        original_results = original_results,
                                                        save_path=save_experiment_results,
                                                        overwrite = overwrite)
    
    return experiment_results