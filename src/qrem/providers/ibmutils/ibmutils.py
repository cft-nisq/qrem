"""qrem.providers.ibmutils module contains helper functions for translating and preparation of circuits for running a characterisation/mitigation/benchmarking experiment
on IBM Quantum machines via QiskitRuntimeService or IBMServiceBackend.
"""

from typing import List, Union, Dict, Optional,Tuple
import collections
import numpy as np
import numpy.typing as npt
import orjson
from tqdm import tqdm
from pathlib import Path
# qiskit imports
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import IBMBackend as  IBMServiceBackend
from qiskit_ibm_provider import IBMBackend

from qrem.common import io, printer, convert
from qrem.qtypes import CircuitCollection, ExperimentResults
# qrem imports
from qrem.common.constants import EXPERIMENT_TYPE_SYMBLOS
import numpy as np
from typing import Dict, Tuple

def add_measurements(
        
    qiskit_circuit:QuantumCircuit,
    q_reg:QuantumRegister,
    c_reg:ClassicalRegister,
    valid_qubit_indices:List[int]):
    """
    Helper function that adds a measurement for every valid qubit defined in valid_qubit_indices list on existing qiskit_circuit (QuantumCircuit).
    """  
    # - we need to measure for every available qubit (number of qubits = classical_register_size = len(valid_qubit_indices)),
    # from a valid qubit with an index taken from a sorted list of valid_qubit_indices
    # - we only need as many qubits as max defined value of valid indices in our circuit definition (q_reg.size)

    for qubit_index in range(c_reg.size):
        which_qubit = q_reg[valid_qubit_indices[qubit_index]]
        which_classical_output = c_reg[qubit_index]
        qiskit_circuit.measure(which_qubit, which_classical_output)  

def circuits_translator_DDOT( circuit:np.ndarray, 
                                qiskit_circuit:QuantumCircuit,
                                valid_qubit_indices:list) -> QuantumCircuit:
    """
    Helper method that creates DDOT circuits in format native to QISKIT and applies to existing (empty) qiskit_circuit,
    based on the passed circuits and valid_qubit_indices, specifying what gates should be applied to what qubits.
    """

    
    for qubit_index in range(circuit.size):

        qubit_id = valid_qubit_indices[qubit_index]
        if circuit[qubit_index] == 0 :
            qiskit_circuit.id(qubit_id)
        elif circuit[qubit_index] == 1:
            qiskit_circuit.x(qubit_id)
        else:
            raise ValueError('Wrong circuit label: ', circuit)

    
    return qiskit_circuit

def circuits_translator_QDOT( circuit:np.ndarray, 
                                qiskit_circuit:QuantumCircuit,
                                valid_qubit_indices:list) -> QuantumCircuit:
    """
    Helper method that creates QDOT circuits in format native to QISKIT and applies to existing (empty) qiskit_circuit,
    based on the passed circuits and valid_qubit_indices, specifying what gates should be applied to what qubits.
    """
    # Opisac to lepiej i dodac ze to konkretnie dla Pauliego
        
    
    for qubit_index in range(circuit.size):
        qubit_id = valid_qubit_indices[qubit_index]

        if circuit[qubit_index] == 0:
            qiskit_circuit.id(qubit_id)
        elif circuit[qubit_index] == 1:
            qiskit_circuit.x(qubit_id)
        elif circuit[qubit_index] == 2:
            qiskit_circuit.h(qubit_id)
        elif circuit[qubit_index] == 3:
            qiskit_circuit.x(qubit_id)
            qiskit_circuit.h(qubit_id)
        elif circuit[qubit_index] == 4:
            qiskit_circuit.h(qubit_id)
            qiskit_circuit.s(qubit_id)
        elif circuit[qubit_index] == 5:
            qiskit_circuit.x(qubit_id)
            qiskit_circuit.h(qubit_id)
            qiskit_circuit.s(qubit_id)
        else:
            raise ValueError(
                f"Wrong circuit label: '{circuit[qubit_index]} from circuit: '{circuit}'")

        return qiskit_circuit




def transpile_circuits(qiskit_circuits:List[QuantumCircuit],
                       backend):
    """
    Helper method that transpiles circuits for the device. Optimization methodd set now to 0
    """    
    #TODO PP discuss optimization level here, should it be 0 or default 3
    return transpile(qiskit_circuits, 
                     backend = backend, 
                     optimization_level=0,
                     scheduling_method="asap")  



def download_results(backend: Union[IBMServiceBackend, IBMBackend], 
                    job_IDs:List[str],
                    backup: Union[bool,str,Path]=False,
                    from_jobs_backup: Union[bool,str,Path]=False,
                    overwrite:bool = False):    
    """
    Executes circuits prepared in QISKIT format (qiskit_circuits) on the backend provided by a qrem.providers.ibm.connection(...) function.

    Parameters
    ----------
    provider: IBMProvider
    job_IDs:List[str]

    backup: Union[bool,str,Path]=False
        Should original results be backed up somewhere (mostly for debugging)
    
    overwrite:bool
        Should original results overwrite any existing file

    Returns
    -------
    job_id_list: List[str]
        Lsit of submitted job IDs
    """         
    job_list = []
    all_jobs_circuits = list()
    all_jobs_results = []
    backup_dict={}
    
    #[1] Get results and circuits for every job ID from IBM machine
    if not from_jobs_backup:
        for job_ID in job_IDs:
            job = backend.retrieve_job(job_ID)

            if not job.done():
                print(f"ERROR IN Job execution: {job.rerror_message}")
            job_list.append(job)
            #print(f"{job_ID} - created: {job.creation_date()} finished: {job.time_per_step()['finished']} ")
            
        if not job_list:
            raise ValueError(f"No job files downloaded from the chosen backend")
        sorted_job_list = sorted(job_list, key=lambda x: x.time_per_step()['finished'])
        #[print(f"{job.job_id()} - created: {job.creation_date()} finished: {job.time_per_step()['finished']} ") for job in sorted_job_list ]


        # Save if backup path is given
        if backup:
            for job in sorted_job_list:
                dictres = {}
                dictres["results"] = job.result()
                dictres["circuits"] = job.circuits()
                dictres["id"] = job.job_id()
                dictres["time"] = job.time_per_step()['finished']
                io.save(dictionary_to_save=dictres,directory=backup,custom_filename='ibmjob___'+job.job_id(), overwrite = overwrite)

    else:
        #path = Path("/mnt/bkpT7/work_piotr/test_ibm")
        if isinstance(from_jobs_backup,str):
            from_jobs_backup = Path(from_jobs_backup)
        if isinstance(from_jobs_backup,Path):
            
            for file in from_jobs_backup.glob("*.pkl"):
                if "ibmjob" in str(file.stem):
                    job_list.append(io.load(file_path=str(file)))
            if not job_list:
                raise ValueError(f"No job files found in {from_jobs_backup}")
            sorted_job_list = sorted(job_list, key=lambda x: x["time"])
    #-----------------------
    #-----------------------

    #[2] Distillate results and circuits from jobs
    for job in sorted_job_list:
        #[1.1] retrive_circuits:
        if not from_jobs_backup:
            circuits= job.circuits()
        else:
            circuits= job["circuits"]
        job_circuits=[]
        for circuit in circuits:
            splitstr = circuit.name.split("___")
            label = splitstr[-2] #we set structure of  this label in translate_circuits_to_qiskit_format()
            id = splitstr[-1] #we set this label in translate_circuits_to_qiskit_format()
            job_circuits.append(label)
        all_jobs_circuits = all_jobs_circuits + job_circuits
        
        
        #[1.3] retrieve counts:
        if not from_jobs_backup:
            result = job.result()
        else:
            result= job["results"]
        job_results = result.get_counts()
        all_jobs_results = all_jobs_results + job_results



    return all_jobs_results, all_jobs_circuits




def compare_circuits(circuit_labels :List[str],
                    original_circuits_path: Union[str,Path,bool] = False,
                    original_circuits: Union[CircuitCollection,bool]= False):
    
    original_circuits_list = []
    original_circuit_labels = []
    circuits_collection = None

    #[1] check path for circutits object. if provided - read in
    if original_circuits_path and Path(original_circuits_path).exists():
        original_circuits_fromfile = CircuitCollection()
        original_circuits_fromfile.import_json(json_import_path=original_circuits_path)

        original_circuits_list = original_circuits_fromfile.circuits
        circuits_collection = original_circuits_fromfile

    #[2] if  original_circuits are provided, and path is provided, only the object will be used - version from file will be discarded, (if there is a difference, warning will appear)
    if original_circuits and isinstance(original_circuits,CircuitCollection):
        if original_circuits_path and not collections.Counter(original_circuits.circuits) == collections.Counter(original_circuits_list):
            printer.warprint("WARNING: Collection in the file different than provided collection. Using provided collection")

        original_circuits_list = original_circuits.circuits
        circuits_collection = original_circuits        


    # [3] if original_circuits_list is empty will not iterate
    if not circuits_collection:
        printer.warprint("WARNING: No original circuits provided. Skipping comparison")
        return None
    if not circuit_labels and circuits_collection:
        printer.warprint("WARNING: No circuit labels provided. Skipping comparison")
        return circuits_collection
    
    for idx, numpy_label in enumerate(original_circuits_list):
        # add and compare circuit labels one by one
        original_circuit_labels.append(convert.ndarray_to_bitstring(numpy_label))

        if circuit_labels[idx] != original_circuit_labels[-1]:
            printer.warprint(f"WARNING: Different circuit downloaded from IBM Provider than in original collection at index {idx}: \n{circuit_labels[idx]} \nvs \n{original_circuit_labels[-1]}")
            if original_circuit_labels[-1] in circuit_labels:
                printer.warprint(f"WARNING: New index for {original_circuit_labels[-1] } is {circuit_labels.index(original_circuit_labels[-1])}")
            else:
                printer.warprint(f"WARNING: label {original_circuit_labels[-1] } missing from the set downloaded from IBM Provider")


    if not collections.Counter(circuit_labels) == collections.Counter(original_circuit_labels):
        printer.warprint(f"ERROR: Discrepancy between original labels and labels provided from results. ")
        circuits_collection.circuits_labels = circuit_labels

    # if circuit labels missing from original file - add them
    if not circuits_collection.circuits_labels:
        circuits_collection.circuits_labels = circuit_labels        

    else:
        printer.qprint(f"INFO: All circuits downloaded from IBM Provider match original circuits and were processed successfully.")

    
    return circuits_collection


def ibm_to_qrem_restults(qrem_circuits:CircuitCollection, 
                        original_results: List[Dict[str, int]],
                        save_path: Union[bool,str,Path]=False,
                        overwrite:bool = False,
                        format:str="pkl") -> Dict[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]]:
    """
    Loads results from IBM backend into ExperimentResults object based on CircuitCollection
    """
    
    counts = {}
    for circuit, result in zip(qrem_circuits.circuits_labels, original_results):
        keys = list(result.keys())
        values = list(result.values())   

        # Convert keys to boolean matrix; IBM returns qubit results in little-endian format, qrem works with Big-Endian format of bitstrings
        bool_matrix = np.array([list(map(int, key[::-1])) for key in keys], dtype=bool)

        # Convert values to 1D vector
        int_vector = np.array(values, dtype=int)

        counts[circuit] = (bool_matrix, int_vector)

    experiment_results = ExperimentResults(source=qrem_circuits)
    experiment_results.counts = counts
      
    if save_path:
        json_export_path = io.prepare_outfile(outpath = save_path, overwrite = overwrite , default_filename= 'experiment_results.json')
        if format == "pkl":
            experiment_results.export_pickle(pickle_export_path = json_export_path, overwrite = overwrite)
        else:
            experiment_results.export_json(json_export_path = json_export_path, overwrite = overwrite)
        
    return experiment_results
