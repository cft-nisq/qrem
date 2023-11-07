"""qrem.providers.ibmutils.translate module contains helper functions for translating and preparation of circuits for running a characterisation/mitigation/benchmarking experiment
on IBM Quantum machines via QiskitRuntimeService or IBMServiceBackend.
"""

from typing import List, Union, Optional,Tuple
import numpy as np
from tqdm import tqdm
from pathlib import Path
# qiskit imports
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister



from qrem.types import CircuitCollection
# qrem imports
from qrem.common.constants import EXPERIMENT_TYPE_SYMBLOS

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
    #TODO (PP) (MO) discuss optimization level here, should it be 0 or default 3
    return transpile(qiskit_circuits, 
                     backend = backend, 
                     optimization_level=0,
                     scheduling_method="asap")  



def download_results(provider, 
                    job_IDs:List[str],
                    backup: Union[bool,str,Path]=False,
                    overwrite:bool = False):    
    """
    Executes circuits prepared in QISKIT format (qiskit_circuits) on the backend provided by a qrem.common.providers.ibm.connection(...) function.

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
    
    all_jobs_circuits = []
    all_jobs_results = []
    backup_dict={}

    #[1] Get results and circuits for every job ID
    for job_ID in job_IDs:
        job = provider.backend.retrieve_job(job_ID)
        result = job.result()

       
        #[1.1] retrive_circuits:
        circuits= job.circuits()
        job_circuits=[]
        for circuit in circuits:
            splitstr = circuit.name.split("___")
            label = splitstr[-2] #we set structure of  this label in translate_circuits_to_qiskit_format()
            id = splitstr[-1] #we set this label in translate_circuits_to_qiskit_format()
            job_circuits.append(label)
        all_jobs_circuits.append(job_circuits)
        
        
        #[1.1] retrieve counts:
        job_results = result.get_counts()
        all_jobs_results.append(job_results)

        if backup:
            backup_dict[job_ID] = result.to_dict()
            
    if backup:
        json_export_path = io.prepare_outfile(outpath = backup, overwrite = overwrite , default_filename= 'backup_ubm_original_results.pkl')
        
        json_data = orjson.dumps(backup_dict, default=lambda o: o.__dict__,
                    option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SORT_KEYS)
        
        with open(json_export_path, 'wb') as outfile:
            outfile.write(json_data)

    return all_jobs_results, all_jobs_circuits


def compare_circuits(circuit_labels :List[str],
                    original_circuits_path: Union[str,Path] = False,
                    original_circuits: Union[CircuitCollection,bool]= False):
    pass
