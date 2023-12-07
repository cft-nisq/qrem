"""qrem.providers.awsutils module contains helper functions for translating and preparation of circuits for running a characterisation/mitigation/benchmarking experiment
on AWS Braket Quantum machines .
"""

from typing import List, Union, Dict, Optional,Tuple
import numpy as np
import numpy.typing as npt
from pathlib import Path

import orjson
from braket.aws import AwsQuantumJob, AwsQuantumTask
# qiskit imports
from qrem.common import io, printer, convert
from qrem.qtypes import CircuitCollection, ExperimentResults
# qrem imports
from qrem.common.constants import EXPERIMENT_TYPE_SYMBLOS
import numpy as np
from typing import Dict, Tuple

def download_results(task_ARNs:List[str],
                    backup: Union[bool,str,Path]=False,
                    from_tasks_backup: Union[bool,str,Path]=False,
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
    task_list = []
    task_circuits = []
    completion_times = []
    all_task_circuits = list()
    all_task_results = []
    
    missing_circuit_labels = False
    #[1] Get results and circuits for every task ID from AWS machine
    if not from_tasks_backup:
        for task_ARN in task_ARNs:
            task = AwsQuantumTask(task_ARN)
            circuit_label = None

            if not task.state() == "COMPLETED":
                print(f"ERROR IN Job execution")

            if "circuit" in task.metadata()["tags"]:
                circuit_label = task.metadata()["tags"]["circuit"]
            else:
                circuit_label = None
                missing_circuit_labels = True

            task_list.append(task.result())
            task_circuits.append(circuit_label)
            if "endedAt" in task.metadata():
                completion_times.append(task.metadata()["endedAt"])
            
        if not task_list:
            raise ValueError(f"No job files downloaded from the chosen backend")
        
        #Sortin should not be necessary here, however, could be done by  completion_times list
        #sorted_job_list = sorted(job_list, key=lambda x: x.time_per_step()['finished'])
       

        # Save if backup path is given
        if backup:
            for id, task_result in enumerate(task_list):
                dictres = {}
                dictres["result"] = task_result.measurement_counts
                dictres["circuit"] = task_circuits[id]
                dictres["task_ARN"] = task_ARNs[id]
                if len(completion_times) > id:
                    dictres["time"] = completion_times[id]
                io.save(dictionary_to_save=dictres,directory=backup,custom_filename='awstaks___'+task_ARNs[id], overwrite = overwrite)

    else:
        #path = Path("/mnt/bkpT7/work_piotr/test_ibm")
        if isinstance(from_tasks_backup,str):
            from_jobs_backup = Path(from_tasks_backup)
        if isinstance(from_tasks_backup,Path):
            
            for file in from_tasks_backup.glob("*.pkl"):
                if "awstaks" in str(file.stem):
                    task_list.append(io.load(file_path=str(file)))
            if not task_list:
                raise ValueError(f"No job files found in {from_jobs_backup}")
            #sorted_job_list = sorted(job_list, key=lambda x: x["time"])
    #-----------------------
    #-----------------------

    #[2] Distillate results and circuits from jobs
    for id, task_result in enumerate(task_list):
        #[1.1] retrive_circuits:
        if not from_jobs_backup:
            circuit_label= task_circuits[id]
        else:
            circuit_label= task_result["circuit"]

        all_task_circuits = all_task_circuits + circuit_label
        
        #[1.3] retrieve counts:
        if not from_jobs_backup:
            result = task_result.measurement_counts
        else:
            result= task_result["result"]
        all_task_results = all_task_results + result

    if missing_circuit_labels:
        printer.warprint("Some circuits were not labelled, results are missing circuit labels, they will need to come from original list")
        all_task_circuits = None

    return all_task_results, all_task_circuits



#TODO PP - refactor this fucntion with the one in ibmutils (only difference = endianness of bitstrings)
def aws_to_qrem_restults(qrem_circuits:CircuitCollection, 
                        original_results: List[Dict[str, int]],
                        save_path: Union[bool,str,Path]=False,
                        overwrite:bool = False,
                        format:str="pkl") -> Dict[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]]:
    """
    Loads results from AWS backend frotmat into ExperimentResults object based on CircuitCollection
    """
    
    counts = {}
    for circuit, result in zip(qrem_circuits.circuits_labels, original_results):
        keys = list(result.keys())
        values = list(result.values())   

        # Convert keys to boolean matrix; AWS returns qubit results in Big-Endian format, qrem also works with Big-Endian format of bitstrings
        bool_matrix = np.array([list(map(int, key)) for key in keys], dtype=bool)

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
