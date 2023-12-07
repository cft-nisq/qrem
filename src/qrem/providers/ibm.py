"""
IBM Provider Module
===================

This module (`qrem.providers.ibm`) provides functionalities essential for running characterisation, 
mitigation, and benchmarking experiments on IBM Quantum machines. It enables interactions 
with IBM Quantum services through QiskitRuntimeService and/or IBMServiceBackend. Key functionalities 
include methods for connecting to IBM quantum backends, querying backend properties, translating 
and executing quantum circuits in Qiskit format, and retrieving experiment results.

Functions
---------
connect : 
    Establishes a connection to an IBM Quantum backend.
get_valid_qubits_properties : 
    Retrieves properties of valid qubits from the backend.
translate_circuits_to_qiskit_format : 
    Translates circuit collections to Qiskit QuantumCircuit objects.
execute_circuits : 
    Executes a list of Qiskit QuantumCircuits on an IBM Quantum backend.
retrieve_results : 
    Retrieves and processes experiment results from the IBM Quantum backend.

Internal Helper Functions
-------------------------
_connect_via_runtime : 
    Connects to IBM Quantum services using QiskitRuntimeService.
_connect_via_provider : 
    Connects to IBM Quantum services using IBMProvider.
"""
import os
import math
import re
from pathlib import Path

from time import sleep
from typing import List, Union, Optional,Tuple
import numpy as np
from tqdm import tqdm
# qiskit imports
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.compiler import assemble
from qiskit_ibm_runtime import IBMBackend as  IBMServiceBackend
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_provider import IBMProvider, IBMBackend
from qrem.common.config import QremConfigLoader

from qrem.providers.ibmutils import ibmutils 

# qrem imports
from qrem.common.constants import EXPERIMENT_TYPE_SYMBLOS
from qrem.common.printer import qprint, errprint, warprint
from qrem.common import convert, io
from qrem.common.utils import get_full_object_size
from qrem.qtypes import CircuitCollection
from qrem.qtypes import ExperimentResults


QISKIT_IBM_TOKEN = os.getenv('QISKIT_IBM_TOKEN')
QISKIT_IBM_CHANNEL = os.getenv('QISKIT_IBM_CHANNEL')


#Hardcoded limit for job  input size in megabytes
JOB_PAYLOD_SIZE_LIMIT = 15 #Mb


#-----------------------
# PART 0: Connect to the IBM Quantum specific backend using your token 
#-----------------------
#TODO (PP) in theory you can check which backend is the best and take the best one from the backends for a fiven processor list. Would be nice to implement
# service.least_busy(operational=True, min_num_qubits=5)
# service.backends(simulator=False, operational=True, min_num_qubits=5)
#which connection method we will use

def connect(name:str = "", 
            channel: Optional[str] = QISKIT_IBM_CHANNEL, 
            method:str = "RUNTIME", #available are "RUNTIME", "PROVIDER", "RUNTIME SESSIONS", "JOB EXECUTE"
            instance: Optional[str] = 'ibm-q/open/main', 
            verbose_log = False) -> Tuple[Union[IBMServiceBackend,IBMBackend],Optional[QiskitRuntimeService],Optional[IBMProvider]]:
    """
    Initializes connection to the IBM Quantum backend based on the API_TOKEN defined in the environment variables (.env file).
    Depending on the <method> parameter will start either QiskitRuntimeService object or IBMProvider object

    Parameters
    ----------
    name : str
        Name of the backend to connect to.
    channel: str, default='ibm_quantum' 
        Channel used for the connection (e.g., 'ibm_quantum' or 'ibm_cloud'). Defaults to `QISKIT_IBM_CHANNEL`.
    method: str, default='RUNTIME' 
        Method of connection, choices are 'RUNTIME', 'PROVIDER', 'RUNTIME SESSIONS', 'JOB EXECUTE'. Defaults to 'RUNTIME'.
    instance: str, default='ibm-q/open/main' 
        sSpecific instance for the connection, e.g., 'ibm-q/open/main'. Defaults to 'ibm-q/open/main'.
    verbose_log: bool, default=False
        If True, enables verbose logging. Defaults to False.
        
    Returns
    -------
    Tuple[Union[IBMServiceBackend, IBMBackend], Optional[QiskitRuntimeService], Optional[IBMProvider]]
        A tuple containing:
        - the connected backend object returned by QiskitRuntimeService (IBMServiceBackend) or IBMProvider (IBMBackend),
        - Connected Instance of QiskitRuntimeService (if applicable),
        - Connected Instance of IBMProvider (if applicable).

    """
    
    #[1] Connect and save provider/service objects
    service = None
    provider = None
    if "RUNTIME" in method:
        service = _connect_via_runtime(backend_name= name, channel = channel, instance = instance, verbose_log = verbose_log)
        backend = service.get_backend(name)
    elif "DEBUG" in method:
        service = None
        backend = "DEBUG"
        
    else:
        provider = _connect_via_provider(backend_name= name, channel = channel, instance = instance, verbose_log = verbose_log)
        backend = provider.get_backend(name)    

    if backend != "DEBUG":
        #[2] Check status of the backened
        status = backend.status()
        is_operational = status.operational
        is_internal = status.status_msg
        jobs_in_queue = status.pending_jobs
        max_circuits = backend.max_circuits
    else:
        is_operational = "debug"
        is_internal = "mode"
        jobs_in_queue = 0
        max_circuits = 127

    if verbose_log:
        
        qprint("----------")
        qprint(f"The backend {name} status:", f"{is_operational} : {is_internal}")
        qprint(f"- jobs in queue:", f"{jobs_in_queue}")
        qprint(f"- official circuits limit:", f"{max_circuits}")
              

    #use this first to connect to a backend
    if not is_operational or is_internal == "internal":
        errprint(f"ERROR: The backend {name} is currently not available. ", f"Confirm on https://quantum-computing.ibm.com/services/resources ")
    
    return backend, service, provider

def _connect_via_runtime(backend_name:str = "", channel: Optional[str] = 'ibm_quantum', instance: Optional[str] = 'ibm-q/open/main', verbose_log = False):
    """
    Helper function for qrem.providers.ibm.connect(...) to connect via QiskitRuntimeService
    """
    if instance == '':
        service = QiskitRuntimeService(channel = channel, token=QISKIT_IBM_TOKEN)
    else:
        service = QiskitRuntimeService(channel = channel, token=QISKIT_IBM_TOKEN, instance = instance)

    return service
  
def _connect_via_provider(backend_name:str = "", channel: Optional[str] = 'ibm_quantum', instance: Optional[str] = 'ibm-q/open/main', verbose_log = False):
    """
    Helper function for qrem.providers.ibm.connect(...) to connect via IBMProvider
    """
    if instance == '':
        provider = IBMProvider(token=QISKIT_IBM_TOKEN)
    else:
        provider = IBMProvider(token=QISKIT_IBM_TOKEN, instance = instance)

    return provider


#-----------------------
# PART 1: Get backend properties (most importantly - good qubits)
#-----------------------
def get_valid_qubits_properties(  backend: Union[IBMBackend,IBMServiceBackend], 
                            threshold: Optional[float] = 0,
                            verbose_log = False ):
    """
    Queries the backend for properties of valid qubits, identifying qubits that are operational and below a given error threshold (not flagged faulty by IBM and below given sx gate error treshold ).

    Parameters
    ----------
    backend : Union[IBMBackend, IBMServiceBackend]
        The IBM backend instance from which to retrieve qubit properties returned during connection by QiskitRuntimeService (IBMServiceBackend) or IBMProvider (IBMBackend)
    threshold : float, optional
        Gate error threshold for qualifying a qubit as valid. Defaults to 0 (no threshold). If 0, will not discard qubits based on gate error tresholding
    verbose_log : bool, optional
        Enables verbose logging if set to True. Defaults to False.

    Returns
    -------
    
    Returns
    -------
    dict
        Dictionary describing valid qubits for experiment. 
        {"good_qubits_indices": list of valid qubit indices,
        "number_of_good_qubits": count of valid qubits,
        "original_quantum_register_size": orginal number of available qubits,
        "faulty_qubits": list of qubits indices marked faulty by IBM at the time of connection,
        "non_operational_qubits": list of qubit indices above sx gate error treshold,
        "readout_errors": list of returned readout errors on each of the qubit}
    
    """


    if backend == "DEBUG":
        #  DEBUG MODE - backend is null
        qprint("Running in a DEBUG mode (backend is Null)")
        return {"good_qubits_indices": list(range(0,127)),
                "number_of_good_qubits": 127,
                "original_quantum_register_size": 127,
                
                "faulty_qubits": [],
                "operational_qubits": list(range(0,127)),
                "readout_errors": list (0.001*np.random.sample(127))}
    elif backend == None:
        errprint("Backend is null - something went wrong with connection.")

    #TODO (PP) test on a fake propetries object

    properties = backend.properties()
    # - quantum register size
    quantum_register_size = backend.configuration().n_qubits

    # [1] get faulty qubits -> Important for some quantum machines like Osprey
    faulty_qubits = [q for q in range(quantum_register_size) if q in properties.faulty_qubits()]

    # [2] get qbits above treshold
    if(threshold!= None and threshold > 0):
    # [2.1] for each qubit in backend get single qubit gate errors (of gate sqrt(X), which is treated as a model gate):
        gate_errors_list = []
        for qubit in range(quantum_register_size):
            gate_errors_list.append(properties.gate_error('sx', qubit))

        qbits_above_error_treshold = [idx for idx, val in enumerate(gate_errors_list) if val > threshold]
    else:
        qbits_above_error_treshold = []

    # SHOULD WE ALSO CHECK FOR OPERATIONAL QUBITS?  FOR NOW SKIPPED
    operational_qubits = [q for q in range(quantum_register_size) if properties.is_qubit_operational(q)]
    
    # ANYTHING WE SHOULD DO ABOUT READOUT ERRORS? FOR NOW SKIPPED
    readout_errors = [properties.readout_error(q) for q in range(quantum_register_size)]

    # [3] get correct qubits
    all_bad_qubits = faulty_qubits + qbits_above_error_treshold #we don't need to make a correct set union here, but it would be more proper
    good_qubits_indices =  [q for q in range(quantum_register_size) if q not in all_bad_qubits]
    # [3.1] Renmove double entries and sort indices to ensure, that results can easily be interpreted. (NOT NECESSARY HERE MOST LIKELY)
    good_qubits_indices = list(set(good_qubits_indices)) 

    # [3.2] Important - indicies should be always sorted
    good_qubits_indices = sorted(good_qubits_indices) 

    number_of_good_qubits = len(good_qubits_indices)

    return {"good_qubits_indices": good_qubits_indices,
            "number_of_good_qubits": number_of_good_qubits,
            "original_quantum_register_size": quantum_register_size,
            
            "faulty_qubits": faulty_qubits,
            "operational_qubits": operational_qubits,
            "readout_errors": readout_errors}
    pass



#-----------------------
# PART 2: Translate circuits to IBM / QISKIT FORMAT
#-----------------------
def translate_circuits_to_qiskit_format(circuit_collection:CircuitCollection) -> List[QuantumCircuit]:
    """
    Translates circuits defined as ndarray of unit8s with QREM internal circuit labelling (labels defined in qrem.common.constants.EXPERIMENT_TYPE_SYMBLOS)
    into a list of Qiskit Circuits (not yet transpiled)

    Parameters
    ----------
    circuit_collection: CircuitCollection
        A collection of circuits to be translated, structured as a CircuitCollection object.
        
    Returns
    -------
    List[QuantumCircuit]
        A list of Qiskit QuantumCircuits, with barrier between each of the qubit (read more on qiskit docs about QuantumCircuits)
    """     
    

    # prerequisite: no repeating circuits in the accepted collection
    qiskit_circuits = []

    quantum_register_size = max(circuit_collection.qubit_indices) + 1
    classical_register_size = len(circuit_collection.qubit_indices)

    for id in range(0,len(circuit_collection.circuits)):
        #[1] Make a label string from circuit labels
        #TODO (PP) won't work for machines with 1 million qubits
        base_label = np.array2string(circuit_collection.circuits[id], precision=0, separator='', max_line_width = 1000000)[1:-1]
        label = circuit_collection.experiment_name + "___" + base_label + "___" +str(id)


        #[2]  Make an empty circuit
        q_reg = QuantumRegister(quantum_register_size)
        c_reg = ClassicalRegister(classical_register_size)
        qiskit_circuit = QuantumCircuit(q_reg, c_reg, name=label)
    
        #[3]  Fill in according to experyment type:
        if(circuit_collection.experiment_type.lower() == "ddot"):
            ibmutils.circuits_translator_DDOT(  circuit=circuit_collection.circuits[id], 
                                        qiskit_circuit = qiskit_circuit,
                                        valid_qubit_indices=circuit_collection.qubit_indices) 
        elif(circuit_collection.experiment_type.lower() == "qdot"):
            ibmutils.ccircuits_translator_QDOT(  circuit=circuit_collection.circuits[id],  
                                        qiskit_circuit = qiskit_circuit,
                                        valid_qubit_indices=circuit_collection.qubit_indices) 
        
        else:
            errprint(f"ERROR: not supported experiment type:",f"{circuit_collection.experiment_type}")
        #[4] Barrier between all qubits (no optimizartion allowed)
        qiskit_circuit.barrier()

        ibmutils.add_measurements( qiskit_circuit = qiskit_circuit,
                            q_reg=q_reg,
                            c_reg=c_reg,
                            valid_qubit_indices=circuit_collection.qubit_indices)


        #[5] Append to list of all circuits
        qiskit_circuits.append(qiskit_circuit)

    return qiskit_circuits


#-----------------------
# PART 3: Run circuits
#-----------------------
def execute_circuits(qiskit_circuits:List[QuantumCircuit],
                     method:str = "RUNTIME",
                     job_tags:List[str] = ["QREM_JOB",],
                     instance:str = 'ibm-q/open/main',
                     service:QiskitRuntimeService = None,
                     backend:Union[IBMServiceBackend,IBMBackend] = None,
                     number_of_repetitions:int = 1024,
                     log_level:str ='WARNING',
                     verbose_log:bool = False):
    """
    Executes a list of  QuantumCircuits (qiskit-formatted) on an IBM Quantum backend using the specified method.

    Parameters
    ----------
    qiskit_circuits: List[QuantumCircuit]
        List of QuantumCircuits, not yet transpiled, to be executed on chosen IBM machine
    method: str, default = "RUNTIME"
        The connection/execution method to use, e.g., 'RUNTIME', 'RUNTIME_SESSIONS', 'JOB_EXECUTE', 'PROVIDER'. Defaults to 'RUNTIME'.
        Determined by the execution of qrem.providers.ibm.connection(...) (should be same as there)
    job_tags: List[str] default=["QREM_JOB"]
        Tags to be applied to each job. Defaults to ['QREM_JOB'].
    instance : str, optional
        The instance to use for execution. Defaults to 'ibm-q/open/main'.
    service: QiskitRuntimeService, default=None
        Connected Instance of QiskitRuntimeService. Can be None if connecting via IBMProvider
    backend: Union[IBMServiceBackend,IBMBackend]
        The backend on which to execute the circuits. Defaults to None.
    number_of_repetitions:int, default=1024
        The number of times each circuit should be repeated (shots). Defaults to 1024.
    log_level: str, default = "WARNING"
        Log level for the execution, e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'. Defaults to 'WARNING'.
        look https://qiskit.org/ecosystem/ibm-runtime/stubs/qiskit_ibm_runtime.RuntimeOptions.html#qiskit_ibm_runtime.RuntimeOptions for more information
    verbose_log : bool, optional
        Enables verbose logging if True. Defaults to False.  

    Returns
    -------
    List[str]
        List of job IDs from the executed jobs.
    """         
    
    
    #TODO (PP) check for null Service, provider and backend type - for DEBUG and other purposes

    # The valid log levels are: DEBUG, INFO, WARNING, ERROR, and CRITICAL. The default level is WARNING.
    # translate to IBM - qiskit Circuits
    if(number_of_repetitions > backend.max_shots):
        warprint(f"WARNING: Number of repetitions ({number_of_repetitions}) is larger than max number of shots ({backend.max_shots}) defined on chosen backend {backend.name}, defaulting to: ", f"({backend.max_shots})")
    # transpile all circuits
    transpiled_circuits = ibmutils.ibm_translate.transpile_circuits(qiskit_circuits,backend)

    #[X] Caculate size of a given batch:
    qobject = assemble (experiments = transpiled_circuits,shots = number_of_repetitions)
    full_payload_size = get_full_object_size(qobject)/1000000.0

    number_of_circuits = len(transpiled_circuits)
    batch_division = math.ceil(full_payload_size/JOB_PAYLOD_SIZE_LIMIT)
    batch_size = min(backend.max_circuits-1, math.floor(number_of_circuits/batch_division))
    if batch_size > backend.max_circuits:
        warprint(f"WARNING: Calculated batch circuit count based on payload size limit ({batch_size}) is larger than machine max circuits count. Defaulting to max circuits count -1:", f"{backend.max_circuits-1}")
    #[X] divide into batches based on the calculated quantities
    batches = [transpiled_circuits[i:i + batch_size] for i in range(0, number_of_circuits, batch_size)] 

    job_id_list = []
    # choose specific method (ibm_provider vs ibm_runtime?)
    if method == "RUNTIME_SESSIONS":
        if verbose_log:
            qprint(f"Now sending  prepared jobs to the IBM backend {backend.name} via Runtime with sessions:", f"{len(qiskit_circuits)} jobs repeated {number_of_repetitions} times")
            qprint(f"- logging level:", log_level)
            qprint(f"- IBM connection instance:", instance)
            qprint(f"- memory:", "True")
            qprint(f"- meas_level:", 2)
            qprint(f"- number of repetitions:", number_of_repetitions)
            qprint(f"- job tags:", job_tags)
            warprint(f"-------------------")

        runtime_options = {
            'backend': backend.name,
            'job_tags': job_tags,
            #'optimization_level': 0, #no circuit optimisation
            'log_level': log_level, 
            'instance': instance
        }

        #divide 
        with Session(service=service, backend=backend) as session:
            job_list = []
            count = 0
            tdqm_batches = tqdm(batches)
            for batch in tdqm_batches:
                # more in https://quantum-computing.ibm.com/services/programs/prototypes?program=circuit-runner
                runtime_inputs = {
                    'circuits': batch,
                    'shots': number_of_repetitions,  # integer
                    'memory': True,  # integer#per-shot measurement bitstrings
                    'meas_level': 2, # The measurement output 2 is the discriminated measurement counts. Level 1 is the IQ measurement kernel values.
                }

                job = service.run(
                    program_id='circuit-runner',
                    options=runtime_options,
                    inputs=runtime_inputs,
                )
                sleep(0.5)
                qprint(f"Sent job ({count}/{len(batches)}) ")
                job_list.append(job)
                job_abbrv = str(job.job_id())
                job_abbrv =  (job_abbrv[6] + '..') if len(job_abbrv) > 8  else job_abbrv
                tdqm_batches.set_description(f"Sent job ({job_abbrv}):")

                # job.result() is blocking, so this job is now finished and the session can be safely closed.
                # Close the session only if all jobs are finished, and you don't need to run more in the session
            sleep(0.5) #not sure if necessarry?
            for job in job_list:
                job_id_list.append(job.job_id())
                # Do not close here, the job might not be completed!
                # result = job.result()
                # UNCERTAIN: shall we already save this result?
            session.close()

    elif method == "RUNTIME":
        if verbose_log:
            qprint(f"Sending  prepared jobs to the IBM backend {backend.name} via Runtime without sessions:", f"{len(qiskit_circuits)} jobs repeated {number_of_repetitions} times")
            qprint(f"- logging level:", log_level)
            qprint(f"- IBM connection instance:", instance)
            qprint(f"- memory:", "True")
            qprint(f"- meas_level:", 2)
            qprint(f"-------------------")
            qprint(f"- number of repetitions:", number_of_repetitions)
            qprint(f"- job tags:", job_tags)

        runtime_options = {
            'backend_name': backend.name,
            'job_tags': job_tags,
            'log_level': log_level, #no mitigation
            'instance': instance
        }
        
        for batch in batches:
            runtime_inputs = {
                'circuits': batch,
                'shots': number_of_repetitions,  # integer
            }
            job = service.run(
                program_id='circuit-runner',
                options=runtime_options,
                inputs=runtime_inputs,
            )
            sleep(0.5) #not sure if necessarry?
            job_id_list.append(job.job_id())

    elif method == "JOB_EXECUTE":
        if verbose_log:
            qprint(f"Sending  prepared jobs to the IBM backend {backend.name} via Qiskit Job.execute:", f"{len(qiskit_circuits)} jobs repeated {number_of_repetitions} times")
            qprint(f"- memory:", "True")
            qprint(f"- optimization_level:", 0)
            qprint(f"-------------------")
            qprint(f"- number of repetitions:", number_of_repetitions)
            qprint(f"- job tags:", job_tags)   

        for batch in batches:
            job = execute(
                            experiments=batch, 
                            backend=backend, 
                            shots=number_of_repetitions, 
                            optimization_level = 0, 
                            memory=True)
            sleep(0.5) #not sure if necessarry?
            job_id_list.append(job.job_id())
            job.update_tags(job_tags)
            sleep(0.1)
        pass

    elif method == "PROVIDER":
        if verbose_log:
            qprint(f"Sending  prepared jobs to the IBM backend {backend.name} via IBMProvider Bakend.run:", f"{len(qiskit_circuits)} jobs repeated {number_of_repetitions} times")
            qprint(f"-------------------")
            qprint(f"- number of repetitions:", number_of_repetitions)
            qprint(f"- job tags:", job_tags)   

        for batch in batches:
            job = backend.run(circuits=batch, shots = number_of_repetitions)
            sleep(0.5) #not sure if necessarry?
            job_id_list.append(job.job_id())
            job.update_tags(job_tags)
            sleep(0.1)
    pass

    return job_id_list

#-----------------------
# PART 3: Not used yet
#-----------------------
def retrieve_results(device_name:str = 'ibm_sherbrooke',
                    provider_instance:str = 'ibm-q/open/main',
                    job_IDs:List[str] =[],
                    original_circuits: Union[CircuitCollection,bool]= False,
                    original_circuits_path: Union[bool,str,Path] = False,
                    save_experiment_results: Union[bool,str,Path]=False,
                    backup_original_job_results: Union[bool,str,Path]=False,
                    from_jobs_backup: Union[bool,str,Path] = False,
                    overwrite = False,
                    verbose_log = False) -> ExperimentResults:   
    
    """
    Retrieves and processes results from an IBM Quantum backend based on provided job IDs.

    Parameters
    ----------
    provider : IBMProvider
        The IBMProvider instance used to retrieve results.
    job_IDs : List[str]
        List of job IDs for which results are to be retrieved.
    original_circuits : Union[CircuitCollection, bool], optional
        The original circuits used in the jobs. Set to False if not provided. Defaults to False.
    original_circuits_path : Union[bool, str, Path], optional
        Path to the file containing the original circuits. Defaults to False.
    config_file_path : Union[bool, str, Path], optional
        Path to the file containing the original submission config. Defaults to False.
    save_experiment_results : Union[bool, str, Path], optional
        Path to save the processed experiment results. Defaults to False.
    backup_original : Union[bool, str, Path], optional
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
    
    _, _, provider = connect(name = device_name, method = "PROVIDER", instance = provider_instance, verbose_log = verbose_log)
    backend = provider.backend
    #[1] Download original results from provider
    original_results, circuit_labels = ibmutils.download_results(backend = backend, 
                                                                 job_IDs = job_IDs,
                                                                 backup =  backup_original_job_results,
                                                                 from_jobs_backup = from_jobs_backup,
                                                                 overwrite = overwrite);
    
    #[2] Retrieve, prepare circuits collection; look for discrepancies
    if original_circuits_path or original_circuits:
        #this will compare circuits and provide original circuits if they are not provided in original_circuits, but path is filled
        original_circuits = ibmutils.compare_circuits(circuit_labels, original_circuits_path, original_circuits );

    #[3] Retrieve, prepare circuits collection; look for discrepancies
    if not original_circuits:
        warprint("METADATA MISSING. No original circuits file provided. ExperimentResults will miss metadata. Always try to match experiment with original circuit collection.")
        processed_circuits = CircuitCollection()
        processed_circuits.job_IDs = job_IDs
        processed_circuits.circuits = [ convert.bitstring_to_ndarray(label) for label in circuit_labels ]
        processed_circuits.circuits_labels = circuit_labels

    else:
        processed_circuits = original_circuits
        if processed_circuits.job_IDs:
            warprint("INFO: CricuitCollection already contains job_IDs. Will update them for ExperimentResult object based on the given list of job IDs")
        processed_circuits.job_IDs = job_IDs 

    #[4] Prepare results object
    experiment_results = ibmutils.ibm_to_qrem_restults( qrem_circuits = processed_circuits, 
                                                        original_results = original_results,
                                                        save_path=save_experiment_results,
                                                        overwrite = overwrite)
    

    return experiment_results
    

def _test_results():
    from qrem.qtypes import CircuitCollection
    from qrem import load_config

    #[1] Initial data necessary for the test
    circuit_collection = CircuitCollection()
    #circuit_collection.import_json(json_import_path = "D:\\WORKDESK\\QREM_SECRET_DEVELOPMENT\\.local_private\\ibm\\ibm_connection_test\\ibm_sherbrooke_input_circuit_collection.json")
    circuit_collection.import_json(json_import_path = "D:\\WORKDESK\\QREM_SECRET_DEVELOPMENT\\.local_private\\ibm\\ibm_connection_test\\ibm_sherbrooke_input_circuit_collection.json")

    job_IDs = ["cmehkrzwcht0008ybs0g","cmehkrf4z12g008ndkmg","cmehkqpwcht0008ybs00","cmehkpyzjkb0008cs440","cmehkp6te72g008jt8x0","cmehkmywq3k0008t08hg","cmehkmey4c00008hytdg","cmehkkezjkb0008cs43g","cmehkjpwcht0008ybrzg","cmehkhyzjkb0008cs430","cmehkh6wq3k0008t08h0","cmehkg6wq3k0008t08gg","cmehkfdwcht0008ybrxg","cmehke54z12g008ndkk0","cmehkd5wq3k0008t08g0"]
    

    #[2] test the IBM connection version
    #CONFIG_PATH = ["--config_file", "D:\\WORKDESK\\QREM_SECRET_DEVELOPMENT\\.local_private\\ibm\\ibm_connection_test\\test_ibm.ini"]
    # CONFIG_PATH = ["--config_file", "D:\\WORKDESK\\QREM_SECRET_DEVELOPMENT\\.local_private\\ibm\\ibm_connection_test\\test_ibm.ini"]
    # config = load_config(cmd_args = CONFIG_PATH, verbose_log = True)
    #backend, service, provider = connect(name = "config.device_name", method = "PROVIDER", instance = config.provider_instance, verbose_log = True)
    #backend = provider.backend
    # original_results, circuit_labels = ibmutils.download_results(backend = backend, 
    #                                                             job_IDs = job_IDs,
    #                                                             backup =  False, #"/mnt/bkpT7/work_piotr/test_ibm"
    #                                                             from_jobs_backup= Path("D:\\WORKDESK\\QREM_SECRET_DEVELOPMENT\\.local_private\\ibm\\ibm_connection_test")
    #                                                             overwrite = True);

    
    #[3] test loading backuped jobs
    backend = None    
    original_results, circuit_labels = ibmutils.download_results(backend = backend, 
                                                                job_IDs = job_IDs,
                                                                backup =  False, #"/mnt/bkpT7/work_piotr/test_ibm"
                                                                from_jobs_backup= Path("D:\\WORKDESK\\QREM_SECRET_DEVELOPMENT\\.local_private\\ibm\\ibm_connection_test"),
                                                                overwrite = True);
                      
    #[4] Compare circuits from the ones send and ones received                                    
    pass     
    processed_circuits = ibmutils.compare_circuits(circuit_labels, False, circuit_collection );
    

    #[5] Prepare results object
    experiment_results = ibmutils.ibm_to_qrem_restults( qrem_circuits = processed_circuits, 
                                                        original_results = original_results,
                                                        save_path="D:\\WORKDESK\\QREM_SECRET_DEVELOPMENT\\.local_private\\ibm\\ibm_connection_test\\results.pkl",
                                                        overwrite = True,
                                                        format = "pkl")
    
    pass
    values_example = list(experiment_results.counts.values())[0]
    print(np.sum(values_example[1]))
    pass

# test:
if __name__ == "__main__":
    _test_results()
   




#----------------------------------------------------
#----------------------------------------------------
#---IBM circuit_runner runtime inputs (runtime_inputs)
#----------------------------------------------------
#----------------------------------------------------
	# # A circuit or a listof QuantumCircuits or a list
	# # of QASM 2.0 strings orna Dict of QasmQobj/PulseQobj.
	# 'circuits': None, # [object,array] (required)

	# # Number of repetitions of each circuit, for sampling. Default: 1024.
	# 'shots': 4000, # integer

	# # User input that will bemattached to the job and will be copied to the corresponding result header.
	# 'header': None, # object

	# # Whether to reset the qubits to the ground state for each shot.
	# 'init_qubits': True, # boolean

	# # The measurement output level. Level 2 is the discriminated measurement counts. Level 1 is the IQ measurement kernel values.
	# 'meas_level': 2, # integer

	# Whether to return per-shot measurement bitstrings.
	# 'memory': False, # boolean
    
	# Whether to use excited state
	# promoted (ESP) readout for measurements,
	# which are the terminal instruction
	# to a qubit. ESP readout
	# can offer higher fidelity than
	# standard measurement sequences.
	# 'use_measure_esp': None # boolean

	# # Delay between programs in seconds.
	# 'rep_delay': None, # number

	# # List of measurement LO frequencies in Hz. Overridden by schedule_los if specified.
	# 'meas_lo_freq': None, # array

	# Type of measurement data to return. Only applicable for meas_level=1.
	# If 'single' is specified, per-shot information is returned. If 'avg'
	# is specified, average measurement output is returned.
	# 'meas_return': None, # string


	# # List of job level qubit drive LO frequencies in Hz. Overridden by schedule_los if specified.
	# 'qubit_lo_freq': None, # array

	# Experiment level LO frequency configurations for qubit drive and measurement channels, in Hz.
	# 'schedule_los': None, # array

