"""qrem.providers.ibm module contains all necessary functions for running a characterisation/mitigation/benchmarking experiment
on IBM Quantum machines via QiskitRuntimeService or IBMServiceBackend.
"""
import os
import orjson
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

from qrem.common.providers.ibmutils import ibmutils 

# qrem imports
from qrem.common.constants import EXPERIMENT_TYPE_SYMBLOS
from qrem.common.printer import qprint, errprint, warprint
from qrem.common import convert, io
from qrem.common.utils import get_full_object_size
from qrem.circuit_collection import CircuitCollection
from qrem.experiment_results import ExperimentResults


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

#Optional instance: 'ibm-q-psnc/internal/reservations'
def connect(name:str = "", 
            channel: Optional[str] = QISKIT_IBM_CHANNEL, 
            method:str = "RUNTIME", #available are "RUNTIME", "PROVIDER", "RUNTIME SESSIONS", "JOB EXECUTE"
            instance: Optional[str] = 'ibm-q/open/main', 
            verbose_log = False) -> Tuple[Union[IBMServiceBackend,IBMBackend],Optional[QiskitRuntimeService],Optional[IBMProvider]]:
    """
    Initializes connection to the backend based on the API_TOKEN defined in the environment variables.
    Depending on the <method> parameter will start either QiskitRuntimeService object or IBMProvider object

    Parameters
    ----------
    channel: str, default='ibm_quantum' 
        specifies which channel to use for connection (ibm_quantum or ibm_cloud)
    method: str, default='RUNTIME' 
        specifies connection method. Available are "RUNTIME", "PROVIDER", "RUNTIME SESSIONS", "JOB EXECUTE"
    instance: str, default='ibm-q/open/main' 
        specifies instance for connection, example values are: 'ibm-q/open/main' or 'ibm-q-psnc/internal/reservations' 
    verbose_log: bool, default=False
        turn on verbose logging for more printouts with info
        
    Returns
    -------
    backend
        Instance of backend object returned by QiskitRuntimeService (IBMServiceBackend) or IBMProvider (IBMBackend)
    service
        Connected Instance of QiskitRuntimeService. Null if connecting via IBMProvider
    provider
        Connected Instance of IBMProvider. Null if connecting via QiskitRuntimeService
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
    Helper function for qrem.common.providers.ibm.connect(...) to connect via QiskitRuntimeService
    """
    if instance == '':
        service = QiskitRuntimeService(channel = channel, token=QISKIT_IBM_TOKEN)
    else:
        service = QiskitRuntimeService(channel = channel, token=QISKIT_IBM_TOKEN, instance = instance)

    return service
  
def _connect_via_provider(backend_name:str = "", channel: Optional[str] = 'ibm_quantum', instance: Optional[str] = 'ibm-q/open/main', verbose_log = False):
    """
    Helper function for qrem.common.providers.ibm.connect(...) to connect via IBMProvider
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
    get_valid_qubits_properties queries the backend object to provide information about which qubits are valid
    (not flagged faulty by IBM and below given sx gate error treshold ). Returns an elaborate dictionary with 

    Parameters
    ----------
    backend: Union[IBMBackend,IBMServiceBackend]
        Instance of backend object returned during connection by QiskitRuntimeService (IBMServiceBackend) or IBMProvider (IBMBackend)
    threshold: floatm default = 0
        gate error treshold. If 0 (default value), will not discard qubits based on gate error tresholding
    verbose_log: bool, default=False
        turn on verbose logging for more printouts with info
        
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
        Instance of CircuitCollection object, needs to be already initialized with chosen circuits, proper valid qubit_indices and all other fields.
        
    Returns
    -------
    qiskit_circuits: QuantumCircuit
        Lsit of QuantumCircuits, with barrier between each of the qubit (read more on qiskit docs about QuantumCircuits)
    """     

    # TODO (PP) sort out repeating circuits already in CircuitCollection
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


def _add_measurements(
        
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

def _circuits_translator_DDOT( circuit:np.ndarray, 
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

def _circuits_translator_QDOT( circuit:np.ndarray, 
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

#-----------------------
# PART 2: Run circuits
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
    Executes circuits prepared in QISKIT format (qiskit_circuits) on the backend provided by a qrem.common.providers.ibm.connection(...) function.

    Parameters
    ----------
    qiskit_circuits: List[QuantumCircuit]
        List of QuantumCircuits, not yet transpiled, to be executed on chosen IBM machine
    method: str, default = "RUNTIME"
        Connection method, determined by the execution of qrem.common.providers.ibm.connection(...) (should be same as there)
    job_tags: List[str] default=["QREM_JOB"]
        List of tags to be applied to each job
    service: QiskitRuntimeService, default=None
        Connected Instance of QiskitRuntimeService. Can be None if connecting via IBMProvider
    backend: Union[IBMServiceBackend,IBMBackend]
        Provide connected Backend here
    number_of_repetitions:int, default=1024
        Number of repetitions for the circuit collection (persistently named number of shots by providers)
    log_level: str, default = "WARNING"
        Logging level to log job execution on machine. The valid log levels are: DEBUG, INFO, WARNING, ERROR, and CRITICAL. The default level is WARNING.
        look https://qiskit.org/ecosystem/ibm-runtime/stubs/qiskit_ibm_runtime.RuntimeOptions.html#qiskit_ibm_runtime.RuntimeOptions for more information
        
    Returns
    -------
    job_id_list: List[str]
        Lsit of submitted job IDs
    """         
    
    
    #TODO (PP) check for null Service, provider and backend type - for DEBUG and other purposes

    # The valid log levels are: DEBUG, INFO, WARNING, ERROR, and CRITICAL. The default level is WARNING.
    # translate to IBM - qiskit Circuits
    if(number_of_repetitions > backend.max_shots):
        warprint(f"WARNING: Number of repetitions ({number_of_repetitions}) is larger than max number of shots ({backend.max_shots}) defined on chosen backend {backend.name}, defaulting to: ", f"({backend.max_shots})")
    # transpile all circuits
    transpiled_circuits = ibm_translate.transpile_circuits(qiskit_circuits,backend)

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

                # XXXXXXXXXXXXXXXXXXXXXXXXX

                # job.result() is blocking, so this job is now finished and the session can be safely closed.
                # Close the session only if all jobs are finished, and you don't need to run more in the session
            sleep(0.5) #not sure if necessarry?
            for job in job_list:
                job_id_list.append(job.job_id())
                # Do not close here, the job might not be completed!
                result = job.result()
                #TODO (PP) UNCERTAIN: shall we already save this result?
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
def retrieve_results(provider, 
                    job_IDs:List[str],
                    original_circuits: Union[CircuitCollection,bool]= False,
                    original_circuits_path: Union[bool,str,Path] = False,
                    save_experiment_results: Union[bool,str,Path]=False,
                    backup_original: Union[bool,str,Path]=False,
                    overwrite = False):   
    

    #[1] Download original results from provider
    original_results, circuit_labels = ibmutils.download_results(provider = provider, 
                                                                 job_IDs = job_IDs,
                                                                 backup =  backup_original,
                                                                 overwrite = overwrite);
    
    #[2] Retrieve, prepare circuits collection; look for discrepancies
    if(original_circuits_path):
        #this will compare circuits and provide original circuits if they are not provided in original_circuits
        original_circuits = ibmutils.compare_circuits(circuit_labels, original_circuits_path, original_circuits );

    #[3] Retrieve, prepare circuits collection; look for discrepancies
    if not original_circuits:
        warprint("METADATA MISSING. No original circuits provided. ExperimentResults will miss metadata")
        original_circuits = CircuitCollection("Unknown")
        original_circuits.circuits = [ convert.bitstring_to_ndarray(label) for label in circuit_labels ]
        original_circuits.circuits_labels = circuit_labels
    experiment_results = ExperimentResults()
    experiment_results.from_ibm(qrem_circuits = original_circuits, original_results = original_results, circuit_labels = circuit_labels)
    
    
    export_path = io.prepare_outfile(outpath = save_experiment_results, overwrite = overwrite , default_filename= 'experiment_results.pkl')
    experiment_results.export_json(json_export_path = export_path)

    return 
    




# test:
if __name__ == "__main__":
    pass
    # from datetime import datetime

    # from qrem.types import CircuitCollection
    # from qrem.common.experiment.tomography import compute_number_of_circuits, create_circuits

    # experiment_type = 'DDOT'
    # device_name = 'ibmq_kolkata'
    # provider_instance = ''  # TODO: input something

    # provider = IBMProvider(instance=provider_instance)
    # backend = provider.get_backend(device_name)

    # [0] get from provider information on which qubits have gate error belowe threshold:
    # qubit_indices = get_qubits_below_threshold(backend)
    # no_qubits = len(qubit_indices)

    # # [1] create the circuits:
    # no_circuits = compute_number_of_circuits(experiment_type=experiment_type, number_of_qubits=no_qubits)
    # circuit_collection = create_circuits(experiment_type, no_qubits, no_circuits)
    # #TODO: [1] should result in a CircuitCollection
    # #TODO: add checking completeness
    #
    # # PLACEHOLDER instead of [1]:
    # circuit_collection = CircuitCollection('test_name')
    # print(circuit_collection.get_dict_format())
    # dictionary_to_load = {'experiment_type': 'qdot',
    #                       'circuits_list': [[0, 5, 3], [1, 2, 0], [3, 1, 4]],
    #                       'qubit_indices': [0, 2, 5],
    #                       'gate_error_threshold': 0.005,
    #                       'no_shots': 1,
    #                       'datetime_created_utc': datetime.utcnow(),
    #                       'author': 'tester',
    #                       'notes': 'some string note'}
    # circuit_collection.load_from_dict(dictionary_to_load)

    # [2] execute the circuits:
    # job_IDs = execute_circuits(device_name, circuit_collection)

    # [3] download results:
    # job_IDs = ["",]#TODO: input something
    # results_qiskit = get_results(provider, job_IDs)
    # results_qrem =




#----------------------------------------------------
#----------------------------------------------------
#---IBM circuit_runner runtime inputs (runtime_inputs)
# TODO (PP) (MO) talk options for init_qubits,  memory,  use_measure_esp
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

