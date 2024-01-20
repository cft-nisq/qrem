import os
import time
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm
import qiskit
from qiskit import Aer, IBMQ
from qiskit.providers.jobstatus import JobStatus

from qrem.common.convert import reorder_probabilities
from qrem.common import povmtools

from qrem.common import io
from qrem.common.printer import qprint

#ORGANIZATION - poorly documented code (MO)

def get_qiskit_provider(backend_name: str,
                        provider_data: Dict[str, str]=None):
    IBMQ.load_account()
    if backend_name.lower() in ['ionq.qpu', 'ionq.simulator']:
        from azure.quantum.qiskit import AzureQuantumProvider

        if provider_data is None:
            AZURE_RESOURCE_ID = os.environ['AZURE_RESOURCE_ID_IONQ']
            AZURE_LOCATION = os.environ['AZURE_LOCATION_IONQ']
        else:
            AZURE_RESOURCE_ID = provider_data['AZURE_RESOURCE_ID_IONQ']
            AZURE_LOCATION = provider_data['AZURE_LOCATION_IONQ']

        provider = AzureQuantumProvider(
            resource_id=AZURE_RESOURCE_ID,
            location=AZURE_LOCATION)

    else:
        if provider_data is None:
            IBMQ_hub = os.environ['IBMQ_HUB']
            IBMQ_group = os.environ['IBMQ_GROUP']
            IBMQ_project = os.environ['IBMQ_PROJECT']
        else:
            IBMQ_hub = provider_data['IBMQ_HUB']
            IBMQ_group = provider_data['IBMQ_GROUP']
            IBMQ_project = provider_data['IBMQ_PROJECT']

        provider = IBMQ.get_provider(hub=IBMQ_hub,
                                     group=IBMQ_group,
                                     project=IBMQ_project)

    return provider

def add_gate_to_circuit(circuit,
                        qreg,
                        q_index,
                        unitary):
    # TODO: Check whether this is needed - at some point I remember there were troubles with parametrizing those two unitaries
    if unitary[0, 0] == 1 and unitary[1, 1] == 1:
        pass
    elif unitary[0, 1] == 1 and unitary[1, 0] == 1:
        circuit.X(qreg[q_index])

    else:
        # get angles for single-qubit state change unitary
        current_angles = povmtools.get_su2_parametrizing_angles(unitary)

        # implement unitary
        circuit.u3(current_angles[0],
                   current_angles[1],
                   current_angles[2],
                   qreg[q_index])
    return circuit, qreg


#ORGANIZE: copied to experiment_results. Delete when refactoring complete
def get_frequencies_from_counts(counts_dict,
                                crs=None,
                                classical_register=None,
                                shots_number=None,
                                reorder_bits=True):
    if crs is None:
        crs = len(list(list(counts_dict.keys())[0]))

    d = 2 ** crs

    if classical_register is None:
        classical_register = ["{0:b}".format(i).zfill(crs) for i in range(d)]

    normal_order = []

    for j in range(d):
        if classical_register[j] in counts_dict.keys():
            counts_now = counts_dict[classical_register[j]]
            normal_order.append(counts_now)

        else:
            normal_order.append(0)
    if reorder_bits:
        frequencies = reorder_probabilities(normal_order, range(crs)[::-1])
    else:
        frequencies = normal_order

    if shots_number is None:
        frequencies = frequencies / np.sum(frequencies)
    else:
        frequencies = frequencies / shots_number

    return frequencies


def download_multiple_jobs(backend_name,
                           job_IDs_list,
                           provider_data=None):
    IBMQ.load_account()
    if backend_name in ['qasm_simulator', 'statevector_simulator', 'unitary_simulator']:
        raise ValueError('Local simulators do not store jobs online.')

    provider = get_qiskit_provider(backend_name=backend_name,
                                   provider_data=provider_data)

    backend = provider.get_backend(backend_name)

    all_jobs = []
    for job_ID in tqdm(job_IDs_list):
        # qprint('Getting job with ID:', job_ID)
        job = backend.retrieve_job(job_ID)
        # qprint('Got it!')
        all_jobs.append(job)
    return all_jobs


#ORGANIZE: copied to experiment_results, but possibly not needed.
def add_counts_dicts(all_counts, modulo, dimension):
    frequencies = [np.zeros(dimension) for i in range(modulo)]

    for counts_index in tqdm(range(len(all_counts))):
        true_index = counts_index % modulo

        freqs_now = povmtools.counts_dict_to_frequencies_vector(all_counts[counts_index], True)
        frequencies[true_index][:] += freqs_now[:]

        # print(freqs_now)
    for i in range(modulo):
        frequencies[i] *= 1 / np.sum(frequencies[i])

    return frequencies

#MOcomm is it just for qiskit or more general

def run_batches(batches,
                backend_name,
                shots,
                saving_IDs_dictionary=dict(saving=False, directory=None, file_name=None, dictionary_to_save={}),
                provider_data:Optional[Dict[str,str]] = None, verbose_log = True):
    """
    Description:
        This function takes a list of circuit batches and sends them for execution on the specified backend.
        User can choose to save the resulting jobs' IDs, using parameters in saving_IDs_dictionary
    Parameters:
        :param batches: (list(list(qiskit.circuit.quantumcircuit.QuantumCircuit))): list of batches, where each batch is
        a list of qiskit circuits and each batch has length <= maximum number of circuits in job
        :param backend_name: (str): if value is of the names of a qiskit simulator: ('qasm_simulator',
        'statevector_simulator', 'unitary_simulator'), provider is chosen to be Aer; in other cases provider is taken
         from 'provider_data'
        #TODO: consider cases where other backend names are provided (e.g. aer_simulator)
        :param shots: number of repetitions of each circuit from batches
        :param saving_IDs_dictionary:
        :provider_data: Dict[str]: contains access data required by providers, like user ID etc. If None, as is default,
        this data is obtained from environmental variables by function get_qiskit_provider (see function for details)
    Returns:
        jobs: list of qiskit jobs (results of qiskit.execute())
    Notes:

    """
    saving = saving_IDs_dictionary['saving']

    qprint('\nSending jobs to execution on: ', backend_name + '.')
    qprint('Number of shots: ', str(shots) + ' .')
    qprint('Target number of jobs: ', str(len(batches)) + ' .')
    print()

    iterations_done = 0
    wait_time_in_minutes = 10
    jobs = []

    qiskit_simulators = ['qasm_simulator', 'statevector_simulator', 'unitary_simulator']
    braket_simulators = ['BraketLocalBackend']
    braket_quantum_devices = ['Aspen-M-1', 'Aspen-M-2']

    # set provider and backend according to passed argument 'backend_name':
    if backend_name in qiskit_simulators:
        backend = Aer.get_backend(backend_name)
    else:
        provider = get_qiskit_provider(backend_name=backend_name,
                                       provider_data=provider_data)
        backend = provider.get_backend(backend_name)

    # Run the circuits on backend:
    while iterations_done < len(batches):
        qprint('job number:', str(iterations_done))
        circuits = batches[iterations_done]
        try:
            time.sleep(2)
            qprint("Sending quantum program to: ", backend_name + '.')
            # below we use pure qiskit:
            if backend_name not in braket_simulators and backend_name not in braket_quantum_devices:
                job = qiskit.execute(circuits, backend, shots=shots, max_credits=200)#, qobj_id=qobj_id)

            # below we use braket (note that we use 'job' here in the qiskit sense, in braket this is called 'task'):
            else:
                job = backend.run(circuits, shots=shots, disable_qubit_rewiring=True)
                # save the circuit names as metadata to use in get_counts_from_jobs and get_counts_from_result_object:
                job.metadata['circuit_names'] = [circuit.name for circuit in circuits]
                # alternatives: job._tasks[idx]._result.task_metadata.braketSchemaHeader.name; Tag property of task
            jobs.append(job)

            if saving and backend_name not in ['qasm_simulator', 'statevector_simulator', 'unitary_simulator']:
                job_ID = job.job_id()
                dict_to_save = saving_IDs_dictionary['dictionary_to_save']
                dict_to_save['job_ID'] = job_ID

                io.save(dictionary_to_save=dict_to_save,
                         directory=saving_IDs_dictionary['directory'],
                         custom_name=saving_IDs_dictionary['file_name'] + '_job%s' % iterations_done,
                         overwrie = False)
            while job.status() == JobStatus.INITIALIZING:
                print(job.status())
                time.sleep(2)
            qprint("Program sent for execution to: ", backend_name + '.')

        except BaseException as ex:
            print('There was an error in the circuit!. Error = {}'.format(ex))
            print(f'Waiting {wait_time_in_minutes} minute(s) before next try.')
            time.sleep(wait_time_in_minutes * 60)
            continue

        print()
        iterations_done += 1
    return jobs
