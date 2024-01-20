""" Needs to be refactored into somewhere else, """

from typing import List, Dict, Union, Optional
import numpy as np
from tqdm import tqdm
from qiskit.result import Result

from qrem.common.printer import qprint
from qrem.common.convert import reorder_probabilities

def get_shots_results_from_qiskit_result(result_object):
    pass


# ORGANIZE: Three functions below are copied from qiskit_utilities. and may be not necessary
def get_frequencies_array_from_probabilities_list(frequencies_list:List[Union[List[float], np.ndarray]],
                                                  reverse_order:Optional[bool]=False):
    """
    :param frequencies_list: list of probabilities estimated in experiments
    :param reverse_order: specify whether probabiltiies lists should be reordered, which corresponds
                          to changing qubits' ordering

    """

    number_of_qubits = int(np.log2(len(frequencies_list[0])))
    frequencies_formatted = frequencies_list
    if reverse_order:
        frequencies_formatted = [reorder_probabilities(probs, range(number_of_qubits)[::-1])
                                  for probs in frequencies_list]

    frequencies_array = np.ndarray(shape=(len(frequencies_list), len(frequencies_list[0])))

    for probe_state_index in range(len(frequencies_list)):
        frequencies_array[probe_state_index][:] = frequencies_formatted[:]
    return frequencies_array


def get_frequencies_array_from_results(results_list: List[Result]) -> np.ndarray:
    """
    Description:
        Creates an array of frequencies from given qiskit job results. This method is working with
        qiskit 0.16. The shape of the array is
            c x 2 ** q,
        where c denotes circuits number and q denotes number of qubits.
    Parameters:
        :param results_list: List of qiskit jobs_list results.
    Returns:
        np.ndarray with shape=0 if there were no circuits in the job, or with shape c x 2 ** q
        containing frequencies data for each possible state.
    Notes:
        Possible states are numbered increasingly from |00000 ... 0>, |10000 ... 0> up to |1111 ... 1>.
    """

    all_circuits_number = sum(len(results.results) for results in results_list)

    if all_circuits_number == 0:
        return np.ndarray(shape=0)

    # The length of a state describes how many qubits were used during experiment.
    number_of_qubits = len(next(iter(results_list[0].get_counts(0).keys())))

    classical_register = ["{0:b}".format(i).zfill(number_of_qubits) for i in range(2 ** number_of_qubits)]
    frequencies_array = np.ndarray(shape=(all_circuits_number, len(classical_register)))

    # TR: This has to be rewritten as it's too nested.
    for results in results_list:
        number_of_circuits_in_results = len(results.results)
        for i in range(number_of_circuits_in_results):
            counts = results.get_counts(i)
            shots_number = results.results[i].shots

            # accounting for reversed register in qiskit
            normal_order = []
            for j in range(len(classical_register)):
                if classical_register[j] in counts.keys():
                    normal_order.append(counts[classical_register[j]] / shots_number)
                else:
                    normal_order.append(0)
            # frequencies = reorder_probabilities(normal_order, range(states_len))
            frequencies = reorder_probabilities(normal_order, range(number_of_qubits)[::-1])

            frequencies_array[i][:] = frequencies[:]

    return frequencies_array


def get_counts_from_result_object(results_object,
                                  counts_dictionary_to_update={},
                                  circuit_names_list=None,
                                  job_index=0):
    """
    This functions takes a qiskit Result object and returns a dictionary with counts for each Experiment in this object.
    returns:
    counts_dictionary_to_update[str, counts]: the key is the name of a circuit, the value is a dictionary of counts,
    where the key is a string denoting classical outcome and the value the number of its occurrences in given experiment.
    The circuits are named according to convention from get_circuits in SeparableCircuitsCreator:
    circuit_name = "experiment name" + "-" + "circuit label" +
        "no" + "integer identifier for multiple implementations of the same circuit", e.g.: "DDOT-010no3"
    """
    number_of_experiments = len(list(results_object.results))
    for exp_index in tqdm(range(number_of_experiments)):
        try:
            circuit_name_now = results_object.results[exp_index].header.name
            counts_now = results_object.get_counts(circuit_name_now)
        except (KeyError, AttributeError) as warn:
            # print(warn)
            if circuit_names_list is None:
                qprint(f"Experiment: no {exp_index} does not have a header. "
                               f"using_default_name")
                circuit_name_now = f"default_name_no_{job_index}-{exp_index}"
                counts_now = results_object.get_counts(exp_index)
            else:
                circuit_name_now = circuit_names_list[exp_index]
                counts_now = results_object.get_counts(exp_index)
        counts_dictionary_to_update[circuit_name_now] = counts_now
    return counts_dictionary_to_update

def get_counts_from_qiskit_jobs(jobs_list,
                         return_job_headers=False) -> Dict[str, Dict[str, int]]:
    """
    This function takes a list of qiskit jobs (e.g. what 'run_batches' returns) and returns a dictionary of outcomes of
    each of the circuits in the list, where keys are circuit names and values are dictionaries of counts (where the key
    is a string denoting classical outcome and the value the number of its occurrences in given experiment)

    """
    qprint('Getting counts...')
    counts_dictionary, job_headers = {}, []

    for job_index in tqdm(range(len(jobs_list))):
        job_now = jobs_list[job_index]
        job_header_now = job_now.result().qobj_id
        job_headers.append(job_header_now)
        results_object_now = job_now.result()
        circuit_name_list = None
        if hasattr(job_now, 'metadata'):
            if 'circuit_names' in job_now.metadata:
                circuit_name_list = job_now.metadata['circuit_names']  # this attribute exists only for braket jobs

        counts_dictionary = get_counts_from_result_object(results_object=results_object_now,
                                                          counts_dictionary_to_update=counts_dictionary,
                                                          circuit_names_list=circuit_name_list,
                                                          job_index=job_index)
    qprint("GOT IT!")

    if return_job_headers:
        return counts_dictionary, job_headers
    else:
        return counts_dictionary
