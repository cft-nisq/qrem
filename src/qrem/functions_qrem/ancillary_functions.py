
#ORGANIZE - would be great to know what is the role of all these packages
#--------------------------------------------------
# standard and technical imports 
#--------------------------------------------------
import os
import warnings
import re
import itertools #ORGANIZE - would be good if we trie dto be effitienta and used fast structures for iterations. @Piotr: this should be fast
import multiprocessing
import datetime as dt 
from collections import defaultdict
from typing import List, Dict, Optional, Callable, Tuple
import pickle
 
#--------------------------------------------------
# scientific imports
#--------------------------------------------------
import cmath as c
import numpy as np
import pandas as pd
from colorama import Fore, Style

from qrem.common.printer import qprint
from qrem.common import utils
from qrem.common import convert
#--------------------------------------------------
# local imports
#--------------------------------------------------
#NONE

    





#MOVE_TO >> core.utils
# core.utils.DitstringConverter
# MO - check if usage makes sense? Can be iefficient
def get_qubits_keystring(list_of_qubits: List[int]) -> str:
    """ from subset of qubit indices get the string that labels this subset
        using convention 'q5q6q12...' etc.
    :param list_of_qubits: labels of qubits

    :return: string label for qubits

     NOTE: this function is "inverse operation" to get_qubit_indices_from_keystring.
    """

    if list_of_qubits is None:
        return None
    return 'q' + 'q'.join([str(s) for s in list_of_qubits])

#MOVE_TO >> core.utils core.numerics
def all_possible_bitstrings_of_length(number_of_bits: int,
                reversed: Optional[bool] = False):
    """Generate outcome bitstrings for n-qubits.

    Args:
        number_of_qubits (int): the number of qubits.

    Returns:
        list: arrray_to_print list of bitstrings ordered as follows:
        Example: n=2 returns ['00', '01', '10', '11'].
"""
    if (reversed == True):
        return [(bin(j)[2:].zfill(number_of_bits))[::-1] for j in list(range(2 ** number_of_bits))]
    else:
        return [(bin(j)[2:].zfill(number_of_bits)) for j in list(range(2 ** number_of_bits))]

#MOVE_TO >> core.utils core.numerics (PP) core.utils.DitConverter
#(PP): compare with povmtools version and usage across package
#(PP): change name to bitstring inside
def get_classical_register_bitstrings(qubit_indices: List[int],
                                      quantum_register_size: Optional[int] = None,
                                      rev: Optional[bool] = False):
    """
    Register of qubits of size quantum_register_size, with only bits corresponding to qubit_indices
    Gets list of bitstrings of length quantum_register_size when only bits in qubit_indicies can be 0 and 1, others have to be 0

    Qubits indices are always indexed from right to get proper output, use ref if you need a format with indexing from left.

    use rev when input indices were 


    varying

    :param qubit_indices:
    :param quantum_register_size:
    :param rev:
    :return:
    """

    # TODO FBM: refactor this function.

    # again assumes that qubit_indices contains unique values
    if quantum_register_size is None:
        quantum_register_size = len(qubit_indices)

    if quantum_register_size == 0:
        return ['']

    if quantum_register_size == 1:
        return ['0', '1']

    all_bitstrings= all_possible_bitstrings_of_length(quantum_register_size, rev)
    not_used = []

    for j in list(range(quantum_register_size)):
        if j not in qubit_indices:
            not_used.append(j)

    bad_names = []
    for bitstring in all_bitstrings: #0000111
        for k in (not_used):
            rev_name = bitstring[::-1] #1110000 reverses order of string - why?
            if rev_name[k] == '1':
                bad_names.append(bitstring)

    relevant_names = []
    for bitstring in all_bitstrings:
        if bitstring not in bad_names:
            relevant_names.append(bitstring)

    return relevant_names

#MOVE_TO >> core.utils core.numerics (PP) core.utils.DitConverter
def register_names_qubits(qubit_indices: List[int],
                          quantum_register_size: Optional[int] = None,
                          rev: Optional[bool] = False):
    # depreciated
    return get_classical_register_bitstrings(qubit_indices=qubit_indices,
                                             quantum_register_size=quantum_register_size,
                                             rev=rev)

#MOVE_TO >> core.utils? DELETE
def convert_qubits_string_to_tuple(qubits_string):
    return tuple(convert.get_qubit_indices_from_keystring(qubits_string))
    
#MOVE_TO >> core.constans core.backends
def get_historical_experiments_number_of_qubits(backend_name: str):
    if backend_name == 'ibmq_16_melbourne':
        number_of_qubits = 15
    elif backend_name == 'ASPEN-8':
        number_of_qubits = 23
    elif backend_name.upper() == 'ASPEN-9':
        number_of_qubits = 20
    else:
        raise ValueError('Wrong backend name')

    return number_of_qubits

     


# =======================================================================================
# NUTILS
# =======================================================================================




# =======================================================================================
# IO FILESYSTEM
# =======================================================================================

# #MOVE_TO >> core.io
# def get_local_storage_directory():
#     # please create the environment variable in your virtual environment!
#     # print(get_local_storage_directory())
#     # print(os.listdir(os.environ['LOCAL_STORAGE_LOCATION']))
#     # print(os.environ.keys())

#     return os.environ['LOCAL_STORAGE_LOCATION']


# #MOVE_TO >> core.io
# def open_file_pickle(file_path):
#     with open(file_path, 'rb') as filein:
#         data_object = pickle.load(filein)

#     return data_object



# #MOVE_TO >> core.io TO DELETE PP CAN BE A MARKER FOR UNUSED STUFF
# def save_results_pickle(dictionary_to_save: dict,
#                         directory: str,
#                         custom_name: Optional[str] = None,
#                         get_cwd: Optional[bool] = False):
    
#     from pathlib import Path
#     if directory :
#         dir = Path(directory) 
#         dir.mkdir(mode=0o777, exist_ok=True, parents= True) 
#     else:
#         dir = Path.home().joinpath("qrem_results")
#         dir.mkdir(mode=0o777, exist_ok=True) 
    

#     # if (directory != None):
#     #     #!?! add / at the end ...
#     #     fp0 = [s for s in directory]
#     #     if (fp0[len(fp0) - 1] != '/'):
#     #         fp0.append('/')
#     #     fp = ''.join(fp0)
#     # else:
#     #     fp = ''

#     # Time& date
#     if (get_cwd):
#         cwd = os.getcwd()
#     else:
#         cwd = ''
#     ct3 = gate_proper_date_string()
#     # original_umask = os.umask(0)
#     # os.umask(original_umask)

#     main_directory = cwd + fp

#     # permission_mode=int('0777',8)
#     # os.chmod(main_directory,permission_mode)
#     check_dir = os.path.exists(main_directory)

#     if (check_dir == False):
#         try:
#             os.makedirs(main_directory)

#         except(FileExistsError):
#             import shutil
#             try:
#                 shutil.rmtree(main_directory)
#                 os.makedirs(main_directory)
#             except(FileExistsError):
#                 os.makedirs(main_directory)

#         print( Fore.CYAN + Style.BRIGHT + 'Attention: ' + Style.RESET_ALL + 'Directory ' + '"' + main_directory + '"' + ' was created.')
#         # os.umask(oldmask)
#         # os.chmod(main_directory,permission_mode)
#     if custom_name is None:
#         file_path = str(main_directory + 'Results_Object' + '___' + str(ct3))
#     else:
#         file_path = str(main_directory + custom_name)

#     # os.chmod(main_directory)
#     dict_results = dictionary_to_save

#     add_end = ''

#     if (file_path[len(file_path) - 4:len(file_path)] != '.pkl'):
#         add_end = '.pkl'

#     with open(file_path + add_end, 'wb') as f:
#         pickle.dump(dict_results, f, pickle.HIGHEST_PROTOCOL)












# =======================================================================================
# CANDIDATES TO BE DELETED BELOW
# =======================================================================================

#MOVE_TO >> core.utils / duplicate
# #ORGANIZE - this function is used only once for some flder opeprations (MO)
# def get_drive_path():
#     try:
#         return os.environ['LOCAL_DRIVE_DIRECTORY']
#     except(KeyError):
#         return os.environ['LOCAL_STORAGE_LOCATION']


#MOVE_TO >>  DELETE
# def generate_classical_register_bistrings_product(clusters_list: List[Tuple[int]]):
#     clusters_lengths = [len(cl) for cl in clusters_list]
#     if np.max(clusters_lengths) > 15:
#         raise ValueError(f"Too big cluster with size {np.max(clusters_lengths)}")

#     # number_of_qubits = sum(clusters_lengths)

#     local_registers = {cluster: get_classical_register_bitstrings(qubit_indices=range(len(cluster)))
#                        for cluster in clusters_list}
#     # for cluster in clusters_list:
#     #     register_cluster =
#     #     local_registers.append([(cluster, x) for x in register_cluster])
#     #
#     # cartesian_product = itertools.product(local_registers)
#     #
#     # bitstrings_list = []
#     #
#     # for bitstrings_list_with_cluster in cartesian_product:
#     #     bitstring_global = ['0' for _ in range(number_of_qubits)]
#     #     print(bitstrings_list_with_cluster[0])
#     #     for cluster, bitstring_local in bitstrings_list_with_cluster[0]:
#     #         # print(cluster, bitstring_local)
#     #         for index in range(len(bitstring_local)):
#     #             bitstring_global[cluster[index]] = bitstring_local[index]
#     #
#     #     bitstrings_list.append(bitstring_global)
#     return local_registers



#(PP)ORGANIZE - this function is i) trivial and ii)  nerer used - a cleare candidate for deletion  
#def decompose_qubit_matrix_in_pauli_basis(matrix):
#    return {pauli_label: 1 / 2 * np.trace(matrix @ pauli_matrix)
#            for pauli_label, pauli_matrix in pauli_sigmas.items()}

# def get_bell_basis():
#     return list(___bell_states___.values())


#(PP)ORGANIZE - this class is not used anywhere - and hence it is a candidate for deletion (MO)
# (PP) Also, what class does in the function folder?

# class key_dependent_dict(defaultdict):
#     """
#     This is class used to construct dictionary which creates values of keys in situ, in case there
#     user refers to key that is not present.

#     COPYRIGHT NOTE
#     This code was taken from Reddit thread:
#     https://www.reddit.com/r/Python/comments/27crqg/making_defaultdict_create_defaults_that_are_a/

#     """

#     def __init__(self, f_of_x=None):
#         super().__init__(None)  # base class doesn't get a factory
#         self.f_of_x = f_of_x  # save f(x)

#     def __missing__(self, key):  # called when a default needed
#         ret = self.f_of_x(key)  # calculate default value
#         self[key] = ret  # and install it in the dict
#         return ret


#ORGANIZE: this function does not appear anywther - candidate for deletion

#def get_time_now():
#    ct0 = str(dt.datetime.today())     #MO: in this lane I replaced datetime.datetime.today() to dt.datetime.today() need to return to previous verion if does not work
#    ct1 = str.replace(ct0, ':', '_')
#    ct2 = str.replace(ct1, '.', '_')
#    ct3 = ct2[0:19]
#
#    return ct3



#ORGANIZE: this function does not appear anywther - candidate for deletion (MO)

#def wrapped_multiprocessing_function_depreciated(tuple_of_main_arguments: List[tuple],
#                                                 additional_kwargs: dict,
#                                                 function_to_multiprocess: Callable,
#                                                 number_of_threads: Optional[int] = None,
#                                                 # debug=False,
#                                                 printing=False):
#    """
#
#    :param tuple_of_main_arguments:
#
#    This is list of tuples that is divided into batches that are passed to different threads of
#    multiprocessing. This, therefore should be set of VARIABLE arguments that are passed to the
#    function of interest
#
#    :param additional_kwargs:
#    This is dictionary of arguments that are CONSTANT for all function evaluations.
#    The dictionary is COPIED for each thread and passed to the function.
#
#
#    :param function_to_multiprocess:
#    :param number_of_threads:
#    :return:
# """         if number_of_threads is None:
#         number_of_threads = int(
#             np.min([multiprocessing.cpu_count() - 1, len(tuple_of_main_arguments)]))

#     length = len(tuple_of_main_arguments)

#     # IF there is less arguments than threads, we reduce number of threads
#     if length < number_of_threads:
#         number_of_threads = length

#     division_cores = length // number_of_threads

#     all_indices, arguments = [], []

#     # if debug:
#     # print(length, division_cores)

#     for process_pool_index in range(number_of_threads):
#         if process_pool_index == number_of_threads - 1:
#             slice_now = slice(process_pool_index * division_cores,
#                               -1)
#         else:
#             slice_now = slice(process_pool_index * division_cores,
#                               (process_pool_index + 1) * division_cores)

#         sample_indices_now = tuple_of_main_arguments[slice_now]
#         arguments.append((sample_indices_now,
#                           additional_kwargs))

#     # if debug:
#     #     raise KeyError
#     if printing:
#         qprint(f'Running {number_of_threads} threads!', '', 'green')
#     pool = multiprocessing.Pool(number_of_threads)
#     results = pool.starmap_async(function_to_multiprocess,
#                                  arguments)
#     pool.close()
#     pool.join()

#     res_multiprocessing = results.get()

#     results_dictionary_all = {}

#     for dict_now in res_multiprocessing:
#         results_dictionary_all = {**results_dictionary_all, **dict_now}

#     # last_run_indices = list(range(int(division_cores * number_of_threads), length))
#     # last_mp = [([tuple_of_main_arguments[x]], additional_kwargs) for x in last_run_indices]

#     # if debug:
#     #     print('hey bro', last_run_indices, len(last_run_indices))

#     return results_dictionary_all

# """


#ORGANIZE - the function below is not used throughout the project - I commented it out (MO)

#def try_to_do_something_multiple_times(what_to_do: Callable,
#                                       print_string: str,
#                                       how_many_tries=1000):
    # """
    # Codes try to run "what_to_do" callable "how_many_tries" times,
    # waiting 10 seconds each time there is an error.

    # :param what_to_do:
    # :param print_string:
    # :return:
    # """

#    for error_counter in range(how_many_tries):
#        try:
#           return what_to_do()
#        except:
#            print(
#                f"WARNING: An error occurred for experiment: '{print_string}'.")
#            traceback.print_exc()
#
 #       print(f"Waiting {10}s.")
  #      time.sleep(10)
   #     print(f"Retrying experiment: '{print_string}'.")
#
#        if (error_counter + 1) % 25 == 0:
#            qprint(
#                f"\n\nAlready tried '{print_string}' {str(error_counter)} times! Something seems to be wrong...\n\n",
#                '',
#                'red')