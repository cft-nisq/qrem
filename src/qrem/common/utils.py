"""qrem.common.utils module contains all helpful functions used throughout the projcets.

Current contents of this module will relate to:
- dit/bitstring conversions
- python/numpy format conversions
- boolean operators on lists (that treat lists as sets)
- other helpful functions not covered elswhere

May be split into other modules in the future (very likely)

"""
import sys
import inspect
import warnings
import multiprocessing
from typing import List, Dict, Iterable, Optional, Callable

import numpy as np

from qrem.common.printer import qprint



#===================================================
# Boolean operation on lists, dicts and  sets
#===================================================

def lists_intersection(lst1: list,
                       lst2: list):
    """Intersection of 2 lists lst1 and lst2 after conversion of both to sets.
    
    Returns
    -------
    list

    Notes
    -----
    Multiple entries within each of the lists will merge 
    """
    return list(set(lst1) & set(lst2))

def lists_difference(lst1: list,
                     lst2: list):
    """Difference between 2 lists lst1 and lst2 after conversion of both to sets.
    
    Returns
    -------
    list
    
    Notes
    -----
    Multiple entries within each of the lists will merge 
    """
    return list(set(lst1) - set(lst2))

def lists_sum(lst1: list,
              lst2: list):
    """Union between 2 lists lst1 and lst2 after conversion of both to sets. 

    Returns
    -------
    list
    
    Notes
    -----
    Multiple entries within each of the lists will merge 
    """
    return list(set(lst1).union(set(lst2)))

def lists_sum_multiple(lists: List[list]):
    """Union between all lists in the input  list of lists (lists) after conversion to sets. 
    
    Returns
    -------
    list

    Notes
    -----
    Multiple entries within each of the lists will merge 
    """
    return list(set().union(*lists))

def lists_intersection_multiple(lists: List[list]):
    """Intersection of all lists in the input  list of lists (lists) after conversion  to sets. Returns a list.

    Notes
    -----
    Multiple entries within each of the lists will merge 
    """
    l0 = lists[0]
    l1 = lists[1]

    int_list = lists_intersection(l0, l1)
    for l in lists[2:]:
        int_list = lists_intersection(int_list, l)
    return int_list

#===================================================
# Checks and operations on lists, dicts etc
#===================================================
#TEST_ME test and improve efficiency of the check
def check_for_multiple_occurences(lists: List[list]) -> bool:    
    """Checks for duplicate elements between multiple lists. Multiple occurences within each of lists are not taken into consideration

    Returns
    -------
    bool
    """
    # possible reimplementation, not sure if faster / slower
    # u = reduce(set.union, map(set, ll))
    # sd = reduce(set.symmetric_difference, map(set, lists))
    # len(u - sd) != 0 ->there are repeating elements
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            if len(lists_intersection(lists[i], lists[j])) != 0:
                return True

    return False

def enumerate_dict(list: List[int])->Dict:
    ''' This function takes in a list 'list' and returns a dictionary where the keys are the indices of the elements in 'some_list' 
    and the values are the elements themselves.

    Parameters
    ----------
    list: List[int]
        list to be enumerated
    '''
    return dict(enumerate(list))

def sort_things(stuff_to_sort: Iterable, according_to_what: Iterable) -> Iterable:
    """Sort stuff according to some other stuff assuming that the stuff_to_sort 
    is sorted in natural order (0, 1, 2, ...)"""
    return [element for _, element in sorted(zip(according_to_what, stuff_to_sort), key=lambda pair: pair[0])]

def swap_keys_and_values(enumerated_dict: Dict[int, int]) -> Dict[int, int]:
    """
   This function takes in a dictionary 'enumerate' where the keys are integers and the values are also integers. 
   It returns a new dictionary where the keys and values are reversed from the input dictionary

    Parameters
    ----------
    enumerated_dict: Dict[int, int]

    """
    reversed_map = {}
    for index_sorted, true_index in enumerated_dict.items():
        if true_index in reversed_map:
            raise warnings.warn(f"# WARNING # Value in the list repeats itself. Would silently merge and discard this element. Maybe should be like that?")
        reversed_map[true_index] = index_sorted
    return reversed_map

# formerly get_reversed_enumerated_from_indices
def map_index_to_order(indices: List[int]) -> Dict[str, int]:
    """
    Given indices list, enumerate them and return map which is inverse of enumerate
    Parameters
    ----------
    indices: List[int]
        list of indices (e.g. qubit indices), that will be mapped in a dict to their natural order 
    """
    return swap_keys_and_values(enumerate_dict(indices))

#===================================================
# Helpful, programming funtionts
#===================================================

def wrapped_multiprocessing_function(tuple_of_main_arguments: List[tuple],
                                     additional_kwargs: dict,
                                     function_to_multiprocess: Callable,
                                     number_of_threads: Optional[int] = None,
                                     # debug=False,
                                     printing=False):
    """Wrapper for executing a custom defined function using multiprocessing over a list of variable arguments. 
    Additional list of constant arguments can be provided 

    Parameters
    ----------
    tuple_of_main_arguments: List[tuple]
        This is list of tuples that is divided into batches that are passed to different threads of
        multiprocessing. This, therefore should be set of VARIABLE arguments that are passed to the
        function of interest

    additional_kwargs: dict
        This is dictionary of arguments that are CONSTANT for all function evaluations.
        The dictionary is COPIED for each thread and passed to the function.

    function_to_multiprocess: Callable
    number_of_threads: int
    """

    if number_of_threads is None:
        number_of_threads = int(
            np.min([multiprocessing.cpu_count() - 1, len(tuple_of_main_arguments)]))

    length = len(tuple_of_main_arguments)
    # print(length)
    # raise KeyboardInterrupt
    # IF there is less arguments than threads, we reduce number of threads
    if length < number_of_threads:
        number_of_threads = length


    division_cores = length // number_of_threads

    all_indices, arguments = [], []
    for process_pool_index in range(number_of_threads):
        if process_pool_index == number_of_threads - 1:
            slice_now = slice(process_pool_index * division_cores,
                              -1)
        else:
            slice_now = slice(process_pool_index * division_cores,
                              (process_pool_index + 1) * division_cores)
        sample_indices_now = tuple_of_main_arguments[slice_now]
        arguments.append((sample_indices_now,
                          additional_kwargs))

    if printing:
        qprint(f'Running {number_of_threads} threads!', '', 'green')

    pool = multiprocessing.Pool(number_of_threads)
    results = pool.starmap_async(function_to_multiprocess,
                                 arguments)
    pool.close()
    pool.join()

    res_multiprocessing = results.get()

    results_dictionary_all = {}

    for dict_now in res_multiprocessing:
        results_dictionary_all = {**results_dictionary_all, **dict_now}

    return results_dictionary_all 

def get_full_object_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_full_object_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_full_object_size(v, seen) for v in obj.values()))
        size += sum((get_full_object_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum((get_full_object_size(i, seen) for i in obj))
        except TypeError:
            print("Unable to get size of %r. This may lead to incorrect sizes. Please report this error.", obj)
    if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
        size += sum(get_full_object_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))
        
    return size

# import numpy as np
# import orjson
# from pathlib import Path
# class ExampleDataClass:
#     def __init__(self, name):

#         #[X] default value definitions
#         self.name = name                # name of the Data set
#         dt = np.dtype(np.uint8)         # type definition - helper def for below
#         self.circuits = np.array([],dt) # some data type (for circuits it should be probably list of np.arrays or array of arrays)
#         self.id = -1                    # unique id, set manually during export (field not used currently)
#         self.experiment_type = "DDOT"   # type of experiment - can be coded as string, or as sth else
#         pass

#     def getDictFormat(self):
#         '''returns this class as a json-like dictionary structure'''
#         return self.__dict__

#     def getJSON(self):
#         return  orjson.dumps(self, default=lambda o: o.__dict__,
#         option=orjson.OPT_SERIALIZE_NUMPY, sort_keys=True)
    
#     def exportJSON(self,json_export_path,overwrite = True):
        
#         #[7] Save into json file
#         if(Path(json_export_path).is_file() and not overwrite):
#             print(f"WARNING:: Ommiting export to existing file: <{json_export_path}>")     
#         else:
#             with open(json_export_path, 'w') as outfile:
#                 outfile.write(self.getJSON())

#     def importJSON(self,json_import_path):
#         '''import JSON'''
#         #stub, easy to implement
#         #first orjson.load contents of the file
#         #then assign elements of loaded dict into the class fields
#         pass
