import copy
import time
from typing import Optional, List, Dict, Union

import numpy as np
from scipy.special import binom 
from tqdm import tqdm

from qrem.noise_characterization.base_classes.overlapping_estimation_base import OverlappingEstimationBase
from qrem.noise_characterization.tomography_design.overlapping import overlapping_tomography_functions as otf

from qrem.common import convert, math
from qrem.common.printer import qprint

class OverlappingTomographyBase(OverlappingEstimationBase):
    """


    """
    def __init__(self,
                 number_of_qubits: int,
                 experiment_name: str,
              #   maximal_circuits_number: Optional[int] = 1500,  #This can be retreived latter
                 subsets_list: Optional[List[List[int]]] = None, #for consitency I would use subsets for this variable or subset_list -conflict with parent class terminology
                 subsets_locality: Optional[int] = 2,
                 show_progress_bars: Optional[bool] = False
                 ):

        """

        :param number_of_qubits:
        :param number_of_symbols: number of different experimental setups to consider
                                 for example - Diagonal Detector Overlapping Tomography (DDOT)
                                 requires only 2 symbols corresponding to X and identity gates

        :param subsets_list: list of subsets of qubits for which marginal tomography
                             should be performed
                             NOTE: for tomography which is to be performed NOT on ALL subsets
                             of given locality, this must be provided

        :param subsets_locality: locality of ALL considered subsets
                                 NOTE: this should be provided only if ALL subsets have the same
                                       locality


        #TODO: describe rough functionalities and ligh level logic of this class (depending on optional parameters give n)
        """
        # when no subsets are given and no locality of subsets provided return erros 
        if subsets_list is None and subsets_locality is None:
            raise ValueError('Please provide subsets list or desired locality')
        #  when subset_list is empty but subset_locality is given - generate all k element
        elif subsets_list is None and subsets_locality is not None:
            subsets_list = math.get_k_local_subsets(number_of_elements=number_of_qubits, 
                                                         subset_size=subsets_locality)


        # depending on type of experiments use different number of symbols for circuits      
        #ORGANIZATION - nie do konca rozumiem czym QDT rozni sie od QDOT - (pp) z tego co rozuzmiem niczym, po prostu możn użzyć obu nazw
        if experiment_name.upper() == 'QDT' or experiment_name.upper() == 'QDOT':
            number_of_symbols = 6
        elif experiment_name.upper() == 'DDOT':
            number_of_symbols=2

# ORGANIZATION : check logic of the incheritance of this class (MO)
        super().__init__(number_of_qubits=number_of_qubits,
                         number_of_symbols=number_of_symbols,
                         subsets=subsets_list,
                  #       maximal_circuits_amount=maximal_circuits_number, #this can be retreived latter
                         show_progress_bars=show_progress_bars

                         )

#asignement of subset locality variable to local variant 
        self._subsets_locality = subsets_locality
        if subsets_locality is not None:


#ORGANIZATION: variable _number_of_elements_collection is an upper bound on the number of elements needed for "perfect tomography"
# This variable is used exclusivelly in internal function _compute_perfect_collection_bruteforce. Hence this is a candidate to removal (MO)

            self._elements_in_perfect_collection = int(
                binom(self._number_of_qubits,self._subsets_locality) *
                self._number_of_symbols ** self._subsets_locality)
        else:
            self._elements_in_perfect_collection = int(sum(
                [number_of_symbols ** len(subset_now) for subset_now in subsets_list]))

#------------------------------

        '''
        I changed subset_list to subset for consistency in the comprehension below (MO)
        Dictionary below has keys that label qubit tuples (in the format q0q1 etc. ), for each key the value is an array
        storing occurrences of arrangements of number_of_symbols among  |size if subset| slots (there are number_of_symbols ** len(subset)) of them
        This structure is presumably used to test the existence of families of perfect functions

        dtape=int is there to indicate integer occurrences 

        '''

        #MOcomm - is it the best thing to do alghorimically? perhaps other data structures are faster? 
        # For example -kays can be just given by subsets and (perhaps) it would be faster to iterate over them

        #deleted redundant "not private" version of this dictionary
        self._dictionary_symbols_counting = {
            convert.qubit_indices_to_keystring(subset): np.zeros(self._number_of_symbols ** len(subset),
                                                       dtype=int) for 
            subset in self._subsets}


#MOcomm I constructed properties dictionary in one line 
#MOcomm - not sure what is the meaning of this dictionary and these parameters
        self._circuits_properties_dictionary = {'number_of_circuits': 0,
                                          'absent_elements_amount': 10 ** 6,
                                          'minimal_amount': 10 ** 6,
                                          'maximal_amount': 10 ** 6,
                                          'median': 10 ** 6,
                                          'mean': 10 ** 6,
                                          'SD': 10 ** 6}


        #MOcomm - not sure what is the meaning of this
        if number_of_symbols < 11:
            self._integer_representation = lambda integer: str(integer)
        else:
            self._integer_representation = lambda integer: self.integer_to_charstring(integer)

    @property
    def dictionary_symbols_counting(self) -> Dict[str, np.ndarray]:
        return self._dictionary_symbols_counting

    @property
    def circuits_properties_dictionary(self) -> Dict[str, float]:
        return self._circuits_properties_dictionary

    @property
    def circuits_list(self) -> List[List[int]]:
        return self._circuits_list

    @circuits_list.setter
    def circuits_list(self, circuits_list: List[List[int]]):
        self.reset_everything()
        self._circuits_list = circuits_list


#ORGANIZE- in my opinion there is no need to have aditional dictionary_symbols_counting dicrtionary
#We can add it to begin with to private dict (MO)

    def add_dictionary_subsets_symbols_counting_template(self):
        dictionary_symbols_counting = {
            convert.qubit_indices_to_keystring(subset_list): np.zeros(self._number_of_symbols ** len(subset_list),
                                                       dtype=int) for
            subset_list in self._subsets}

        self._dictionary_symbols_counting = dictionary_symbols_counting

    #MOVE_TO >> core.utils 
    def integer_to_charstring(integer: int):
        if integer < 10:
            return str(integer)
        else:
            return chr(integer + 97 - 10)

# MOcomm - not clear the meaning of it
    def update_dictionary_subsets_symbols_counting(self,
                                                   circuits_list: Optional[List[List[Union[int,str]]]] = None,
                                                   count_added_subcircuits: Optional[bool] = False):
        if circuits_list is None:
            circuits_list = self._circuits_list

        subsets_range = range(len(self._subsets))
        if self._show_progress_bars:
            subsets_range = tqdm(subsets_range)

        if count_added_subcircuits:
            added_subcircuits_counter = 0
        else:
            added_subcircuits_counter = None

        for subset_index in subsets_range:
            for circuit in circuits_list:
                qubits_key = convert.qubit_indices_to_keystring(self._subsets[subset_index])

                subset_symbols_now = [circuit[qubit_index] for qubit_index in self._subsets[subset_index]]

                # print(subset_symbols_now, any(s > self._number_of_symbols for s in subset_symbols_now))

                if any(s >= self._number_of_symbols for s in subset_symbols_now):
                    pass
                else:
                    subset_circuit_identifier = int(
                        ''.join([self._integer_representation(s) for s in subset_symbols_now]),
                        self._number_of_symbols)

                    if count_added_subcircuits:
                        if self._dictionary_symbols_counting[qubits_key][subset_circuit_identifier] == 0:
                            added_subcircuits_counter += 1

                    self._dictionary_symbols_counting[qubits_key][subset_circuit_identifier] += 1

        if count_added_subcircuits:
            return added_subcircuits_counter

    def get_absent_symbols_amount(self):

        t0 = time.time()
        zero_subsets = 0
        for subset in self._subsets:
            subset_counts = self._dictionary_symbols_counting[convert.qubit_indices_to_keystring(subset)]
            zero_subsets += len(subset_counts) - np.count_nonzero(subset_counts)

        self._circuits_properties_dictionary['absent_elements_amount'] = zero_subsets
        qprint('This took:', time.time() - t0)

        return zero_subsets
    #
    def get_absent_symbol_indices(self):
        zero_subsets = {}
        for subset in tqdm(self._subsets):
            subset_counts = self._dictionary_symbols_counting[convert.qubit_indices_to_keystring(subset)]
            stuff_now = []
            for index in range(len(subset_counts)):
                if subset_counts[index]==0:
                    stuff_now.append(index)
            zero_subsets[subset] = stuff_now

        return zero_subsets

    def calculate_properties_of_circuits(self) -> None:

        big_list = []
        for subset in self._subsets:
            big_list += list(self._dictionary_symbols_counting[convert.qubit_indices_to_keystring(subset)])

        minimal_amount, maximal_amount = min(big_list), max(big_list)

        big_list = np.array(big_list)

        mean, SD, median = np.mean(big_list), np.std(big_list), np.median(big_list)

        absent_elements_amount = len(big_list) - np.count_nonzero(big_list)

        self._circuits_properties_dictionary['number_of_circuits'] = len(self._circuits_list)
        self._circuits_properties_dictionary['absent_elements_amount'] = absent_elements_amount
        self._circuits_properties_dictionary['minimal_amount'] = minimal_amount
        self._circuits_properties_dictionary['maximal_amount'] = maximal_amount
        self._circuits_properties_dictionary['median'] = median
        self._circuits_properties_dictionary['mean'] = mean
        self._circuits_properties_dictionary['SD'] = SD


    #MOcomm a function reseting everything in an object
    def reset_everything(self):
        #MOcomment - I constructed private dictionary with properties in one line now
        self._circuits_properties_dictionary= {'number_of_circuits': 0,
                                          'absent_elements_amount': 10 ** 6,
                                          'minimal_amount': 10 ** 6,
                                          'maximal_amount': 10 ** 6,
                                          'median': 10 ** 6,
                                          'mean': 10 ** 6,
                                          'SD': 10 ** 6}

        self._circuits_list = []

#changed argument in this comprehension
        dictionary_symbols_counting = {
            convert.qubit_indices_to_keystring(subset): np.zeros(self._number_of_symbols ** len(subset),
                                                       dtype=int) for
            subset in self._subsets}

        self._dictionary_symbols_counting = dictionary_symbols_counting

    def _cost_function_circuits_amount(self):
        return len(self._circuits_list)

    def _cost_function_circuits_SD(self):
        return self._circuits_properties_dictionary['SD']

    def _cost_function_minimal_amount_of_circuits(self):
        return -self._circuits_properties_dictionary['minimal_amount']

    def _cost_function_absent_elements(self):
        return self._circuits_properties_dictionary['absent_elements_amount']

    def _cost_function_maximal_spread(self):
        return self._circuits_properties_dictionary['maximal_amount'] - \
               self._circuits_properties_dictionary['minimal_amount']




    def __compute_1q_collection(self):

        circuits_now = [[unique_symbol for _ in range(self._number_of_qubits)] for unique_symbol in range(self._number_of_symbols)]
        self.add_circuits(circuits_now)
        self.update_dictionary_subsets_symbols_counting(
            circuits_list=circuits_now,
            count_added_subcircuits=True)

# ORGANIZE - after discussion with Janek we decided that this function should be updated,
#  now it is highly nonoptimal, if not wrong (MO)
    def _compute_perfect_collection_bruteforce(self,
                                               circuits_in_batch: int,
                                               print_updates: bool
                                               ):

        runs_number = 1
        absent_elements_amount = self._elements_in_perfect_collection

        while absent_elements_amount > 0 and len(self.circuits_list) < self._maximal_circuits_amount:
            if runs_number % 20 == 0 and print_updates:
                qprint('Run number:', runs_number)
                qprint('Number of circuits:', len(self._circuits_list))
                qprint('Absent elements amount:', absent_elements_amount)

            circuits_now = self.get_random_circuits_list(circuits_in_batch)
            self.add_circuits(circuits_now)

            added_elements = self.update_dictionary_subsets_symbols_counting(
                circuits_list=circuits_now,
                count_added_subcircuits=True)
            absent_elements_amount -= added_elements

            runs_number += 1




    def _get_proper_cost_function(self,
                                  optimized_quantity: str):

        if optimized_quantity in ['circuits_amount', 'circuits_number', 'amount', 'circuits']:
            cost_function = self._cost_function_circuits_amount
        elif optimized_quantity in ['std', 'SD', 'standard_deviation']:
            cost_function = self._cost_function_circuits_SD
        elif optimized_quantity in ['minimal_amount']:
            cost_function = self._cost_function_minimal_amount_of_circuits
        elif optimized_quantity in ['spread', 'maximal_spread']:
            cost_function = self._cost_function_maximal_spread
        elif optimized_quantity in ['absent', 'absent_elements']:
            cost_function = self._cost_function_absent_elements
        else:
            raise ValueError('Wrong optimized quantity string: ' + optimized_quantity + '.')

        return cost_function

    def _add_cost_functions(self,
                            dictionary_cost_functions: Dict[str, float]):

        def cost_functions_added():
            returned_quantity = 0
            for function_label, function_weight in dictionary_cost_functions.items():
                returned_quantity += function_weight * self._get_proper_cost_function(function_label)()
            return returned_quantity

        return cost_functions_added

    def _compute_perfect_collection_bruteforce_randomized(self,
                                                          number_of_iterations: int,
                                                          circuits_in_batch: int,
                                                          print_updates: bool,
                                                          optimized_quantity: Union[
                                                              str, Dict[str, float]],
                                                          additional_circuits: Optional[int] = 0,
                                                          fixed_circuits:Optional[List[List[int]]]=None
                                                          ):
        """
        This function implements self._compute_perfect_collection_bruteforce
        for number_of_iterations times, then adds additional_circuits number of random circuits,
        computes cost function and chooses the family that minimizes cost function.

        :param number_of_iterations: how many times random perfect family should be generated
        :param circuits_in_batch: see self._compute_perfect_collection_bruteforce
        :param print_updates: whether to print updates during optimization
        :param optimized_quantity: specify what cost function is
        Possible string values:
        1. 'minimal_amount' - maximizes number of least-frequent subset-circuits
        2. 'spread' - minimizes difference between maximal and minimal number of subset-circuits

        3. 'circuits_amount' - minimizes number of circuits
                        (NOTE: it does not make sense to choose this option with additional_circuits>0)
        4. 'SD' - minimizes standard deviation of occurrences of subset-circuits

        It is possible to use combined cost functions_qrem.
        Dictionary must be provided where KEY is label for optimized quantity and VALUE is its weight.

        For example:
        optimized_quantity = {'minimal_amount': 1.0,
                              'spread':0.5}

        will give cost function which returns 1.0 * (-number of least frequent circuits)
                                          + 0.5 * (difference between most and least fequenet circuits)


        :param additional_circuits: number of circuits which are to be added to the PERFECT collection
                                    obtained in optimization loop. Those are "additional" circuits in
                                    a sense that they are not needed for collection to be perfect,
                                    but instead are used to obtain better values of cost function
                                    or just add more experiments reduce statistical noise.
        :return:
        """

        if isinstance(optimized_quantity, str):
            cost_function = self._get_proper_cost_function(optimized_quantity=optimized_quantity)
        elif isinstance(optimized_quantity, dict):
            cost_function = self._add_cost_functions(dictionary_cost_functions=optimized_quantity)
        else:
            raise ValueError("No cost function was specified!")


        runs_range = range(number_of_iterations)
        if self._show_progress_bars:
            runs_range = tqdm(runs_range)

        # circuit_families = []
        # best_family = None
        global_cost, best_family = 10 ** 6, None

        for runs_number in runs_range:
            if runs_number % int(np.ceil(number_of_iterations / 20)) == 0 and print_updates:
                qprint('Run number:', runs_number, 'red')
                qprint('Current best value of cost function:', global_cost)

            self.reset_everything()

            if fixed_circuits is not None:
                self.add_circuits(fixed_circuits)
                self.update_dictionary_subsets_symbols_counting()



            self._compute_perfect_collection_bruteforce(circuits_in_batch=circuits_in_batch,
                                                        print_updates=False
                                                        )

            if additional_circuits > 0:
                current_length, maximal_length = len(
                    self._circuits_list), self._maximal_circuits_amount
                if current_length < maximal_length:
                    if additional_circuits > maximal_length - current_length:
                        adding_circuits = maximal_length - current_length
                    else:
                        adding_circuits = copy.deepcopy(additional_circuits)
                    # print(adding_circuits)
                    new_circuits = self.get_random_circuits_list(adding_circuits)

                    self.update_dictionary_subsets_symbols_counting(new_circuits)
                    self.add_circuits(new_circuits)

            self.calculate_properties_of_circuits()

            cost_now = cost_function()

            if cost_now < global_cost:
                best_family = copy.deepcopy(self._circuits_list)
                global_cost = cost_now

        qprint('best family length', len(best_family), 'red')
        self.reset_everything()
        self._circuits_list = best_family
        self.update_dictionary_subsets_symbols_counting()
        self.calculate_properties_of_circuits()

    def _add_absent_elements_by_hand(self):

        best_family = copy.deepcopy(self._circuits_list)

        missing_elements = self.get_absent_symbol_indices()

        number_of_added_circuits_multiplier = 3

        added_circuits = []
        for subset, circuit_indices_list in tqdm(missing_elements.items()):
            register_local = dict(enumerate(convert.get_ditstrings_register(base=self._number_of_symbols,
                                                         number_of_dits=len(subset))))

            for _ in range(number_of_added_circuits_multiplier):
                for circuit_index in circuit_indices_list:
                    circuit_now_global = self.get_random_circuit()

                    local_circuit_now = register_local[circuit_index]

                    for symbol_index in range(len(subset)):
                        circuit_now_global[subset[symbol_index]] = local_circuit_now[symbol_index]
                    added_circuits.append(circuit_now_global)


        # self.reset_everything()
        # print(len(added_circuits))
        self._show_progress_bars = True
        self.update_dictionary_subsets_symbols_counting(circuits_list=added_circuits)
        self._circuits_list = best_family + added_circuits
        self.calculate_properties_of_circuits()

    #(PP) same as in base class?
    def compute_random_collection(self,
                                  number_of_circuits=None):

        if number_of_circuits is None:
            #FBM: add probabilistic bounds
            raise ValueError()


        self._circuits_list = self.add_random_circuits(number_of_circuits)

        return self._circuits_list


    # (PP) not used as of now, commenting 
    # (PP) TODO_MO - what is the usecase for this function?
    # def optimize_perfect_collection(self,
    #                                 method_name='bruteforce_randomized',
    #                                 method_kwargs=None):

    #     """
    #     Find perfect collection of overlapping circuits.
    #     "Perfect" means that for each subset of qubits self._subsets_list[i],
    #      each symbol out of self._number_of_symbols^self._subsets_list[i],
    #      appears in the collection at least once.

    #     :param method_name:
    #     possible values:

    #     1. 'bruteforce' - see self._compute_perfect_collection_bruteforce
    #     2. 'bruteforce_randomized' - see self._compute_perfect_collection_bruteforce_randomized

    #     :param method_kwargs: kwargs for chosen method, see corresponding methods' descriptions
    #     :return:
    #     """

    #     if self._subsets_locality == 1:
    #         self.__compute_1q_collection()

    #     else:

    #         if method_name == 'bruteforce':
    #             if method_kwargs is None:
    #                 method_kwargs = {'circuits_in_batch': 1,
    #                                  'print_updates': True}

    #             self._compute_perfect_collection_bruteforce(**method_kwargs)

    #         elif method_name == 'bruteforce_randomized':
    #             if method_kwargs is None:
    #                 method_kwargs = {'number_of_iterations': 100,
    #                                  'circuits_in_batch': 1,
    #                                  'print_updates': True,
    #                                  'optimized_quantity': 'minimal_amount',
    #                                  'additional_circuits': 0,
    #                                  'fixed_circuits':None}

    #             self._compute_perfect_collection_bruteforce_randomized(**method_kwargs)


    #         absent_elements = self._circuits_properties_dictionary['absent_elements_amount']

    #         if absent_elements != 0:
    #             qprint('________WARNING________:',
    #                            'The collection is not perfect. '
    #                            'It is missing %s' % absent_elements
    #                            + ' elements!\nTry increasing limit on the circuits amount.',
    #                            'red')

    #             if anf.query_yes_no("Do you wish to add missing elements by hand?"):
    #                 self._add_absent_elements_by_hand()

    #         absent_elements = self._circuits_properties_dictionary['absent_elements_amount']

    #         if absent_elements != 0:
    #             qprint('________WARNING________:',
    #                            'The collection is not perfect. '
    #                            'It is missing %s' % absent_elements
    #                            + ' elements!\nTry increasing limit on the circuits amount.',
    #                            'red')

    #             if anf.query_yes_no("Do you wish to add missing elements by hand?"):
    #                 self._add_absent_elements_by_hand()

    def get_parallel_tomography_on_subsets(self,
                                           number_of_circuits: int,
                                           non_overlapping_subsets:List[List[int]]=None,
                                           ) -> List[List[int]]:
        """

        :param number_of_circuits: should be power of 2
        :param subsets:
        :param number_of_symbols:
        :return:
        :rtype:
        """

        if non_overlapping_subsets is None:
            non_overlapping_subsets = self._subsets

        number_of_symbols = self._number_of_symbols

        return otf.get_parallel_tomography_on_non_overlapping_subsets(
            number_of_circuits=number_of_circuits,
            number_of_symbols=number_of_symbols,
            non_overlapping_subsets=non_overlapping_subsets)
