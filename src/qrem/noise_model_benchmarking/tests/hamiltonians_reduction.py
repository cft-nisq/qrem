"""
@author: Filip Maciejewski
@contact: filip.b.maciejewski@gmail.com
"""
# DELETE THIS FILE -- talk with Jan what is inside (not sure if this is a real test)
# This test uses specific data which is test data - should be moved maybe to test folder?
#check if this test is 

import pickle

from qrem.functions_qrem import ancillary_functions as anf, quantum_ancillary_functions as quanf
from qrem.common import io

# specify data used for testing
backend_name = 'ASPEN-9'

date, bitstrings_right_to_left, number_of_qubits = anf.get_historical_experiments_number_of_qubits(
    backend_name=backend_name)

date = '2021-09-27'

module_directory = anf.get_local_storage_directory()

tests_directory_main = module_directory + '/saved_data/characterization2021/'
tests_directory_low = f"{date}/{backend_name}/N{number_of_qubits}/"
directory_to_open = tests_directory_main + 'processed_data/' + tests_directory_low

tests_directory_low = f"ground_states_implementation/2SAT/{date}/N{number_of_qubits}/"
directory_to_open = tests_directory_main + 'processed_data/' + tests_directory_low

with open(directory_to_open + "00_raw_results.pkl", 'rb') as filein:
    experiments_results = pickle.load(filein)

with open(directory_to_open + "full_information_2SAT.pkl", 'rb') as filein:
    hamiltonians_data = pickle.load(filein)

# io.save(dictionary_to_save=hamiltonians_data['N23'],
#                         directory=directory_to_open,
#                         custom_filename="full_information_2SAT_N23")

target_number_of_qubits = 20
better_hamiltonians_data = {}


for hamiltonian_index, ham_dictionary in hamiltonians_data['N20'].items():

    # raise KeyError
    weights_dictionary = ham_dictionary['weights_dictionary']
    ground_state = ham_dictionary['ground_state']

    weights_dictionary_better = {anf.convert_qubits_string_to_tuple(qubits_string=qubits_string):weight
                          for qubits_string,weight in weights_dictionary.items()}

    ideal_energy = quanf.get_energy_from_bitstring_diagonal(bitstring=ground_state,
                                                            weights_dict=weights_dictionary_better)
    ham_dictionary['ground_state_energy'] = ideal_energy
    ham_dictionary['weights_dictionary'] = weights_dictionary_better



    fixed_qubits = list(range(target_number_of_qubits,len(ground_state)))
    ground_state_of_fixed_qubits = [ground_state[x] for x in fixed_qubits]

    string_ground_state_of_fixed_qubits = ''.join(ground_state_of_fixed_qubits)

    ground_state_of_variable_qubits = [ground_state[x] for x in list(range(target_number_of_qubits))]
    string_ground_state_now = ''.join(ground_state_of_variable_qubits)

    # print(ground_state)
    # print(string_ground_state_now)
    # print(fixed_qubits)
    # print(string_ground_state_of_fixed_qubits)
    # raise KeyError

    if string_ground_state_now in better_hamiltonians_data.keys():
        better_hamiltonians_data[string_ground_state_now][string_ground_state_of_fixed_qubits] = ham_dictionary
    else:
        better_hamiltonians_data[string_ground_state_now] = {string_ground_state_of_fixed_qubits: ham_dictionary}



    #
print('saving')
io.save(dictionary_to_save=better_hamiltonians_data,
                        directory=directory_to_open,
                        custom_filename="full_information_2SAT_N20")

    # raise KeyError
#     print(number_of_qubits)
