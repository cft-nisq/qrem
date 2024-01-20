import random
from qrem.qtypes import CNModelData as CNModelData
from qrem.qtypes import ExperimentResults
from qrem.cn import simulation as cnsimulation
import qrem.common.experiment.tomography as tom
import qrem.common.io as io
import numpy as np
import time
from datetime import date
from qrem.common.printer import warprint
def simulate_clean_experiment(circuits, number_of_shots, data_directory: str=None, experiment_type="DDOT", save_result=False, name_id=''):
    """Function takes a collection of input circuits and the number of shots per circuit, returns a dictionary of ideal results with counts per result.
    In the DDOT case the result is always identical to the input state. The "QDOT" case will not be used currently, but may be added.

    Parameters:
    ------------
    circuits: List[str]
        List of input states in a form of a list
    number_of_shots: int
        number of repetitions per circuit
    experiment type: str
        "DDOT" for the eigenstates of the computational basis only
        "QDOT" sigma_x and sigma_y eigenstates allowed as well

    Return:
    results_dict: Dict
        results in a format of {key=input_state: value:dict{result: counts}}
    """
    number_of_qubits = len(circuits[0])
    result_dict = {}
    if experiment_type == "DDOT":
        for c in circuits:
            bit_string = ''.join([str(x) for x in c])
            single_result = {bit_string: number_of_shots}
            if bit_string in result_dict.keys():
                result_dict[bit_string][bit_string] += number_of_shots
                #print("WARNING: repeated circuit")
            else:
                result_dict[bit_string] = single_result

    elif experiment_type == "QDOT":
        for c in circuits:
            input_array = [str(x) for x in c]
            input_string = ''.join(input_array)

            for s in range(number_of_shots):
                result_array = [input_array[i]*int(c[i]<2)+str(random.randint(0,1))*int(c[i]>=2) for i in range(len(c))]
                result_string  = ''.join(result_array)
                if input_string in result_dict.keys():
                    if result_string in result_dict[input_string].keys():
                        result_dict[input_string][result_string]+=1
                    else:
                        result_dict[input_string][result_string]=1
                else:
                    result_dict[input_string]={}
                    result_dict[input_string][result_string] = 1




        pass
    else:
        raise ValueError(experiment_type+" not implemented")
    if save_result and data_directory is not None:
        name_string = str(date.today()) + " q" + str(number_of_qubits) + name_id
        file_name_results = experiment_type + '_noiseless_simulation_results_' + name_string + '.pkl'
        io.save(dictionary_to_save=result_dict, directory=data_directory, custom_filename=file_name_results)

        print("noiseless results saved")
    return result_dict

def simulate_noisy_experiment(noise_model: type[CNModelData], number_of_circuits: int, number_of_shots: int,
                                  data_directory: str = '', experiment_type: str = 'DDOT',
                                  name_id: str = '', save_data: bool = False, new_data_format: bool = True, ground_states_circuits: list = [], add_noise: bool = True):
        if experiment_type == 'DDOT':
            number_of_qubits = noise_model.number_of_qubits
            qubit_indices = [i for i in range(number_of_qubits)]

            name_string = str(date.today()) + " q" + str(number_of_qubits) + name_id

            circuits = tom.generate_circuits(number_of_qubits=number_of_qubits,
                                             imposed_max_random_circuit_count=number_of_circuits)
            all_circuits = np.array(list(circuits[0])+list(ground_states_circuits))
            unprocessed_results = simulate_clean_experiment(circuits=all_circuits, number_of_shots=number_of_shots, data_directory=data_directory)


            results_dictionary = {}
            results_dictionary['noiseless_results_dictionary']=unprocessed_results

            new_format_dict = {}
            for input, result in unprocessed_results.items():
                keys = list(result.keys())
                values = list(result.values())

                bool_matrix = np.array([list(map(int, key)) for key in keys], dtype=bool)
                int_vector = np.array(values, dtype=int)
                new_format_dict[input] = (bool_matrix, int_vector)
            results_dictionary['noiseless_new_data_format'] = new_format_dict


            if add_noise:
                t0 = time.time()
                noisy_results_dictionary = cnsimulation.simulate_noise_results_dictionary(unprocessed_results, noise_model)
                t1 = time.time()
                print(f"noisy results generated in: {t1 - t0} seconds")
                results_dictionary['noisy_results_dictionary'] = noisy_results_dictionary

                if new_data_format:
                    new_format_dict = {}
                    for input, result in noisy_results_dictionary.items():
                        keys = list(result.keys())
                        values = list(result.values())

                        bool_matrix = np.array([list(map(int, key)) for key in keys], dtype=bool)
                        int_vector = np.array(values, dtype=int)
                        new_format_dict[input] = (bool_matrix, int_vector)
                    results_dictionary['noisy_new_data_format'] = new_format_dict

                if save_data:
                    file_name_noisy_results = experiment_type + '_noisy_simulation_results_' + name_string + '.pkl'

                    io.save(dictionary_to_save=results_dictionary, directory=data_directory, custom_filename=file_name_noisy_results)

            experiment_result = ExperimentResults()
            experiment_result.counts = new_format_dict
            return experiment_result

        else:
            raise NameError(experiment_type + " noisy results simulation not implemented yet.")




