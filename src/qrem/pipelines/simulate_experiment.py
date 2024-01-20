from typing import Dict, Tuple, List
import time
import numpy as np
from qrem.common import convert 
from qrem.common import io
from qrem.qtypes import CNModelData, ExperimentResults
from qrem.cn import simulation as cnsimulation
from datetime import date
import qrem.common.experiment.tomography as tom
from qrem.common import probability 

def custom_model_creator(number_of_qubits: int, model_specification: list[list[int]], directory_to_save: str, name_id: str = '', save_model: bool = True):
    '''
        Function generates a random noise model for a given number of qubits and specified division into clusters, optionally saves a generated model
        to a file for future use

        Parameters
        ----------

        number_of_qubits: int

        model_specification: list
            list of pairs [[size1, number_of_clusters1],[size2,number_of_clusters_2]...] such that the total of
            size_i*number_of_clusters_i matches the total number of qubits

        directory_to_save: str
            path to save the noise model

        name_id: str
            customary name tag for the file with the noise model

        save_model: bool
            if True the model is saved to a file

        Returns
        ----------
        model: Type[CNModelData]

        Raises
        ----------
        Value Error if the model specification does not match the qubit number
        '''
    count = 0
    for t in model_specification:
        count += t[0] * t[1]
    if (count != number_of_qubits):
        raise ValueError(f"Qubit number mismatch by {count-number_of_qubits}")

    model = cnsimulation.create_random_noise_model(number_of_qubits=number_of_qubits,
                                                         clusters_specification=model_specification)
    model_dict = model.get_dict_format()
    file_name = str(number_of_qubits) + 'qmodel' + name_id
    if save_model:
        io.save(model_dict, directory=directory_to_save, custom_filename=file_name, overwrite=False)

    return model


def intlist_to_string(int_list):
    num_string = ''
    for i in int_list:
        num_string+=str(i)
    return num_string

def simulate_experiment(circuits,number_of_shots,experiment_type="DDOT"):
    """Function takes a collection of input circuits and the number of shots per circuit, returns a dictionary of ideal results with counts per result.
    In the DDOT case the result is always identical to the input state. The "QDOT" case will not be used currently, but may be added.
     
    Parameters:
    ------------
    circuits: List[str]
        List of input states in a form of a list
    number_of_shots: int
        number of repetitions per circuit
    experiment type: str
        "DDOT" for the eigenstates of the computational basis only, "QDOT" sigma_x and sigma_y eigenstates allowed as well (not implemented yet)

    Return:
    results_dict: Dict
        results in a format of {key=input_state: value:dict{result: counts}} 
    """
    if experiment_type=="DDOT":
        result_dict = {}
        for c in circuits:
            bit_string = intlist_to_string(c)
            single_result = {bit_string : number_of_shots}
            if bit_string in result_dict.keys():
                result_dict[bit_string][bit_string]+=number_of_shots
                print("WARNING: repeated circuit")
            else:
                result_dict[bit_string] = single_result
        return result_dict
    else:
        pass


def simulate_noisy_experiment(noise_model: type[CNModelData], number_of_circuits: int, number_of_shots: int, data_directory: str, experiment_type: str = 'DDOT',
                               name_id: str = '', save_data: bool = False,
                              return_ideal_experiment_data: bool = False, new_data_format: bool = True, ground_states_circuits:List[np.array]=None):
    if experiment_type == 'DDOT':
        number_of_qubits = noise_model.number_of_qubits
        qubit_indices = [i for i in range(number_of_qubits)]

        name_string = str(date.today()) + " q" + str(number_of_qubits)+name_id

        circuits = tom.generate_circuits(number_of_qubits=number_of_qubits,imposed_max_random_circuit_count=number_of_circuits)

        if np.any(ground_states_circuits):

            circuits_list = list(circuits)
        
            circuits_list[0] = np.append(circuits[0],ground_states_circuits, axis=0)
            
            circuits_list[1] = circuits[1] +len(ground_states_circuits)

            circuits = tuple(circuits_list)


    
        unprocessed_results = simulate_experiment(circuits=circuits[0],number_of_shots=number_of_shots)
    

        processed_results = convert.convert_counts_overlapping_tomography(counts_dictionary=unprocessed_results,
                                                                  experiment_name='DDOT',reverse_bitstrings=False, old_send_procedure=False)

        file_name_results = experiment_type+'_simulation_workflow_results_' + name_string + '.pkl'
        io.save(dictionary_to_save=processed_results, directory=data_directory, custom_filename=file_name_results)

        print("simulation_workflow saved")
        results_dictionary = {}

      
    
    
        t0=time.time()
        noisy_results_dictionary = cnsimulation.simulate_noise_results_dictionary(processed_results, noise_model)
        t1=time.time()
        print(f"noisy results generated in: {t1-t0} seconds")
        results_dictionary['noisy_results_dictionary'] = noisy_results_dictionary

        if new_data_format:
            new_format_dict = {}
            for input, result in noisy_results_dictionary.items():
                keys = list(result.keys())
                values = list(result.values())

                bool_matrix = np.array([list(map(int, key)) for key in keys], dtype=bool)
                int_vector = np.array(values, dtype=int)
                new_format_dict[input] = (bool_matrix, int_vector)
            results_dictionary['new_data_format']=new_format_dict
        
        if return_ideal_experiment_data:
           
       
            results_dictionary['ideal_results_dictionary'] = probability.compute_marginals_single(results_dictionary=new_format_dict,subsets_list=marginals_to_mitigate,normalization=True)

        if save_data:
            file_name_noisy_results = experiment_type+'_noisy_simulation_results_' + name_string + '.pkl'

            io.save(dictionary_to_save=results_dictionary, directory=data_directory, custom_filename=file_name_noisy_results)

        return results_dictionary

    else:
        raise NameError(experiment_type+" noisy results simulation not implemented yet.")

    
    
