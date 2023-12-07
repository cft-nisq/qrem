"""4. napisać funkcję run_simulation_experiment.py, która (1) stworzy przykladowy model(e), (2) zwróci nam jakąś przykladową symulację."""
import qrem.pipelines.simulate_experiment as simulate_experiment
from qrem.qtypes import CNModelData as CNModelData
import qrem.qtypes.cn_noise_model as cn
from qrem.common.printer import qprint, errprint, warprint
import pickle

DATA_DIRECTORY= r'C:\\Users\\Kasia\\simtest\\'
FILE_NAME_NOISE_MODEL ="10qmodel1st.pkl"

#Optional: noise model dictionary from a file, with the following function:
def noise_model_from_file(data_directory: str, file_name: str):
    with open(data_directory+file_name, 'rb') as filein:
        noise_model_dictionary = pickle.load(filein)

    noise_model = cn.CNModelData(number_of_qubits=noise_model_dictionary['number_of_qubits'])
    noise_matrices = {}
    for key, value in noise_model_dictionary['noise_matrices'].items():
        noise_matrices[key]=value['averaged']

    noise_model.set_noise_model(noise_matrices)
    return noise_model

def run_simulation_experiment(number_of_circuits: int, number_of_shots: int,
                               data_directory: str, name_id: str = '', experiment_type: str = 'DDOT', number_of_qubits: int=None, model_specification: list[list[int]] = None,
                              noise_model: type[CNModelData] = None, save_data: bool = True, new_data_format = True):
    """Function which runs the simulation"""

    if noise_model==None:
        if number_of_qubits == None or model_specification == None:
            raise ValueError("Noise model unspecified")
        noise_model = simulate_experiment.custom_model_creator(number_of_qubits=number_of_qubits, model_specification=model_specification,
                                      name_id=name_id, save_model = True,directory_to_save=data_directory)
    noisy_results = simulate_experiment.simulate_noisy_experiment(data_directory=data_directory,noise_model=noise_model, number_of_circuits=number_of_circuits,
                                                                      number_of_shots=number_of_shots,save_data=save_data,name_id=name_id,new_data_format=new_data_format)

    return noisy_results

DATA_DIRECTORY= r'C:\\Users\\Kasia\\simtest\\'
FILE_NAME_NOISE_MODEL ="10qmodel1st.pkl"

number_of_qubits = 10
model_specification = [[2,5]]
number_of_circuits = 20
number_of_shots = 10**4

if __name__ == "__main__":
    warprint("=============================")
    warprint("START: Executing SIMULATION Experiment pipeline")
    warprint("=============================")
    warprint("Random noise model created for a given specification: ",color="BLUE")
    run_simulation_experiment(number_of_qubits=number_of_qubits,model_specification=model_specification,
                              number_of_circuits=number_of_circuits,number_of_shots=number_of_shots,
                              name_id="1st",data_directory=DATA_DIRECTORY)
    warprint("Second simulation based on the same noise model, now taken from a file: ", color="BLUE")
    model_from_file = noise_model_from_file(DATA_DIRECTORY,FILE_NAME_NOISE_MODEL)
    run_simulation_experiment(noise_model=model_from_file, number_of_circuits=number_of_circuits,number_of_shots=number_of_shots,name_id="2st",
                              data_directory=DATA_DIRECTORY)



