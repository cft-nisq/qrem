
from typing import Optional
from qrem.functions_qrem.ancillary_functions import get_local_storage_directory


#MOcomm - we do not seam to use these functions - hence I think the whole file can be deleted

def get_standarized_directory_main_tutorials(directory_QREM):
    return f"{directory_QREM}/Tutorials/data_storage"


def get_standarized_subdirectory_results_tutorials(experiment_name,
                                                    backend_name,
                                                    number_of_qubits,
                                                    shots_per_setting,
                                                    date_name):


    return f"{backend_name}/{experiment_name}/{date_name}/number_of_qubits_{number_of_qubits}/shots_{shots_per_setting}/"


def get_standarized_subdirectory_dot_collections_tutorials(experiment_name,
                                                    number_of_circuits,
                                                    number_of_qubits):


    return f"{experiment_name}/number_of_qubits_{number_of_qubits}/circuits_amount_{number_of_circuits}/"


def get_standarized_subdirectory_hamiltonians_tutorials(hamiltonian_name,
                                                    number_of_qubits):


    return f"{hamiltonian_name}/number_of_qubits_{number_of_qubits}/"




def _get_standard_subfolder_name(backend_name:str,
                              date:str,
                              number_of_qubits:int):


    return f"{backend_name}/{date}/number_of_qubits_{number_of_qubits}/"

def get_directory_circuits_collections(
        locality:int,
        number_of_qubits:int,
        circuits_amount:int,
        experiment_name='DDOT'):

    directory_to_go = get_local_storage_directory() + f"saved_data/circuits_collections/"
    directory_to_go = directory_to_go\
                      +f"{experiment_name}/locality_{locality}/number_of_qubits_{number_of_qubits}/circuits_amount_{circuits_amount}/"
    return directory_to_go





def get_directory_stored_hamiltonians(number_of_qubits:int,
                                      hamiltonian_name:str):

    directory_to_go = get_local_storage_directory() + f"saved_data/stored_hamiltonians/"
    directory_to_go = directory_to_go\
                      +f"{hamiltonian_name}/number_of_qubits_{number_of_qubits}/"
    return directory_to_go

def get_directory_raw_results(backend_name:str,
                              date:str,
                              number_of_qubits:int):

    directory_to_go = get_local_storage_directory() + f"saved_data/raw_experimental_results/"
    directory_to_go = directory_to_go+_get_standard_subfolder_name(backend_name=backend_name,
                                                                   date=date,
                                                                   number_of_qubits=number_of_qubits)

    return directory_to_go


def get_directory_backend_information(backend_name:str,
                              date:str):

    directory_to_go = get_local_storage_directory() + f"saved_data/backend_information/"
    directory_to_go = directory_to_go+f"{backend_name}/{date}/"
    return directory_to_go



def get_directory_processed_experimental_results(backend_name:str,
                                                 date:str,
                                                 number_of_qubits:int,
                                                 experiment_name:str,
                                                 additional_path:Optional[str]=None):


    directory_to_go = get_local_storage_directory()+f"saved_data/processed_experimental_results/{experiment_name}/"
    directory_to_go = directory_to_go+_get_standard_subfolder_name(backend_name=backend_name,
                                                                   date=date,
                                                                   number_of_qubits=number_of_qubits)

    if additional_path is None:
        additional_path = ''

    return directory_to_go+additional_path