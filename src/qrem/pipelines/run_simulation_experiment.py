"""4. napisać funkcję run_simulation_experiment.py, która (1) stworzy przykladowy model(e), (2) zwróci nam jakąś przykladową symulację."""
import sys
import qrem
import qrem.providers.simulation as simulate_experiment
from qrem.providers.simutils.simutils import noise_model_from_file, custom_model_creator
from qrem.common.printer import warprint, qprint
from pathlib import Path

CONFIG_PATH = ["--config_file", "C:\\Users\\Kasia\\QREM_SECRET_DEVELOPMENT\\src\\qrem\\common\\default_simulation.ini"]


def run_simulation_experiment(cmd_args=CONFIG_PATH, verbose_log = True):
    """Function which runs the simulation"""
    config = qrem.load_config(cmd_args=cmd_args, verbose_log=verbose_log)
    print(config)
    experiment_type = config.experiment_type
    data_directory = Path(config.experiment_path)
    if not data_directory.is_dir():
        data_directory.mkdir(parents=True, exist_ok=True)

    name_id = config.name_id
    save_data = bool(config.save_data)
    new_data_format = bool(config.new_data_format)
    model_from_file = bool(config.model_from_file)
    add_noise = bool(config.add_noise)

    number_of_circuits=int(config.number_of_circuits)
    number_of_shots=int(config.number_of_shots)
    try:
        number_of_qubits=int(config.number_of_qubits)
    except ValueError:
        number_of_qubits=None

    model_specification=config.model_specification



    if model_from_file:
        noise_model_directory = Path(config.noise_model_directory)
        noise_model_file = config.noise_model_file
        noise_model = noise_model_from_file(data_directory=noise_model_directory, file_name=noise_model_file)
    else:
        if number_of_qubits == None or model_specification == None:
            raise ValueError("Noise model unspecified")
        noise_model = custom_model_creator(number_of_qubits=number_of_qubits, model_specification=model_specification,
                                      name_id=name_id, save_model = True,directory_to_save=data_directory)
    noisy_results = simulate_experiment.simulate_noisy_experiment(data_directory=data_directory,noise_model=noise_model, number_of_circuits=number_of_circuits, experiment_type=experiment_type,
                                                                      number_of_shots=number_of_shots,save_data=save_data,name_id=name_id,new_data_format=new_data_format,add_noise=add_noise)

    return noisy_results



if __name__ == "__main__":
    warprint("=============================")
    warprint("START: Executing SIMULATION Experiment pipeline")
    warprint("=============================")
    warprint("Random noise model created for a given specification: ",color="BLUE")
    argv = sys.argv[1:]
    import time

    t0 = time.time()

    if argv != []:
        run_simulation_experiment(cmd_args=argv)
    else:
        run_simulation_experiment()

    t1 = time.time()
    qprint("=============================")
    qprint("Elapsed time: {} sec".format(t1 - t0))




