"""Run IBM Experiment pipeline

This module allows to run a characterization/mitigation/benchmark experiment 
on a chosen IBM quantum machine. It uses a config file, which example can be found in 
<configs/ default_ibm.ini> file. Read the config to understand avialable parameters.


 documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

Example
-------
You can run experiment by (i) providing a given config via -C argument, for example

    $ python run_ibm_experiment.py -C \\path\\to\\experiment\\config.ini

    
Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""

import os, sys
import qrem
from pathlib import Path
import orjson

from qrem.common.io import date_time_formatted
from qrem.common.printer import qprint, errprint, warprint
from qrem.common.experiment import tomography as tomography
from qrem.qtypes import CircuitCollection
from qrem.common.constants import CIRCUIT_DATA_TYPE as c_type
from qrem.providers import ibm   #PP: change to qrem.ibm

# ----------------------------------------------------------------
# this are the parameters of running qrem in future / maybe move to config[GENERAL]?:
# you will be able to run it asa command
# ----------------------------------------------------------------
CONFIG_PATH = ["--config_file", "C:\\CFT Chmura\\Theory of Quantum Computation\\QREM_Data\\ibm\\experiment_results\\first_experiment\\first_experiment_config.ini"]

def run(cmd_args=CONFIG_PATH, verbose_log = True): 
    # ----------------------------------------------------------------
    #[1] We load an already defined default config
    # ----------------------------------------------------------------
    config = qrem.load_config(cmd_args = cmd_args, verbose_log = verbose_log)

    EXPERIMENT_NAME = config.experiment_name

    EXPERIMENT_FOLDER_PATH = Path(config.experiment_path)
    if not EXPERIMENT_FOLDER_PATH.is_dir():
        EXPERIMENT_FOLDER_PATH.mkdir( parents=True, exist_ok=True )

    BACKUP_CIRCUITS = config.backup_circuits
    BACKUP_JOB_IDs = config.backup_job_ids
    BACKUP_CIRCUITS_METADATA = config.backup_circuits_metadata
    CONNECTION_METHOD = config.ibm_connection_method
    JOB_TAGS = list(config.job_tags)

    # ----------------------------------------------------------------
    #[2] Get info from provider about valid qubits
    # ----------------------------------------------------------------
    backend, service, provider = ibm.connect(name = config.device_name, method = CONNECTION_METHOD, verbose_log = config.verbose_log)
    valid_qubit_properties = ibm.get_valid_qubits_properties(backend, config.gate_threshold)

    METADATA = {}
    METADATA["date"] = date_time_formatted()
    if BACKUP_CIRCUITS_METADATA:
        METADATA["valid_qubit_properties"] = valid_qubit_properties

    number_of_qubits = valid_qubit_properties["number_of_good_qubits"]
    good_qubits_indices = valid_qubit_properties["good_qubits_indices"]

    # ----------------------------------------------------------------
    #[3] TODO (PP) check if there are specific qubits subset defined in config,. if yes - check against good_qubits_indices and choose only the subset
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    #[4] Generate circuits
    # ----------------------------------------------------------------
    #[4.0] Setup circuit collection object
    qrem_circuit_collection = CircuitCollection()
    qrem_circuit_collection.experiment_name = EXPERIMENT_NAME

    qrem_circuit_collection.load_config(config=config)
    qrem_circuit_collection.device = config.device_name
    qrem_circuit_collection.qubit_indices = good_qubits_indices
    qrem_circuit_collection.metadata = METADATA

    #[4.1] TODO (PP) finish implementation of generate_circuits, so you know how many shots per circuit you need
    qrem_circuit_collection.circuits, _, theoretical_total_circuit_count, theoretical_number_of_shots = tomography.generate_circuits(   number_of_qubits = number_of_qubits,
                                                                            experiment_type = config.experiment_type, 
                                                                            k_locality = config.k_locality,
                                                                            limited_circuit_randomness = config.limited_circuit_randomness,
                                                                            imposed_max_random_circuit_count = config.random_circuits_count,
                                                                            imposed_max_number_of_shots = config.shots_per_circuit)
    
    # - TODO (PP) save final number of shots to collection
    # qrem_circuit_collection.no_shots = ?

    #[5] At this stage, circuits are ready, we can save them to file
    if BACKUP_CIRCUITS:    
        qrem_circuit_collection.export_json(str(EXPERIMENT_FOLDER_PATH.joinpath("input_circuit_collection.json")), overwrite = True)
    else:
        warprint("WARNING: Circuits were not saved to file, as BACKUP_CIRCUITS = False. It is recommended to save circuits to file for future reference.")
    #[6] Now we need to use ibm module to prepare circuits in ibm format
    ibm_circuits = ibm.translate_circuits_to_qiskit_format(qrem_circuit_collection)



    #[6] Now we need to run circuits
    job_ids = ibm.execute_circuits( qiskit_circuits= ibm_circuits,
                                    job_tags = JOB_TAGS,
                                    number_of_repetitions = config.shots_per_circuit,
                                    instance = config.provider_instance,
                                    service = service,
                                    backend = backend,
                                    method = CONNECTION_METHOD,
                                    log_level='INFO',
                                    verbose_log=True)
    
    #[6.1] Backup jobs to circuit collection file
    if BACKUP_CIRCUITS:    
        qrem_circuit_collection.job_IDs = job_ids
        qrem_circuit_collection.export_json(str(EXPERIMENT_FOLDER_PATH.joinpath("input_circuit_collection.json")), overwrite = True)
        
    #[6.2] Save job ids to a file
    if BACKUP_JOB_IDs:     
        json_job_ids=orjson.dumps(job_ids)
        with open(str(EXPERIMENT_FOLDER_PATH.joinpath("input_circuit_collection.json")), 'wb') as outfile:
            outfile.write(json_job_ids)



if __name__ == "__main__":
    warprint("=============================")
    warprint("START: Executing IBM Experiment pipeline")
    warprint("=============================")
    argv = sys.argv[1:]
    import time
    t0 = time.time()

    if argv != []:
        run(cmd_args=argv)
    else:
        run()

    t1 = time.time()
    qprint("=============================")
    qprint("Elapsed time: {} sec".format(t1 - t0))


# ----------------------------------------------------------------
# how to run:
# ----------------------------------------------------------------
    #python run_ibm_experiment.py -c path/to/config.ini
    #python run_ibm_experiment.py
















#------------------------------------
#------------------------------------
#------------------------------------
#------------------------------------
#qrem.run_characterisation_experiment()
#? qrem.run_mitigation_experiment()
#qrem.download_characterisation_experiment_results()
#qrem.download_mitigation_experiment_results()
#qrem.run_characterisation()
#qrem.run_mitigation()

