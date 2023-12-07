"""Run AWS Experiment pipeline

This module allows to run a characterization/mitigation/benchmark experiment 
on a chosen AWS braket quantum machine. It uses a config file, which example can be found in 
<configs/ default_aws.ini> file. Read the config to understand avialable parameters.


 documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

Example
-------
You can run experiment by (i) providing a given config via -C argument, for example

    $ python run_aws_experiment.py -C \\path\\to\\experiment\\config.ini

    
Notes
-----
    @authors: 
    @contact:
"""


import os, sys
import qrem
import pickle

from pathlib import Path

import numpy as np

from qrem.common.io import date_time_formatted
from qrem.common.printer import qprint, errprint, warprint
from qrem.common.experiment import tomography as tomography
from qrem.qtypes import CircuitCollection
from qrem.common.constants import CIRCUIT_DATA_TYPE as c_type
from qrem.providers import aws_braket
# ----------------------------------------------------------------
# this are the parameters of running qrem in future / maybe move to config[GENERAL]?:
# you will be able to run it asa command
# ----------------------------------------------------------------
CONFIG_PATH = ["--config_file", "/media/tuzjan/T7/work_tuzjan/experiment_settings/aws_rigetti_coherence_experiment/aws_rigetti_coherence.ini"]

#TODO (PP) write version of runner that works with existing circuit collection file

def run( cmd_args=CONFIG_PATH, verbose_log = True,idle_run = False): 
    # ----------------------------------------------------------------
    #[1] We load an already defined default config
    # ----------------------------------------------------------------
    config = qrem.load_config(cmd_args = cmd_args, verbose_log = verbose_log)

    EXPERIMENT_NAME = config.experiment_name

    EXPERIMENT_FOLDER_PATH = Path(config.experiment_path)
    if not EXPERIMENT_FOLDER_PATH.is_dir():
        EXPERIMENT_FOLDER_PATH.mkdir( parents=True, exist_ok=True )

    BACKUP_CIRCUITS = config.backup_circuits
    BACKUP_CIRCUITS_METADATA = config.backup_circuits_metadata
    JOB_TAGS = list(config.job_tags)



    # ----------------------------------------------------------------
    #[2] Get info from provider about valid qubits
    # ----------------------------------------------------------------

    aws_device, metadata = aws_braket.get_device(device_full_name = config.device_name, verbose_log = config.verbose_log);
    valid_qubit_properties = aws_braket.get_valid_qubits_properties(device=aws_device, threshold=None, verbose_log = config.verbose_log)#config.gate_threshold, verbose_log = config.verbose_log)

    METADATA = metadata
    METADATA["date"] = date_time_formatted()
    METADATA["JOB_TAGS"] = JOB_TAGS
    if BACKUP_CIRCUITS_METADATA:
        METADATA["valid_qubit_properties"] = valid_qubit_properties

    number_of_qubits = valid_qubit_properties["number_of_good_qubits"]
    good_qubits_indices = valid_qubit_properties["good_qubits_indices"]

    if config.ground_states_circuits_path and config.number_of_ground_states != None:
        circuits_ground_states_preparation_collection = []

        GROUND_STATES_PATH = config.ground_states_circuits_path
        GROUND_STATES_NUMBER = config.number_of_ground_states
        with open(GROUND_STATES_PATH, 'rb') as filein:
            hamiltonians_data_dictionary= pickle.load(filein)
            GROUND_STATES_NUMBER = min(config.number_of_ground_states,len(hamiltonians_data_dictionary))

            for i in range(GROUND_STATES_NUMBER):
                ground_state = hamiltonians_data_dictionary[i]['ground_state']
                if i==0:
                    circuits_ground_states_preparation_collection=np.array([ground_state])
                else:
                    circuits_ground_states_preparation_collection=np.append(circuits_ground_states_preparation_collection,np.array([ground_state]),axis=0)

    # ----------------------------------------------------------------
    #[3] TODO (PP) check if there are specific qubits subset defined in config,. if yes - check against good_qubits_indices and choose only the subset
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    #[4] Generate circuits
    # ----------------------------------------------------------------
    #[4.0] Setup circuit collection object
    qrem_circuit_collection = CircuitCollection()
    qrem_circuit_collection.experiment_name = EXPERIMENT_NAME
    qrem_circuit_collection.device = "" #TODO SET DEVICE
    qrem_circuit_collection.load_config(config=config)
    qrem_circuit_collection.qubit_indices = good_qubits_indices
    qrem_circuit_collection.metadata = METADATA

    #[4.1] TODO (PP) finish implementation of generate_circuits, so you know how many shots per circuit you need
    qrem_circuit_collection.circuits, _, theoretical_total_circuit_count, theoretical_number_of_shots = tomography.generate_circuits(   number_of_qubits = number_of_qubits,
                                                                            experiment_type = config.experiment_type, 
                                                                            k_locality = config.k_locality,
                                                                            limited_circuit_randomness = config.limited_circuit_randomness,
                                                                            imposed_max_random_circuit_count = config.random_circuits_count,
                                                                            imposed_max_number_of_shots = config.shots_per_circuit)
    


    # ----------------------------------------------------------------
    #[4.1] Add coherence witness circuits
    # ----------------------------------------------------------------
    # 

    

    if config.coherence_witness_circuits:

        COHERENCE_WITNESS_CIRCUITS_PATH = Path(config.coherence_witness_circuits_path)

        with open(COHERENCE_WITNESS_CIRCUITS_PATH, 'rb') as filein:
            coherence_witness_circuits =  pickle.load(filein)

        qrem_circuit_collection.circuits = np.append(qrem_circuit_collection.circuits,coherence_witness_circuits,axis=0)

        qrem_circuit_collection.experiment_type = 'QDOT'




    # ----------------------------------------------------------------
    #[4.2] Add ground state circuits
    # ----------------------------------------------------------------
    # - TODO (PP) does not check correctness of circuits now and of pickle object
    
    if config.ground_states_circuits_path and config.number_of_ground_states != 0:

                
        qrem_circuit_collection.circuits = np.append(qrem_circuit_collection.circuits,circuits_ground_states_preparation_collection,axis=0)
    
    # ----------------------------------------------------------------
    #[4.3] Order of circuits is rearranged to ensure that benchmark and 
    # coherence witness circuits are run in the same availability window ""
    # ----------------------------------------------------------------


    #number_of_coherence_and_benchmark_circuits = len(coherence_witness_circuits) + GROUND_STATES_NUMBER 

    #temporary_rearrangement_array = copy.copy(qrem_circuit_collection.circuits[-number_of_coherence_and_benchmark_circuits:])

    #qrem_circuit_collection.circuits[-number_of_coherence_and_benchmark_circuits:] = qrem_circuit_collection.circuits[100:100+number_of_coherence_and_benchmark_circuits]

    #qrem_circuit_collection.circuits[100:100+number_of_coherence_and_benchmark_circuits] = temporary_rearrangement_array



    # - TODO (PP) save final number of shots to collection
    # qrem_circuit_collection.no_shots = ?

    #[5] At this stage, circuits are ready, we can save them to file
    if BACKUP_CIRCUITS:    
        qrem_circuit_collection.export_json(str(EXPERIMENT_FOLDER_PATH.joinpath("input_circuit_collection.json")), overwrite = False)

    #[6] Now we need to use aws_braket module to prepare circuits in aws_braket format
    braket_circuits = aws_braket.translate_circuits_to_braket_format(qrem_circuit_collection,valid_qubit_indices=good_qubits_indices)

    SUBMISSION_FOLDER_PATH = Path(config.experiment_path).joinpath("job_submission")

    if not SUBMISSION_FOLDER_PATH.is_dir():
        EXPERIMENT_FOLDER_PATH.mkdir( parents=True, exist_ok=True )
    
    total_number_of_circuits = len(qrem_circuit_collection.circuits) 
    print(f"Total number of circuits: {total_number_of_circuits} ")


    #[6] Now we need to prepare and run circuits
    circuits_ready = aws_braket.prepare_cricuits( braket_circuits = braket_circuits,
                     circuit_collection = qrem_circuit_collection,
                     good_qubits_indices = good_qubits_indices,
                     number_of_repetitions = config.shots_per_circuit,
                     number_of_task_retries = config.aws_braket_task_retries,
                     experiment_name = EXPERIMENT_NAME,
                     job_tags = JOB_TAGS,
                     pickle_submission = config.aws_pickle_results,
                     metadata = METADATA,
                     verbose_log = config.verbose_log,
                     job_dir = SUBMISSION_FOLDER_PATH,
                     overwrite_output = False)
    


    #TODO UNCOMMENT TO RUN
    if not circuits_ready:
        print("ERROR during circuit creation, aborting.")
    else:
        if not idle_run:
            aws_braket.execute_circuits( device_name=config.device_name,
                            pickle_submission = config.aws_pickle_results,
                            job_dir=SUBMISSION_FOLDER_PATH)



if __name__ == "__main__":
    TEST_RUN=False
    warprint("=============================")
    warprint("START: Executing AWS-BRAKET Experiment pipeline")
    warprint("=============================")
    argv = sys.argv[1:]
    import time
    t0 = time.time()

    if argv != []:
        run(cmd_args=argv, idle_run = TEST_RUN)
    else:
        run(idle_run = TEST_RUN)

    t1 = time.time()
    qprint("=============================")
    qprint("Elapsed time: {} sec".format(t1 - t0))


# ----------------------------------------------------------------
# how to run:
# ----------------------------------------------------------------
    #python run_aws_experiment.py -c path/to/config.ini
    #python run_aws_experiment.py
















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

