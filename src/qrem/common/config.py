"""
QREM Configuration Loader Module
--------------------------------

This module provides functionality for parsing configuration files and command-line arguments
specifically tailored for the Quantum Research Environment Manager (QREM) project.
The module includes helper functions for parsing various data types and a class `QremConfigLoader`
that facilitates loading and managing configuration settings.

The config file is designed to help in preparation and documenting the configuration of each QREM execution on quantum computers.
It is an ini file, that contains all the parameters necessary to configure an experiment. Parameters can also be passed as a command line arguments.

Example config files can be found in the qrem/configs folder of the package and accessed through followint variables:
- qrem.common.config.example_config_ibm_path
- qrem.common.config.example_config_aws_path

Sections
--------
The configuration is organized into sections: 'general', 'data', 'experiment', 'characterization' and 'mitigation'.
    general
        Contains general settings for the experiment, such as the experiment's name, author, and logging level.

    data
        Manages settings related to data handling, including backups of circuits, job IDs, and circuit metadata.

    experiment
        Specifies various parameters and settings directly related to the quantum experiment, including device information, provider details, experiment type, and quantum circuit configuration.

    characterization
        Contains settings for characterization of the quantum device.

    mitigation
        Contains settings for mitigation of the quantum device.


Configuration Parameters
------------------------


[general]
    experiment_name : str
        The name of the experiment, used for bookkeeping in all files.
    author : str
        The name or nickname of the author, saved in all files for tracking.
    verbose_log : bool
        Enables verbose logging for additional information during execution.

[data]
    backup_circuits : bool
        Indicates whether to save the original list of circuits in the QREM definition to a file.
    backup_job_ids : bool
        Specifies whether to save job IDs to a file after submitting data to a backend.
    backup_circuits_metadata : bool
        Determines whether to add additional information (like qubit readout errors, list of bad qubits) to the saved circuits list.

[experiment]
    experiment_path : str
        The file path for storing experiment-related files.
    device_name : str
        The name of the quantum device (e.g., 'ibm_seattle', 'Aspen-M-3').
    provider : str
        The quantum computing provider (e.g., 'IBM', 'AWS-BRAKET').
    ibm_connection_method : str
        Specific to IBM, defines the connection method to the quantum machine (e.g., 'RUNTIME_SESSIONS', 'RUNTIME').
    provider_instance : str
        For IBM, specifies the provider instance, such as 'ibm-q/open/main'.
    aws_pickle_results : bool
        Specifies if results should be pickled when using AWS Braket.
    aws_braket_task_retries : int
        The number of retries for AWS Braket tasks. 
    experiment_type : str
        The type of experiment, such as 'QDoT', 'DDoT', 'QDT', 'RFE'.
    k_locality : int
        The locality of the experiment, with values ranging from 2 to 5.
    gate_threshold : float
        Gate error threshold, if crossed - qubits will be excluded from calculations. ranging from 0 to 1. A value of 0 or Null includes all qubits.
    ground_state_circuits : bool
        Determines if ground state circuits should be included.
    ground_state_circuits_path : str
        File path to the pickle file containing ground state circuits information.
    ground_states_count : int
        The number of ground states to be considered in the experiment.
    limited_circuit_randomness : bool
        Indicates if limitations should be imposed on number of random circuits (e.g., number of random circuits).
    random_circuits_count : int
        Total count of random circuits to be sent, relevant when 'limited_circuit_randomness' is True.
    shots_per_circuit : int
        Number of shots (repetitions) per circuit, relevant when 'limited_circuit_randomness' is True.
    job_tags : tuple
        Tags used for identifying jobs sent to the backend. 
    qbits_of_interest : list or None
        List of qubits to be used, following the device's native indexing convention. Not yet implemented.
    ensure_completnes_on_pairs : bool
        Ensures completeness of the experiment on qubit pairs (currently always true).

Usage
-----
    The configuration file is used by the QREM module to set up and manage quantum experiments. 
    Users can modify the file to customize various aspects of their experiments,
    including experimental parameters, data backup options, and device-specific settings.           

Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""

from pathlib import Path
import configargparse
import ast
import os

from qrem.common.printer import qprint, errprint, warprint


example_config_ibm_path= Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_ibm.ini'))
"""Example configuration file path for execution on IBM Quantum devices."""

example_config_aws_path= Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_aws.ini'))
"""Example configuration file path for execution on AWS Braket Quantum devices."""

#----------------
# Helper functions for parsing ini config files 
#----------------
def parse_str(in_str):
    """
    Helper function to evaluate a string in the config file.

    Parameters
    ----------
    in_str : str
        The input string to be evaluated.

    Returns
    -------
    str
        The evaluated string.

    """
    # could be a straight string, e.g 'string', or (double) quoted string "'string'", depending on the version of configargparse.
    if in_str[0] == '"': # double quoted
        return in_str.replace('"', '')
    elif in_str[0] == "'": # double quoted
        return in_str.replace("'", '')
    else:
        return in_str   
def parse_literal(instr):
    """
    Converts a string specifying a list or tuple.

    Parameters
    ----------
    instr : str
        The input string specifying a list.

    Returns
    -------
    any
        The converted list.

    """
    if instr is None or len(instr) == 0:
        return None
    ret = ast.literal_eval(instr)
    #print(ret)
    if isinstance(ret, str): # it was a list within a list
        #print(ret)
        ret = ast.literal_eval(ret)
    return ret

def parse_boolean(b):
    """
    Interpret config parameter as a boolean values.

    Parameters
    ----------
    b : str
        The input string representing a boolean value.

    Returns
    -------
    bool
        The interpreted boolean value.

    Raises
    ------
    ValueError
        If the input string cannot be interpreted as a boolean.

    """
    if len(b) < 1:
        raise ValueError('Cannot parse empty string into boolean.')
    b = b[0].lower()
    if b in ('t', 'y', '1'):
        return True
    if b in ('f', 'n', '0'):
        return False
    raise ValueError(f'Cannot interprete {b} as a boolean.')


#----------------
# path to local folder that contains find default config
# Order of precedence, from most important:
# - command line args / args added from config
# - environment variables 
# - config file values 
# - defaults defined in code
#----------------
#configs_path = Path(__file__).resolve().parent.parent.joinpath("configs") 

class QremConfigLoader:
    config_parser = None


    def load(cmd_args=None, default_path=str(example_config_ibm_path) , as_dict=False):
        """
        Processes a config file specific for QREM.

        Parameters
        ----------
        cmd_args : list, optional
            Command line arguments (default is None).
        default_path : str, optional
            Path to the default config file (default is 'default.ini' located within "qrem/config" path of the module ).
        as_dict : bool, optional
            If True, returns the configuration as a dictionary (default is False).

        Returns
        -------
        configargparse.Namespace or dict
            The parsed configuration.

        """


        # If we receive a relative path, it is relative to this file location, not the cwd which could vary, e.g. if you run it with hython
        if os.path.isabs(default_path):
            cfg_path = default_path
        else:
            FILEPATH = os.path.dirname(os.path.realpath(__file__))
            cfg_path = os.path.join(FILEPATH, default_path)

        assert os.path.exists(cfg_path) #this should go in the final version of code

        QremConfigLoader.config_parser = configargparse.ArgParser(default_config_files=[cfg_path],
                                                description='QREM run config')
        
        # First, we get existing config file and add 
        QremConfigLoader.config_parser.add('-c', '--config_file', required=False, is_config_file=True, 
                        help='external config file path')
        
        #----
        #[general]
        # experiment_name and author will be saved in all of the files to be able to track experiment authors. can be a nick or any other tag
        QremConfigLoader.config_parser.add_argument('--experiment_name', type=parse_str, required=True, default='Example Experiment',
                                help='Experiment name will be saved in all of the filesfor bookeeping.')
        QremConfigLoader.config_parser.add_argument('--author', type=parse_str, required=True, default='Anonymous',
                                help='Author name/nick that will be saved in all of the filesfor bookeeping.')
        QremConfigLoader.config_parser.add_argument('--verbose_log', type=parse_boolean, required=False, default=False,
                                help='Turn on verbose logging for more printouts with info')

        #----
        #[data]
        QremConfigLoader.config_parser.add_argument('--backup_circuits', type=parse_boolean, required=False, default=True,
                                help='Whether to save original list of circuits in QREM definition to a file in experiment folder')
        QremConfigLoader.config_parser.add_argument('--backup_job_ids', type=parse_boolean, required=False, default=True,
                                help='Whether to save job ids to a file in experiment folder  after sending data to a backend ')
        QremConfigLoader.config_parser.add_argument('--backup_circuits_metadata', type=parse_boolean, required=False, default=False,
                                help='Whether to add more information about qubits and circuits to saved circuits list (like qubit readout errors list, list of bad qubits etc)')

        #----
        #[experiment]
        QremConfigLoader.config_parser.add_argument('--experiment_path', type=parse_str, required=True, default='C:\\experiments\\experiment_example\\',
                                help='Path to save all the expreiment inputs and results')
        QremConfigLoader.config_parser.add_argument('--device_name', type=parse_str, required=True, default='ASPEN-M-2',
                                help='Name of the device to run experiment on')
        QremConfigLoader.config_parser.add_argument('--provider', type=parse_str, required=True, default='AWS-BRAKET',
                                help='Chosen provider. Available: AWS-BRAKET, IBM, TODO: FILL IN')        
        QremConfigLoader.config_parser.add_argument('--ibm_connection_method', type=parse_str, required=True, default='RUNTIME',
                                help='Valid only for IBM - how to connect to the quantum machine, available: RUNTIME_SESSIONS, RUNTIME, PROVIDER, JOB_EXECUTE, DEBUG')
        QremConfigLoader.config_parser.add_argument('--provider_instance', type=parse_str, required=True, default='AWS-BRAKET',
                                help='Valid only for IBM -  Provider instance, used values can be ibm-q/open/main or ... or ibm-q-psnc/internal/reservations')        

        QremConfigLoader.config_parser.add_argument('--experiment_type', type=parse_str, required=True, default='QDoT',
                                help='Chosen experiment type. Available values: QDoT, DDoT, QDT, RFE')
        QremConfigLoader.config_parser.add_argument('--k_locality', type=int, required=False, default=2,
                                help='k_locality value. Currently supported between 2-5')
        QremConfigLoader.config_parser.add_argument('--gate_threshold', type=float, required=False, default=0.005,
                                help='Assumed gate threshold for cut-off for qubits we want to take into account.')
        QremConfigLoader.config_parser.add_argument('--qbits_of_interest', required=False, type=parse_literal, default=None,
                                help='List of specific qubits to target for the experiment.')      
        QremConfigLoader.config_parser.add_argument('--limited_circuit_randomness', required=False, type=parse_boolean, default=True,
                                help='should any limitation of circuit randomness be imposed (e.g. limit on nubmer of random circuits/number of shots)')
        QremConfigLoader.config_parser.add_argument('--random_circuits_count', required=False, type=int, default=None,
                                help='count of random circuits to be sent (total = number of random circuits * how many times we repeat one)')
        QremConfigLoader.config_parser.add_argument('--shots_per_circuit', required=True, type=int, default=None,
                                help='How many times each random circuit should be repeated')
        QremConfigLoader.config_parser.add_argument('--aws_braket_task_retries', required=False, type=int, default=3,
                                help='How many times each aws task should be repeated')
        QremConfigLoader.config_parser.add_argument('--job_tags', required=False, type=parse_literal, default=["QREM_JOB",],
                                help='List of ags for the jobs send to the backend.')
        #QremConfigLoader.config_parser.add_argument('--run_on_all_qbits', required=False, type=parse_boolean, default=True)

        QremConfigLoader.config_parser.add_argument('--ensure_completnes_on_pairs', required=False, type=parse_boolean, default=True,
                                help='should we perform algorighmic drawing of circuits with complete pairs')
        
        QremConfigLoader.config_parser.add_argument('--ground_states_circuits', required=False, type=parse_boolean, default=True,
                                help='should a pre-prepared list of circuits of ground states of hamiltonians be added to random circuits')
        QremConfigLoader.config_parser.add_argument('--number_of_ground_states', type=int, required=False, default=10,
                                help='how many ground state circuits to add')
        QremConfigLoader.config_parser.add_argument('--ground_states_circuits_path', type=parse_str, required=False, default=None,
                                help='Path to pickle file with dictionary of int:circuit of qrem_circuits containing ground states')
        QremConfigLoader.config_parser.add_argument('--aws_pickle_results', type=parse_boolean, required=False, default=True,
                                help='Should we pickle results for AWS submission or calculate circuits on machine?')
        
        QremConfigLoader.config_parser.add_argument('--coherence_witness_circuits', type=parse_boolean, required=False, default=None,
                                help='Do circuits testing POVMs coherence should be included in the experiment? ')
        
        QremConfigLoader.config_parser.add_argument('--coherence_witness_circuits_path', type=parse_str, required=False, default=None,
                                help='Path to coherence witness circuits collection')
        #---
        #[simulation]
        QremConfigLoader.config_parser.add_argument('--name_id', type=parse_str, required=False,
                                                    default=None,
                                                    help='Unique name string included in all saved files from given simulation')
        QremConfigLoader.config_parser.add_argument('--noise_model_directory', type=parse_str, required=False,
                                                    default=None,
                                                    help='Path to the directory where the noise model file is')
        QremConfigLoader.config_parser.add_argument('--noise_model_file', type=parse_str, required=False,
                                                    default=None,
                                                    help='Name of the noise model file')

        QremConfigLoader.config_parser.add_argument('--save_data', type=parse_boolean, required=False,
                                                    default=True,
                                                    help='Should the simulation results be saved?')
        QremConfigLoader.config_parser.add_argument('--new_data_format', type=parse_boolean, required=False,
                                                    default=True,
                                                    help='Should the data be represented in the QREM data format?')
        QremConfigLoader.config_parser.add_argument('--model_from_file', type=parse_boolean, required=False,
                                                    default=False,
                                                    help='Should the noise model be taken from file?')
        QremConfigLoader.config_parser.add_argument('--add_noise', type=parse_boolean, required=False,
                                                    default=True,
                                                    help='Should noise be added? If False, the results of an ideal experiment are obtained')

        QremConfigLoader.config_parser.add_argument('--number_of_circuits', type=int, required=True,
                                                    default=None,
                                                    help='For how many input circuits the simulation should be run?')
        QremConfigLoader.config_parser.add_argument('--number_of_shots', type=int, required=True,
                                                    default=None,
                                                    help='How many shots per input circuit?')
        QremConfigLoader.config_parser.add_argument('--number_of_qubits', type=int, required=False,
                                                    default=None,
                                                    help='How many qubits in the simulated system? ')
        QremConfigLoader.config_parser.add_argument('--model_specification', type=parse_literal, nargs='+', required=False,
                                                    default=None,
                                                    help='Noise model specified as [[size1,number_of_clusters_of_size_1],[size2,number_of_clusters_of_size_2],..,[size_n,number_of_clusters_of_size_n]]')


        


        config, unknown  = QremConfigLoader.config_parser.parse_known_args(cmd_args)


        #print(options)

        #If you want config as dictionary, make all the keys lower letters
        if as_dict:
            # cfg_dict_orig = config.__dict__.copy()
            # cfg_dict = config.__dict__
            # for key, val in cfg_dict_orig.items():
            #     cfg_dict[key] = val.lower()
            return config.__dict__
    
        return config
    
    def values():
        """ Return summary of configuration values """
        print("----------")
        if QremConfigLoader.config_parser is not None:
            qprint("Configuration values:")
            print(QremConfigLoader.config_parser.format_values()) 
        else:
            errprint("ERROR: Config not defined") 

    def help():
        """ Print help for preparation of the config file """
        print("----------")
        if QremConfigLoader.config_parser is not None:
            qprint("Configuration help:")
            print(QremConfigLoader.config_parser.format_help()) 
        else:
            errprint("ERROR: Config not defined") 




if __name__ == "__main__":
    aa = QremConfigLoader.load()
    QremConfigLoader.help()
    QremConfigLoader.values()

    print(aa)

    # for key, val in cfg_dict_orig.items():
    #     cfg_dict[key] = val.lower()
    #print(aa)