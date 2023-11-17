from pathlib import Path
import configargparse
import ast
import os

from qrem.common.printer import qprint, errprint, warprint
#----------------
# Helper functions for parsing ini config files 
#----------------
def parse_str(in_str):
    'Helper function to evaluate string in config file'
    # could be a straight string, e.g 'string', or (double) quoted string "'string'", depending on the version of configargparse.
    if in_str[0] == '"': # double quoted
        return in_str.replace('"', '')
    elif in_str[0] == "'": # double quoted
        return in_str.replace("'", '')
    else:
        return in_str   
def parse_literal(instr):
    """ converts a string specifying a list """
    if instr is None or len(instr) == 0:
        return None
    print(instr)
    ret = ast.literal_eval(instr)
    print(ret)
    if isinstance(ret, str): # it was a list within a list
        print(ret)
        ret = ast.literal_eval(ret)
    return ret

def parse_boolean(b):
    """ Interpret various user inputs as boolean values """
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
common_path = Path(__file__).resolve().parent 

class QremConfigLoader:
    config_parser = None

    def load(cmd_args=None, default_path=str(Path(common_path,'default.ini')) , as_dict=False):
        """ processes config file specific for QREM """

        common_path = Path(__file__).resolve().parent   

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
                                help='Name of the device to run experiment on. Available names: ASPEN-M-2, TODO FILL IN')
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

    # for key, val in cfg_dict_orig.items():
    #     cfg_dict[key] = val.lower()
    print(aa)