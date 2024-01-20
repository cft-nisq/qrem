from qrem.cn import simulation as cnsimulation 
from qrem.mitigation import mitigation_routines
from qrem.benchmarks import hamiltonians
from qrem.characterization import characterization_routine
from datetime import date
from qrem.pipelines import simulate_experiment
from qrem.qtypes.characterization_data import CharacterizationData
from qrem.qtypes.mitigation_data import MitigationData
from qrem.common import convert
import pickle





################################################################################
############################### Data are loaded here ###########################
################################################################################
  


 
#set path to data directory
DATA_DIRECTORY= '/media/tuzjan/T7/work_tuzjan/article_analysis/'




FILE_NAME_RESULTS_RIG =  'RESULTS_RIGETTI_ASPEN-M-3.pkl'

FILE_NAME_GROUND_STATES_LIST_RIG = 'RIGETTI-ASPEN-M-3_GROUND_STATES_LIST.pkl'

COHERENCE_WITNESS_CIRCUITS_PATH_RIG = 'RIGETTI-ASPEN-M-3_COHERENCE_WITNESS_CIRCUITS.pkl'

FILE_NAME_HAMILTONIANS_RIG = 'RIGETTI-ASPEN-M-3_HAMILTONIANS_DICTIONARY.PKL'

FILE_NAME_MARGINALS_RIG = 'RIGETTI-ASPEN-M-3_MARGINALS_DICTIONARY.pkl'







#experimental results are loaded
with open(DATA_DIRECTORY + FILE_NAME_RESULTS_RIG, 'rb') as filein:
    results_dictionary_rig = pickle.load(filein)

#ground states list us loaded
with open(DATA_DIRECTORY+FILE_NAME_GROUND_STATES_LIST_RIG, 'rb') as filein:
    circuits_ground_states_preparation_collection_rig= pickle.load( filein)

#coherence strength circuits are loaded 
with open(DATA_DIRECTORY+COHERENCE_WITNESS_CIRCUITS_PATH_RIG  , 'rb') as filein:
    coherence_witness_circuits_rig =  pickle.load(filein)

#Optional: marginals are loaded
with open(DATA_DIRECTORY + FILE_NAME_HAMILTONIANS_RIG, 'rb') as filein:
    hamiltonians_dictionary_rig = pickle.load(filein)

#Optional: marginals are loaded
with open(DATA_DIRECTORY + FILE_NAME_MARGINALS_RIG, 'rb') as filein:
    marginals_dictionary_rig = pickle.load(filein)
marginals_dictionary_rig = marginals_dictionary_rig['marginals_dictionary']

characterization_data_container = CharacterizationData()

characterization_data_container.experiment_type = 'qdot'

characterization_data_container.results_dictionary = convert.convert_results_dictionary_to_new_format(results_dictionary_rig)

characterization_data_container.ground_states_list  = circuits_ground_states_preparation_collection_rig 

characterization_data_container.marginals_dictionary = marginals_dictionary_rig 

characterization_data_container.coherence_witnesses_list = list(coherence_witness_circuits_rig)

characterization_data_container= characterization_routine.execute_characterization_workflow(characterization_data_container=characterization_data_container,find_neighbors=True,name_id='Rigetti_Aspen-M-3')

