import pickle 

from tqdm import tqdm
import numpy as np

import qrem.common.providers.ibmutils.data_converters




from qrem.cn import simulation as cnsimulation 
from qrem.usecases.mitigation import create_mitigation_data_KKM as mitigation_routines 
from qrem.functions_qrem import functions_benchmarks as fun_ben
from qrem.functions_qrem import ancillary_functions as anf
from qrem.functions_qrem import functions_data_analysis as fdt
from qrem.types import cn_noise_model 
from datetime import date
import statistics






        




if __name__ == '__main__':
 




    #specify n#
    number_of_qubits = 100
    #version_name = 'v4'
    


    #################################################################################
    # Realization: custom_n (n=1,2,3...10) fixed noise model tested for all versions v1,v2,v3,v4,v4_temp (temp =1, temp = 10)
    #################################################################################


    ################################################################################
              
    ################################################################################

    model_directory = '/home/kasiakm/Documents/QREM_DATA/Simulations100q/'
    

    ######################################################################################
    #### mitigation data creation                                       ##################
    ####specify number of circuits, shots and directory where data is to be saved#########
    #### number_of_benchmark_circuits : number of circuits used to perform mitigation  ###
    ######################################################################################
    data_directory = '/home/kasiakm/Documents/QREM_DATA/Simulations100q/'

    number_of_circuits = 1500

    number_of_shots = 10**4

    number_of_benchmark_circuits = 300

    for i in range(1):
        j=i+11
        model_file = '100qmodel'+str(j)+'.pkl'

        with open(model_directory+ model_file, 'rb') as filein:
            model_noise_dictionary = pickle.load(filein)
    
        model_noise_matrices_dictionary = model_noise_dictionary['noise_matrices']
        print('model dictionary loaded:',model_noise_matrices_dictionary.keys())
        print('dict:', model_noise_matrices_dictionary.keys())

        noise_model_simulation = cn_noise_model.CNModelData(number_of_qubits=number_of_qubits)
        noise_model_simulation.set_noise_model(model_noise_matrices_dictionary)
        realization = 'model'+str(j)+'_test'
        



    #characterization procedure is run, noise model data are computed
        noisy_experiment_dictionary = mitigation_routines.simulate_noisy_experiment(noise_model=noise_model_simulation,number_of_circuits=number_of_circuits,number_of_shots=number_of_shots,number_of_benchmark_circuits=number_of_benchmark_circuits,data_directory=data_directory,name_id = realization+str(number_of_circuits)+'x'+str(number_of_shots), return_ideal_experiment_data=False,save_data=True)

        noisy_results_dictionary = noisy_experiment_dictionary['noisy_results_dictionary']