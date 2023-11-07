import numpy as np
from qrem.common import io
from qrem.common import probability

from qrem.cn.simtools.auxiliary_merging_functions import divide_qubits_in_the_clusters as dq

import qrem.cn.simulation 

#Script for custom noise model creation, a loop goes through a list of specifications [cluster_size,number_of_clusters] and saves the models in files for future access
directory = '/home/kasiakm/Documents/QREM_DATA/Simulations100q/'
number_of_qubits = 100
#specifications = [[[4,25]],[[4,20],[2,10]],[[4,15],[3,10],[2,5]],[[3,30],[2,5]],[[4,1],[3,32]],[[4,10],[2,30]],[[3,33],[1,1]],[[3,24],[2,14]],[[2,50]],[[4,20],[3,6],[1,2]]]
specifications = [[[10,10]],[[5,20]]]


for s in specifications:
    count = 0
    for t in s:
        count+=t[0]*t[1]
    if(count!=number_of_qubits):
        print("WARNING!!! qubit number mismatch")


first_label_number = 0

for i in range(len(specifications)):
    model = qrem.cn.simulation.create_random_noise_model(number_of_qubits = number_of_qubits,clusters_specification = specifications[i])
    model_dict = model.get_dict_format()
    file_name = str(number_of_qubits)+'qmodel'+str(i+first_label_number)
    io.save(model_dict,
            directory=directory,
            custom_filename=file_name,
            overwrite=False)
    print(f"{i} done")



