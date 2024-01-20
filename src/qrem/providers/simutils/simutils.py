import qrem.common.io as io
import pickle
from qrem.cn import simulation as cnsimulation
import qrem.qtypes.cn_noise_model as cn

def custom_model_creator(number_of_qubits: int, model_specification: list[list[int]], directory_to_save: str, name_id: str = '', save_model: bool = True):
    '''
        Function generates a random noise model for a given number of qubits and specified division into clusters, optionally saves a generated model
        to a file for future use

        Parameters
        ----------

        number_of_qubits: int

        model_specification: list
            list of pairs [[size1, number_of_clusters1],[size2,number_of_clusters_2]...] such that the total of
            size_i*number_of_clusters_i matches the total number of qubits

        directory_to_save: str
            path to save the noise model

        name_id: str
            customary name tag for the file with the noise model

        save_model: bool
            if True the model is saved to a file

        Returns
        ----------
        model: Type[CNModelData]

        Raises
        ----------
        Value Error if the model specification does not match the qubit number
        '''
    count = 0
    for t in model_specification:
        count += t[0] * t[1]
    if (count != number_of_qubits):
        raise ValueError(f"Qubit number mismatch by {count-number_of_qubits}")

    model = cnsimulation.create_random_noise_model(number_of_qubits=number_of_qubits,
                                                         clusters_specification=model_specification)
    model_dict = model.get_dict_format()
    file_name = str(number_of_qubits) + 'qmodel' + name_id
    if save_model:
        io.save(model_dict, directory=directory_to_save, custom_filename=file_name, overwrite=False)

    return model

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


def model_specification_from_string(model_spec: str):
    if len(model_spec)==0:
        return None
    divided = list(model_spec)

    symbols = ['[',']']
    for s in symbols:
        while divided.count(s):
            divided.remove(s)
    divided.append(",")
    num_count = 0
    spec_list = []
    while divided.count(","):
        i = divided.index(",")
        n = int(''.join(divided[0:i]))

        if num_count%2:
            spec_list[-1].append(n)
        else:
            spec_list.append([n])
        num_count += 1

        del divided[0:i+1]
    return spec_list

if __name__ == "__main__":
    print(model_specification_from_string("[[2,5],[3,4]]"))
