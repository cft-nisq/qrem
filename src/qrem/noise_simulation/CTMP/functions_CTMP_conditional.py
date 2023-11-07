#MOcomm - poorely documented, needs to be understood
import numpy as np
from collections import defaultdict
import random
from typing import Dict, Tuple, List
from tqdm import tqdm
import copy

def generate_binary_strings(n, l):
    if n == 0:
        return l
    else:
        if len(l) == 0:
            return generate_binary_strings(n - 1, ["0", "1"])
        else:
            return generate_binary_strings(n - 1, [i + "0" for i in l] + [i + "1" for i in l])

def kronecker_delta(v, w):
    if v==w:
        return 1
    else:
        return 0


def get_cond_prob(x, y,
                  generators_dictionary:Dict[Tuple[int],
                                             np.ndarray],
                  normalization_quasiprobability, printing=False):
    if printing: print('look for ' + str(x) + '->' + str(y))
    val = 0
    number_of_qubits = len(y)
    all_qubits_indices = set(range(number_of_qubits))
    if printing: print('supp: ' + str(all_qubits_indices))
    val += kronecker_delta(y, x[:number_of_qubits])
    if val == 1:
        if printing: print('DIAGONAL')
    for local_qubits_tuple in generators_dictionary.keys():
        operator = generators_dictionary[local_qubits_tuple]
        if printing: print(operator)
        # op_support_ordered = [int(qubit) for qubit in op_support_ordered_str.split('q')[1:]]
        if printing: print(local_qubits_tuple)

        op_support_cap_support = set(local_qubits_tuple).intersection(set(all_qubits_indices))
        if not op_support_cap_support == set():
            support_cond = True
            if printing: print('oscs: ' + str(op_support_cap_support))
            support_minus_op_support = set(all_qubits_indices).difference(set(local_qubits_tuple))
            for qubit in support_minus_op_support:
                if y[qubit] != x[qubit]:
                    support_cond = False
                    if printing: print('sc failed for qubit' + str(qubit))
                    break
            if (support_cond):
                if printing: print('sc OK')
                op_support_minus_support = set(local_qubits_tuple).difference(set(all_qubits_indices))
                if printing: print('osms: ' + str(op_support_minus_support))
                y_supp_vals = dict()

                for qubit in op_support_cap_support:
                    y_supp_vals[qubit] = y[qubit]

                no_of_y_summed = len(op_support_minus_support)

                if (no_of_y_summed > 0):
                    y_summed_bitstrings = generate_binary_strings(no_of_y_summed, [])
                    for y_summed_bitstr in y_summed_bitstrings:
                        i = 0
                        if printing: print('checking bitstr' + str(y_summed_bitstr))
                        for qubit in op_support_minus_support:
                            y_supp_vals[qubit] = y_summed_bitstr[i]
                            i += 1
                        y_supp_vals_ordered = "".join("%s" % y_supp_vals[qubit] for qubit in local_qubits_tuple)
                        row = int(y_supp_vals_ordered, 2)
                        x_supp_vals_ordered = "".join("%s" % x[qubit] for qubit in local_qubits_tuple)
                        col = int(x_supp_vals_ordered, 2)
                        if printing: print('added val ' + str(operator[row][col] / normalization_quasiprobability))
                        val += operator[row][col] / normalization_quasiprobability
                else:
                    y_supp_vals_ordered = "".join("%s" % y[qubit] for qubit in local_qubits_tuple)
                    row = int(y_supp_vals_ordered, 2)
                    x_supp_vals_ordered = "".join("%s" % x[qubit] for qubit in local_qubits_tuple)
                    col = int(x_supp_vals_ordered, 2)
                    if printing: print('no added bitstr')
                    if printing: print('added val ' + str(operator[row][col] / normalization_quasiprobability))
                    val += operator[row][col] / normalization_quasiprobability

    return val

def sample_from_B_matrix(input_state, generators_dictionary, quasiprobability_normalization, printing=False):
    y = ''
    no_of_bits = len(input_state)
    for bit_no in range(no_of_bits):
        y_0 = y + '0'
        y_1 = y + '1'
        if printing: print('testing ' + y_0 + ' and ' + y_1)
        cp_0 = get_cond_prob(input_state, y_0, generators_dictionary, quasiprobability_normalization)
        cp_1 = get_cond_prob(input_state, y_1, generators_dictionary, quasiprobability_normalization)
        tot = cp_0 + cp_1
        p_0 = cp_0 / tot
        if printing: print('p_0 ' + str(p_0))
        val = random.uniform(0, 1)
        if val < p_0:
            y = y + '0'
            if printing: print('extending ' + y)
        else:
            y = y + '1'
            if printing: print('extending ' + y)
    return y


def sample_from_power_of_B_matrix(matrix_power,
                                  input_state,
                                  generators_dictionary,
                                  noise_strength_CTMP,
                                  printing=False):
    output_state = copy.deepcopy(input_state)
    for step in range(matrix_power):
        if printing: print('pre ' + output_state)
        output_state = sample_from_B_matrix(output_state, generators_dictionary, noise_strength_CTMP)
        if printing: print('post ' + output_state)
    return output_state

def sample_from_CTMP_noise_model_conditional_probabilities(input_state:str,
                                                           number_of_samples:int,
                                                           generators_dictionary:Dict[Tuple[int],
                                                              np.ndarray],
                                                           normalization_quasiprobability:float):
    samples_from_noise_model = defaultdict(float)
    for _ in range(number_of_samples):
        random_power = np.random.poisson(normalization_quasiprobability)
        samples_from_noise_model[sample_from_power_of_B_matrix(random_power,
                                                               input_state,
                                                               generators_dictionary,
                                                               normalization_quasiprobability)]+=1
    return samples_from_noise_model