"""
@authors: Oskar Słowik, Filip Maciejewski
@contact: osslowik@gmail.com
"""

#(PP) this seems to be a messy file, need MO to dig through it
from typing import Dict
import copy
import math
import random

import qutip
import numpy as np
import scipy as sc

from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

from qrem.common.printer import qprint

#MOVE_TO math.py
def l1_norm(v):
    sum = 0
    for x in v:
        sum += abs(x)
    return sum


#MOVE_TO math.py what about accuracy? what are the types here?
def delta(v, w):
    val = 0
    if v == w:
        val = 1
    return val


#TODO_MO - is it a repetition from any other binary string generator?
def generate_binary_strings(n, l):
    if n == 0:
        return l
    else:
        if len(l) == 0:
            return generate_binary_strings(n - 1, ["0", "1"])
        else:
            return generate_binary_strings(n - 1, [i + "0" for i in l] + [i + "1" for i in l])


#(PP) dobuled in other files
#(PP) part of a model of Bravi noise
#(PP) should go to qrem.noise_modelling
def return_1local_gen(type, r):
    if type == '0':
        gen = [[-r, 0], [r, 0]]
    elif type == '1':
        gen = [[0, r], [0, -r]]
    else:
        raise ValueError('Wrong generator type!')
    return np.array(gen)


#(PP) dobuled in other files
#(PP) part of a model of Bravi noise
#(PP) should go to qrem.noise_modelling
def return_2local_gen(type, r):
    if type == '00':
        gen = [[-r, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [r, 0, 0, 0]]
    elif type == '01':
        gen = [[0, 0, 0, 0], [0, -r, 0, 0], [0, r, 0, 0], [0, 0, 0, 0]]
    elif type == '10':
        gen = [[0, 0, 0, 0], [0, 0, r, 0], [0, 0, -r, 0], [0, 0, 0, 0]]
    elif type == '11':
        gen = [[0, 0, 0, r], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -r]]
    else:
        raise ValueError('Wrong generator type!')
    return np.array(gen)


'''
s - a list of numpy vectors representing measurement results
O - a numpy vector with values O(x)
S - a list of numpy arrays representing operatrs S_alpha
c - a list of weights of each S_alpha
kronecker_delta - kronecker_delta parameter
'''


def bravyi_algorithm_1_naive(s, O, S, c, delta):
    c_l1_norm = l1_norm(c)
    # print(c_l1_norm)
    M = len(s)  # number of experiments
    # print(M)
    d = O.shape[0]  # dimension
    # print(d)
    T = math.ceil(4 * c_l1_norm ** 2 / (delta ** 2))
    # print(T)
    xi = np.zeros(T)
    q_alpha = [abs(x) / c_l1_norm for x in c]
    # print(q_alpha)
    for t in range(T):
        i = random.randrange(0, M)
        # print("i"+str(i))
        alpha = np.random.choice(np.arange(0, len(c)), p=q_alpha)
        # print(alpha)
        q_x = S[alpha].dot(s[i])
        # print(q_x)
        x = np.random.choice(np.arange(0, d), p=q_x)
        # print(x)
        xi[t] = np.sign(c[alpha]) * O[x]
        # print(xi[t])
    return c_l1_norm * np.sum(xi) / T


'''
O=np.array([0.2,0.3,-1,1])
S=[np.array([[0,0,0,0],[1,0,0,0],[0,0,0,1],[0,1,1,0]]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]
s=[np.array([0,0,1,0]),np.array([0,0,1,0]),np.array([0,0,1,0]),np.array([0,0,1,0]),np.array([0,0,0,1])]
c=[0.95,0.05]
kronecker_delta=0.1
print(bravyi_algorithm_1(O,S,s,c,kronecker_delta))
'''


# p(y_0...y_k | x_0 ... x_n), k<=n
def get_cond_prob(x, y, G_dict, lambda_par, printing=False):
    if printing: print('look for ' + str(x) + '->' + str(y))
    val = 0
    rank = len(y)
    support = set(range(rank))
    if printing: print('supp: ' + str(support))
    val += delta(y, x[:rank])
    if val == 1:
        if printing: print('DIAGONAL')
    for op_support_ordered_str in G_dict:
        operator = G_dict[op_support_ordered_str]
        if printing: print(operator)
        op_support_ordered = [int(qubit) for qubit in op_support_ordered_str.split('q')[1:]]
        if printing: print(op_support_ordered)

        op_support_cap_support = set(op_support_ordered).intersection(set(support))
        if not op_support_cap_support == set():
            support_cond = True
            if printing: print('oscs: ' + str(op_support_cap_support))
            support_minus_op_support = set(support).difference(set(op_support_ordered))
            for qubit in support_minus_op_support:
                if y[qubit] != x[qubit]:
                    support_cond = False
                    if printing: print('sc failed for qubit' + str(qubit))
                    break
            if (support_cond):
                if printing: print('sc OK')
                op_support_minus_support = set(op_support_ordered).difference(set(support))
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
                        y_supp_vals_ordered = "".join(
                            "%s" % y_supp_vals[qubit] for qubit in op_support_ordered)
                        row = int(y_supp_vals_ordered, 2)
                        x_supp_vals_ordered = "".join("%s" % x[qubit] for qubit in op_support_ordered)
                        col = int(x_supp_vals_ordered, 2)
                        if printing: print('added val ' + str(operator[row][col] / lambda_par))
                        val += operator[row][col] / lambda_par
                else:
                    y_supp_vals_ordered = "".join("%s" % y[qubit] for qubit in op_support_ordered)
                    row = int(y_supp_vals_ordered, 2)
                    x_supp_vals_ordered = "".join("%s" % x[qubit] for qubit in op_support_ordered)
                    col = int(x_supp_vals_ordered, 2)
                    if printing: print('no added bitstr')
                    if printing: print('added val ' + str(operator[row][col] / lambda_par))
                    val += operator[row][col] / lambda_par

    return val


G_dict = dict()
r01_00 = 10
r01_11 = 9
r01_01 = 8

r12_00 = 7
r12_11 = 6
r12_01 = 5

# test_x=[[0,0,1],[0,0,0],[1,1,1]]
G_dict['q0q1'] = [[-r01_00, 0, 0, r01_11], [0, -r01_01, 0, 0], [0, r01_01, 0, 0],
                  [r01_00, 0, 0, -r01_11]]
G_dict['q0q2'] = [[-r12_00, 0, 0, r12_11], [0, -r12_01, 0, 0], [0, r12_01, 0, 0],
                  [r12_00, 0, 0, -r12_11]]

N = 3

x_set = generate_binary_strings(N, [])
y_set = generate_binary_strings(N - 1, [])
lambda_par = 10

'''
process_mat=dict()
for x in x_set:
    for y in y_set:
        process_mat[(x,y)]=get_cond_prob(x, y, G_dict, lambda_par)
        print(x+'->'+y+' '+str(process_mat[(x,y)]))


ser = pd.Series(list(process_mat.values()),
                  index=pd.MultiIndex.from_tuples(process_mat.keys()))
df = ser.unstack().fillna(0).transpose()
sns.heatmap(df)
plt.savefig("output2.png")
'''


def sample_from_B_pow_n(n, x, G_dict, lambda_par, printing=False):
    sample = x
    for step in range(n):
        if printing: print('pre ' + sample)
        sample = sample_from_B(sample, G_dict, lambda_par)
        if printing: print('post ' + sample)
    return sample


def sample_from_B(x, G_dict, lambda_par, printing=False):
    y = ''
    no_of_bits = len(x)
    for bit_no in range(no_of_bits):
        y_0 = y + '0'
        y_1 = y + '1'
        if printing: print('testing ' + y_0 + ' and ' + y_1)
        cp_0 = get_cond_prob(x, y_0, G_dict, lambda_par)
        cp_1 = get_cond_prob(x, y_1, G_dict, lambda_par)
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


# sample_from_power_of_B_matrix(10, x_set[0], G_dict, lambda_par)


O = {'000': 0, '001': 1, '010': 1, '011': 2, '100': 0, '101': 2, '110': 2, '111': 3}
s = ['000'] * 50
s2 = ['001'] * 50
s.extend(s2)
delta_par = 0.1

'''
s - a list of strings representing measurement results
O - a dict with keys - strings x and values - O(x)
kronecker_delta - kronecker_delta parameter
'''


def bravyi_algorithm_1(s, O, G_dict, delta_par, lambda_par):
    c_l1_norm = math.exp(2 * lambda_par)
    M = len(s)  # number of experiments
    # print('M '+str(M))

    T = math.ceil(4 * (c_l1_norm ** 2) / (delta_par ** 2))
    # print('T ' +str(T))
    sum_xi = 0
    for t in range(T):
        i = random.randrange(0, M)
        # print("i"+str(i))
        n = np.random.poisson(lambda_par)
        # print('n '+str(n))
        x = sample_from_B_pow_n(n, s[i], G_dict, lambda_par)
        # print('x '+x)

        if x in O.keys():
            if (n % 2) == 0:
                sign = 1.0
            else:
                sign = -1.0
            sum_xi += sign * O[x]
            # print(xi[t])
    return c_l1_norm * sum_xi / T


# print(bravyi_algorithm_1(s, O, G_dict, delta_par, lambda_par))



def sample_from_bravyi_noise_model(y, N, G_dict, lambda_par):
    out = []
    for t in tqdm(range(N)):
        n = np.random.poisson(lambda_par)
        # print('n ' + str(n))
        out.append(sample_from_B_pow_n(n, y, G_dict, lambda_par))
    return out


def bravyi_noise_model_naive_distribution(y, G_dict):
    N = len(y)
    G = convert_G_dict_to_G(G_dict, N)
    # print(G)
    # print(np.sum(G, axis=0))
    A = sc.linalg.expm(G)
    # print(A)
    # print(np.sum(A, axis=0))
    y_index = int(y, 2)
    # print(y_index)
    pd_x = A[:, y_index]
    return pd_x

#Double entry in
def construct_model_from_rates(r_dict):
    G_dict = dict()
    for op_support_ordered_str in r_dict:
        no_of_qubits = int(len(op_support_ordered_str) / 2)
        dim = int(math.pow(2, no_of_qubits))
        gen_total = np.zeros((dim, dim))
        for op_type in r_dict[op_support_ordered_str]:
            r = r_dict[op_support_ordered_str][op_type]
            if len(op_type) == 1:
                gen = return_1local_gen(op_type, r)
            elif len(op_type) == 2:
                gen = return_2local_gen(op_type, r)
            else:
                raise ValueError('Wrong generator dimension in r_dict!')
            gen_total = gen_total + gen
        G_dict[op_support_ordered_str] = gen_total
    return G_dict


r_dict = {'q0q1': {'00': 10, '11': 9, '01': 8}, 'q0q2': {'00': 7, '11': 6, '01': 5}}


# print(sample_from_CTMP_noise_model_conditional_probabilities(s, 10, G_dict, lambda_par))


# def generate_bravyi_G_dict(rates,N):

def embed_1q_operator(N, gate, index):
    if index == 0:
        embed_operator = np.kron(gate, np.eye(int(2 ** (N - 1))))
        return embed_operator
    else:
        first_eye = np.eye(2 ** (index))
        second_eye = np.eye(2 ** (N - index - 1))

        embeded_operator = np.kron(np.kron(first_eye, gate), second_eye)

        return embeded_operator


def embed_operator(N, gate, indices=[0, 1]):
    N_small = int(np.log2(gate.shape[0]))

    if N_small == 1:
        return embed_1q_operator(N, gate, indices[0])

    qutip_object = qutip.Qobj(gate, dims=[[2, 2] for i in range(N_small)])

    return qutip.qip.expand_operator(qutip_object, N, indices).full()


# def extend_matrix_element(gen_supp_indices, entry, G, N, supp1='', supp2=''):
#    if len(supp2) == N:
#        row
#        # add element
#        return

#    if len(supp1) == N:
#        extend_matrix_element(gen_supp_indices, entry, G, N, supp1, supp2 + '0')
#        extend_matrix_element(gen_supp_indices, entry, G, N, supp1, supp2 + '1')
#    else:
#        extend_matrix_element(gen_supp_indices, entry, G, N, supp1 + '0', supp2)
#        extend_matrix_element(gen_supp_indices, entry, G, N, supp1 + '1', supp2)

def generate_random_r_dict(N, bound):
    r_dict = dict()
    for qubit_1 in range(N):
        supp_1_local = 'q' + str(qubit_1)
        r_dict_sub_1 = dict()
        for op_type_1 in ['0', '1']:
            r_dict_sub_1[op_type_1] = bound * np.random.random()
        r_dict[supp_1_local] = r_dict_sub_1
        for qubit_2 in range(qubit_1 + 1, N):
            supp_2_local = supp_1_local + 'q' + str(qubit_2)
            r_dict_sub_2 = dict()
            for op_type_2 in ['00', '01', '10', '11']:
                r_dict_sub_2[op_type_2] = bound * np.random.random()
            r_dict[supp_2_local] = r_dict_sub_2
    return r_dict


def generate_random_G_dict(N, bound):
    r_dict = generate_random_r_dict(N, bound)
    G_dict = construct_model_from_rates(r_dict)

    # TODO OS: this can be done inside above function
    lambda_b = get_r_sum_from_r_dict(r_dict)
    return G_dict, lambda_b


# q1 q2 ...
def convert_G_dict_to_G(G_dict, N):
    dim = int(math.pow(2, N))
    G = np.zeros((dim, dim))
    for gen_supp in G_dict:
        gen_supp_indices = [int(qubit) for qubit in gen_supp.split('q')[1:]]
        operator = np.array(G_dict[gen_supp])
        # print(N)
        # print(operator)
        operator_emb = embed_operator(N, operator, indices=gen_supp_indices)
        # print(np.shape(operator_emb))
        G = G + operator_emb
    return G


def get_r_sum_from_r_dict(r_dict):
    r_sum = 0
    for op_support_ordered_str in r_dict:
        for op_type in r_dict[op_support_ordered_str]:
            r = r_dict[op_support_ordered_str][op_type]
            r_sum = r_sum + r
    return r_sum


def test_bravyi_sampling(y, bound, no_of_samples):
    no_of_qubits = len(y)
    dim = int(math.pow(2, no_of_qubits))
    pd_x_approx = np.zeros(dim)
    G_dict, lambda_par = generate_random_G_dict(no_of_qubits, bound)
    pd_x = bravyi_noise_model_naive_distribution(y, G_dict)
    samples = sample_from_bravyi_noise_model(y, no_of_samples, G_dict, lambda_par)
    for sample in samples:
        index = int(sample, 2)
        pd_x_approx[index] += 1
    pd_x_approx = pd_x_approx / len(samples)

    pointwise_error = np.absolute(pd_x_approx - pd_x)
    TVD = np.max(pointwise_error)
    print(f'TVD: {TVD:.5f}')
    # print(samples)
    # print(pd_x)
    # print(pd_x_approx)

    ind = np.arange(dim)
    # print(ind)
    width = 0.35
    plt.bar(ind, pd_x_approx, width, label='sampling_alg')
    plt.bar(ind + width, pd_x, width, label='true')

    plt.ylabel('p')
    plt.title(
        'Bravyi sampling pd for y=' + y + ', samples= ' + str(no_of_samples) + f', TVD= {TVD:.5f}')

    xt = [s[::-1] for s in generate_binary_strings(no_of_qubits, [])]

    plt.xticks(ind + width / 2, xt)

    plt.legend(loc='best')
    plt.show()


# test_bravyi_sampling('0101',0.05, 10000)

'''
N = 3
r_d = generate_random_r_dict(N, 2)
print(r_d)
r_sum = get_r_sum_from_r_dict(r_d)
G_dict = construct_model_from_rates(r_d)
G = convert_G_dict_to_G(G_dict, N)
print(r_sum)
print(G)
'''
# print(convert_G_dict_to_G(G_dict, 3))
# G_d=construct_model_from_rates(r_dict)
# print(convert_G_dict_to_G(G_d, 3))


def get_CTMP_rates_from_results(results_dictionary_ddot: Dict[str, Dict[str, int]],
                                number_of_qubits: int):
    single_qubits = list(range(number_of_qubits))
    pairs = [(i, j) for i in range(number_of_qubits) for j in range(i + 1, number_of_qubits)]

    local_noise_matrices_CTMP = {pair: np.zeros((4, 4), dtype=float) for pair in pairs}

    for pair in pairs:
        pair_complement = list(set(single_qubits).difference(set(pair)))

        for global_input_state, results_dictionary in results_dictionary_ddot.items():
            marginal_input_state = ''.join([global_input_state[x] for x in pair])
            # print(marginal_input_state)
            input_state_complement = [global_input_state[x] for x in pair_complement]

            for global_output_state, ticks in results_dictionary.items():
                marginal_output_state = ''.join([global_output_state[x] for x in pair])
                output_state_complement = [global_output_state[x] for x in pair_complement]

                if output_state_complement == input_state_complement:
                    # check if this is their convention!
                    local_noise_matrices_CTMP[pair][int(marginal_output_state, 2),
                                                    int(marginal_input_state, 2)] += ticks

        # print(local_noise_matrices_CTMP[pair])
        # raise KeyError
    # normalize to stochastic matrices
    for pair in pairs:
        for k in range(4):
            if sum(local_noise_matrices_CTMP[pair][:, k]) != 0:
                local_noise_matrices_CTMP[pair][:, k] /= sum(local_noise_matrices_CTMP[pair][:, k])

    # Get G matrices
    # TODO: potential bug with logarithm branch
    # print(local_noise_matrices_CTMP[pair])
    G_matrices = {pair: sc.linalg.logm(local_noise_matrices_CTMP[pair]) for pair in pairs}

    # ancillary function
    def _chop_negatives(M):
        (m, n) = M.shape

        chopped_M = copy.deepcopy(M)
        for i in range(m):
            for j in range(n):
                if i != j and M[i, j] < 0:
                    chopped_M[i, j] = 0

        return chopped_M

    # Get G' matrices
    G_prime_matrices = {pair: _chop_negatives(G_matrices[pair]) for pair in pairs}

    rates_dictionary_2q = {f"q{pair[0]}q{pair[1]}": {'00': 0,
                                                     '01': 0,
                                                     '10': 0,
                                                     '11': 0} for pair in pairs}

    for pair in pairs:
        G_prime_matrix_now = G_prime_matrices[pair]
        rates_dictionary_2q[f"q{pair[0]}q{pair[1]}"]['00'] = G_prime_matrix_now[3, 0]
        rates_dictionary_2q[f"q{pair[0]}q{pair[1]}"]['01'] = G_prime_matrix_now[2, 1]
        rates_dictionary_2q[f"q{pair[0]}q{pair[1]}"]['10'] = G_prime_matrix_now[1, 2]
        rates_dictionary_2q[f"q{pair[0]}q{pair[1]}"]['11'] = G_prime_matrix_now[0, 3]

    rates_dictionary_1q = {f"q{q}": {'0': 0, '1': 0} for q in single_qubits}
    for qj in single_qubits:
        r0, r1 = 0, 0

        for q_other in list(set(single_qubits).difference(set([qj]))):
            G_prime_matrix_now = G_prime_matrices[tuple(sorted([qj, q_other]))]

            r0 += G_prime_matrix_now[2, 0] + G_prime_matrix_now[3, 1] \
                # +G_prime_matrix_now[1,0]+G_prime_matrix_now[3,2]

            r1 += G_prime_matrix_now[0, 2] + G_prime_matrix_now[1, 3] \
                # +G_prime_matrix_now[0, 1] + G_prime_matrix_now[2, 3]

        r0 /= 2 * (number_of_qubits - 1)
        r1 /= 2 * (number_of_qubits - 1)

        rates_dictionary_1q[f"q{qj}"]['0'] = r0
        rates_dictionary_1q[f"q{qj}"]['1'] = r1

    rates_dictionary = {**rates_dictionary_1q,
                        **rates_dictionary_2q}
    return rates_dictionary


def check_CTMP_consistency(no_of_qubits, bound, samples_per_input):
    r_dict = generate_random_r_dict(no_of_qubits, bound)
    G_dict = construct_model_from_rates(r_dict)

    lambda_par = get_r_sum_from_r_dict(r_dict)

    output = dict()

    for input in generate_binary_strings(no_of_qubits, []):
        ticks = dict()
        input_results = sample_from_bravyi_noise_model(input, samples_per_input, G_dict, lambda_par)

        for result in input_results:
            if result in ticks.keys():
                ticks[result] = ticks[result] + 1
            else:
                ticks[result] = 1

        output[input] = ticks

    r_dict_reconstr = get_CTMP_rates_from_results(output, no_of_qubits)
    return r_dict, r_dict_reconstr


number_of_samples = 10 ** 5
r_dict, r_dict_reconstr = check_CTMP_consistency(3, 0.05, number_of_samples)
print(r_dict)
print(r_dict_reconstr)

from qrem.functions_qrem import ancillary_functions as anf

print()
qprint("Number of samples:", number_of_samples, 'red')
for qubit_key in r_dict.keys():

    dict_now_true, dict_now_estimated = r_dict[qubit_key], r_dict_reconstr[qubit_key]

    qprint("QUBITS:", qubit_key, 'green')
    for rate_key in dict_now_true.keys():
        qprint("Rate:", rate_key, 'red')
        qprint("(True value, Estimated value, Difference)",
                       f"({np.round(dict_now_true[rate_key], 4)},"
                       f"{np.round(dict_now_estimated[rate_key], 4)}, "
                       f"{np.round(dict_now_true[rate_key] - dict_now_estimated[rate_key], 4)})")

    print()
