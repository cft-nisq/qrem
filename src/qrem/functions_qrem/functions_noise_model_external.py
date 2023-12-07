"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""

import copy
import math
import random
import qutip
import numpy as np
import scipy as sc
import re
from typing import Dict
import matplotlib.pyplot as plt

from qrem.functions_qrem import functions_data_analysis as fda



def l1_norm(v):
    sum = 0
    for x in v:
        sum += abs(x)
    return sum


def delta(v, w):
    val = 0
    if v == w:
        val = 1
    return val

#(PP) possible overlap with bitstrings
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
delta - delta parameter
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
delta=0.1
print(bravyi_algorithm_1(O,S,s,c,delta))
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
    for op_support_tuple in G_dict:
        operator = G_dict[op_support_tuple]
        if printing: print(operator)
        op_support_ordered = list(op_support_tuple)
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
                        y_supp_vals_ordered = "".join("%s" % y_supp_vals[qubit] for qubit in op_support_ordered)
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


'''
G_dict = dict()
r01_00 = 0.1
r01_11 = 0.05
r01_01 = 0.05

r12_00 = 0.1
r12_11 = 0.05
r12_01 = 0.05

# test_x=[[0,0,1],[0,0,0],[1,1,1]]
G_dict['q0q1'] = [[-r01_00, 0, 0, r01_11], [0, -r01_01, 0, 0], [0, r01_01, 0, 0], [r01_00, 0, 0, -r01_11]]
G_dict['q0q2'] = [[-r12_00, 0, 0, r12_11], [0, -r12_01, 0, 0], [0, r12_01, 0, 0], [r12_00, 0, 0, -r12_11]]

N = 3

x_set = generate_binary_strings(N, [])
y_set = generate_binary_strings(N - 1, [])
lambda_par = 0.4
'''
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


def binary(num, length):
    return format(num, '0{}b'.format(length))


def sample_from_B_naive(x, B):
    dim = len(B[0])
    n = int(math.log(dim, 2))
    B = np.array(B)
    # print(x)
    # supp_str=''.join(x)
    # col=int(supp_str,2)
    col = int(x, 2)
    # print(col)
    prob_vector = B[:, col]
    row = np.random.choice(range(dim), p=prob_vector)
    # print('q'+str(n))
    sample = binary(row, n)
    # print('s'+str(sample))
    return sample


def sample_from_B_pow_n_naive(n, x, B, printing=False):
    sample = x
    for step in range(n):
        if printing: print('pre ' + sample)
        sample = sample_from_B_naive(sample, B)
        if printing: print('post ' + sample)
    return str(sample)


# sample_from_B_pow_n(10, x_set[0], G_dict, lambda_par)

'''
O = {'q0q1': 10, 'q1': -1, 'q2' : -1, 'q3' : -1 }
s = ['0000'] * 50
s2 = ['1111'] * 50
s.extend(s2)
delta_par = 0.1
'''

'''
s - a dict with experimental results
O - a dict with keys - pauli string supports x and values - coeffs
delta - delta parameter
'''
#TODO: rewrite it to s- dict, not a list !!!
def bravyi_algorithm_1_cond_prob(s, O, G_dict, delta_par, lambda_par):
    c_l1_norm = math.exp(2 * lambda_par)
    M = len(s)  # number of experiments
    # print('M '+str(M))

    T = math.ceil(4 * (c_l1_norm ** 2) / (delta_par ** 2))
    print('T ' + str(T))
    sum_xi = 0
    for t in range(T):
        i = random.randrange(0, M)
        # print("i"+str(i))
        n = np.random.poisson(lambda_par)
        # print('n '+str(n))
        x = sample_from_B_pow_n(n, s[i], G_dict, lambda_par)
        print('x ' + x)

        if (n % 2) == 0:
            sign = 1.0
        else:
            sign = -1.0
        sum_xi += sign * fda.get_energy_from_bitstring_diagonal_local(x, O)
        print(fda.get_energy_from_bitstring_diagonal_local(x, O))
    return c_l1_norm * sum_xi / T

#TODO: finish this with Filip's sampling and s as a dict with counts, not a list
def bravyi_algorithm_1(s, O, G_dict, delta_par, lambda_par):
    c_l1_norm = math.exp(2 * lambda_par)
    M = len(s)  # number of experiments
    # print('M '+str(M))

    T = math.ceil(4 * (c_l1_norm ** 2) / (delta_par ** 2))
    print('T ' + str(T))
    sum_xi = 0
    for t in range(T):
        i = random.randrange(0, M)
        # print("i"+str(i))
        n = np.random.poisson(lambda_par)
        # print('n '+str(n))
        x = sample_from_B_pow_n(n, s[i], G_dict, lambda_par)
        print('x ' + x)

        if (n % 2) == 0:
            sign = 1.0
        else:
            sign = -1.0
        sum_xi += sign * fda.get_energy_from_bitstring_diagonal_local(x, O)
        print(fda.get_energy_from_bitstring_diagonal_local(x, O))
    return c_l1_norm * sum_xi / T


# print(bravyi_algorithm_1(s, O, G_dict, delta_par, lambda_par))

# y -bitstring
# N - no of samples
# G_dict - dict with local noise generators
# lambda_par - lambda parameter
def sample_from_bravyi_noise_model(y, N, G_dict, lambda_par):
    out = []
    for t in range(N):
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


def construct_model_from_rates(r_dict):
    local_G_dictionary = dict()


    for op_support_tuple in r_dict:
        no_of_qubits = int(len(op_support_tuple) )
        dim = int(math.pow(2, no_of_qubits))
        gen_total = np.zeros((dim, dim))
        for op_type in r_dict[op_support_tuple]:
            r = r_dict[op_support_tuple][op_type]
            if len(op_type) == 1:
                gen = return_1local_gen(op_type, r)
            elif len(op_type) == 2:
                gen = return_2local_gen(op_type, r)
            else:
                raise ValueError('Wrong generator dimension in r_dict!')
            gen_total = gen_total + gen
        local_G_dictionary[op_support_tuple] = gen_total
    return local_G_dictionary


# r_dict = {'q0q1': {'00': 10, '11': 9, '01': 8}, 'q0q2': {'00': 7, '11': 6, '01': 5}}


# print(sample_from_bravyi_noise_model(s, 10, G_dict, lambda_par))


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

def generate_random_r_dict(qubits_indices, bound):
    r_dict = dict()

    for qubit in qubits_indices:
        supp_1_local = (qubit,)
        r_dict_sub_1 = dict()
        for op_type_1 in ['0', '1']:
            r_dict_sub_1[op_type_1] = bound * np.random.random()
        r_dict[supp_1_local] = r_dict_sub_1

    for ind_1,qubit_1 in enumerate(qubits_indices):
        for qubit_2 in qubits_indices[ind_1+1:]:
            supp_2_local = (qubit_1, qubit_2)
            r_dict_sub_2 = dict()
            for op_type_2 in ['00', '01', '10', '11']:
                r_dict_sub_2[op_type_2] = bound * (np.random.random()+0.01)
            r_dict[supp_2_local] = r_dict_sub_2
    return r_dict

# q1 q2 ...
def convert_G_dict_to_G(G_dict, N):
    dim = int(math.pow(2, N))
    G = np.zeros((dim, dim))
    for gen_supp in G_dict:
        gen_supp_indices = list(gen_supp)
        operator = np.array(G_dict[gen_supp])
        # print(operator)
        operator_emb = embed_operator(N, operator, indices=gen_supp_indices)
        # print(np.shape(operator_emb))
        G = G + operator_emb
    return G

def convert_local_G_dict_to_local_G(G_dict, supp):
    N = len(supp)
    dim = int(math.pow(2, N))
    G = np.zeros((dim, dim))
    for gen_supp in G_dict:
        gen_supp_indices = [supp.index(pos) for pos in gen_supp]
        operator = np.array(G_dict[gen_supp])
        # print(operator)
        operator_emb = embed_operator(N, operator, indices=gen_supp_indices)
        # print(np.shape(operator_emb))
        G = G + operator_emb
    return G

def generate_random_G_dict(qubits_indices, bound):

    r_dict = generate_random_r_dict(qubits_indices, bound)
    G_dict = construct_model_from_rates(r_dict)

    # TODO OS: this can be done inside above function
    lambda_b = get_r_sum_from_r_dict(r_dict)
    return G_dict, lambda_b

def generate_random_local_G_dict(clusters, bound):

    local_G_dictionaries={}
    r_dict= {}

    local_G_matrices = {}

    for cluster in clusters:
        qubits_indices = list(cluster)
        #print(qubits_indices)
        r_dict_clust = generate_random_r_dict(qubits_indices, bound)
        #print(r_dict_clust)
        r_dict.update(r_dict_clust)
        G_dict_clust = construct_model_from_rates(r_dict_clust)
        #print(G_dict_clust)
        local_G_dictionaries[cluster]=G_dict_clust
        local_G_matrix = convert_local_G_dict_to_local_G(G_dict_clust, qubits_indices)
        local_G_matrices[cluster] = local_G_matrix


    return local_G_dictionaries,local_G_matrices, r_dict


'''
clusters=[(0,1,2),(3,4),(5,6,7,8),(9,),(10,)]
bound = 0.05
local_G_dicts,local_G_mat, r_dict=generate_random_local_G_dict(clusters, bound)



for G_dict in local_G_dicts:
    print("New G_dict")
    print(G_dict)
print(r_dict)

for G_mat in local_G_mat:
    print(G_mat)
    print("New G_mat")
    print(local_G_mat[G_mat])
'''

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
    G_dict, lambda_par = generate_random_G_dict([i for i in range(no_of_qubits)], bound)
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
    plt.title('Bravyi sampling pd for y=' + y + ', samples= ' + str(no_of_samples) + f', TVD= {TVD:.5f}')

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
            input_state_complement = [global_input_state[x] for x in pair_complement]

            for global_output_state, ticks in results_dictionary.items():
                marginal_output_state = ''.join([global_output_state[x] for x in pair])
                output_state_complement = [global_output_state[x] for x in pair_complement]

                if output_state_complement == input_state_complement:
                    # check if this is their convention!
                    local_noise_matrices_CTMP[pair][int(marginal_output_state, 2),
                                                    int(marginal_input_state, 2)] += ticks

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


def merge_local_to_global(local_dict, n):
    # print(local_dict)
    merged_list = np.empty(n, dtype='str')
    for supp in local_dict:
        val = local_dict[supp]
        for local_index, q in enumerate(supp):
            merged_list[int(q)] = val[local_index]
            # print(val[local_index])
    # print(merged_list)
    merged = ''.join(merged_list)
    return merged


def trim_str_to_supp_tuple(y, supp):
    y_trimmed = ''
    for q in supp:
        y_trimmed += y[q]
    return y_trimmed


# r_dict, r_dict_reconstr = check_CTMP_consistency(3, 0.05, 10000)
# print(r_dict)
# print(r_dict_reconstr)

# y -bitstring
# N - no of samples
# local_B_dict - dict with keys : local supports and keys : local models as tuples (gamma, B)
def sample_from_local_bravyi_noise_model(y, N, local_B_dict):
    no_of_qubits = len(y)
    out = []
    for t in range(N):
        sample = {}
        for supp in local_B_dict:
            gamma = (local_B_dict[supp])[0]
            B = (local_B_dict[supp])[1]
            n = np.random.poisson(gamma)
            # print('n ' + str(n))
            y_trimmed = trim_str_to_supp_tuple(y, supp)
            sample[supp] = sample_from_B_pow_n_naive(n, y_trimmed, B)
            # print(sample[supp])
        sample_merged = merge_local_to_global(sample, no_of_qubits)
        out.append(sample_merged)
    return out


#local_B_dict = {(0, 1): (10, [[0.9, 0, 0, 0.3], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0.1, 0, 0, 0.7]]),
#                (2, 3): (12, [[0, 0.5, 0.5, 0], [0.2, 0, 0, 0.4], [0.8, 0, 0, 0.6], [0, 0.5, 0.5, 0]])}
#y = '0011'
#N = 50
#print(sample_from_local_bravyi_noise_model(y, N, local_B_dict))


# supp=(0,2,3)
# print(trim_str_to_supp_tuple(y,supp))

def get_global_gamma_from_local_B_dict(local_B_dict):
    global_gamma = 0
    for cluster in local_B_dict:
        global_gamma += (local_B_dict[cluster])[0]
    return global_gamma


def bravyi_algorithm_1_local(s, O, local_B_dict, delta_par):
    no_of_qubits = len(s[0])
    gamma_par = get_global_gamma_from_local_B_dict(local_B_dict)
    c_l1_norm = math.exp(2 * gamma_par)
    M = len(s)  # number of experiments
    # print('M '+str(M))

    T = math.ceil(4 * (c_l1_norm ** 2) / (delta_par ** 2))
    print('T ' + str(T))
    sum_xi = 0
    for t in range(T):
        i = random.randrange(0, M)
        # print("i"+str(i))
        # n = np.random.poisson(lambda_par)
        # print('n '+str(n))
        sample = {}
        n_sum = 0
        for supp in local_B_dict:
            gamma = (local_B_dict[supp])[0]
            B = (local_B_dict[supp])[1]
            n = np.random.poisson(gamma)
            n_sum += n
            # print('n ' + str(n))
            y_trimmed = trim_str_to_supp_tuple(s[i], supp)
            sample[supp] = sample_from_B_pow_n_naive(n, y_trimmed, B)
            # print(sample[supp])
        x = merge_local_to_global(sample, no_of_qubits)

        #print('y ' + s[i])
        #print('x ' + x)

        if (n_sum % 2) == 0:
            sign = 1.0
        else:
            sign = -1.0
        sum_xi += sign * fda.get_energy_from_bitstring_diagonal_local(x, O)
        #print(fda.get_energy_from_bitstring_diagonal_local(x, O))
    return c_l1_norm * sum_xi / T

'''
local_B_dict = {(0, 1): (10, [[0.9, 0, 0, 0.3], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0.1, 0, 0, 0.7]]),
                (2, 3): (12, [[0, 0.5, 0.5, 0], [0.2, 0, 0, 0.4], [0.8, 0, 0, 0.6], [0, 0.5, 0.5, 0]])}
O = {'q0q1': 10, 'q1': -1, 'q2' : -1, 'q3' : -1 }
s = ['0000'] * 50
s2 = ['1111'] * 50
s.extend(s2)
delta_par = 0.1
print(bravyi_algorithm_1_local(s, O, local_B_dict, delta_par))
'''