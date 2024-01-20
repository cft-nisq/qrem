"""
Created on 01.03.2021

@author: Oskar Słowik
@contact: osslowik@gmail.com
"""

import copy
import math
import random
import numpy as np

from tqdm import tqdm



class InfinityException(Exception):
    """Class for handling infinity"""
    pass


class ClusterSizeError(NameError):
    """Class for handling too small max cluster size"""
    pass

class RejectionException(Exception):
    """ Class for handling rejection of all operations """
    pass


# TODO OS: allow 2 qubit clusters but in this case skip the rest of algorithms since init is optimal
# cluster size function.
def f_clust_sharp(C_size, C_maxsize):
    # if(C_max<3):
    #    raise ClusterSizeError
    val = 0
    if C_size > C_maxsize:
        raise InfinityException
    return val


# cluster size function. Assumes C_max>=3. If not, raises exception.
def f_clust(C_size, C_maxsize):
    # if(C_max<3):
    #    raise ClusterSizeError
    val = 0
    if C_size > C_maxsize:
        raise InfinityException
    elif C_size < 3:
        val = 0
    else:
        val = math.pow(C_size / C_maxsize, 2)
    return val


# cluster size function. Assumes C_max>=3. If not, raises exception.
def f_clust_square_sharp(C_size, C_maxsize):
    # if(C_max<3):
    #    raise ClusterSizeError
    val = 0
    if C_size > C_maxsize:
        raise InfinityException
    else:
        val = math.pow(C_size, 2)
    return val


# average inter-cluster attractivness function S_{i,j}
def get_S_ij_av(C_i, C_j, correlations_table):
    val = 0
    for k in C_i:
        for l in C_j:
            val = val + (correlations_table[k, l] + correlations_table[l, k]) / (2 * len(C_i) * len(C_j))
    return val


# intra-cluster attractivness function S_{i}
def get_S_i(C_i, correlations_table):
    if len(C_i) < 2:
        return 0
    val = 0
    for k in C_i:
        for l in C_i:
            if (k == l):
                continue
            val = val + correlations_table[k, l] / (len(C_i) * (len(C_i) - 1))
    return val


# intra-cluster cumulative attractivness function Sc_{i}
def get_Sc_i(C_i, correlations_table):
    if len(C_i) < 2:
        return 0
    val = 0
    for k in C_i:
        for l in C_i:
            if (k == l):
                continue
            val = val + correlations_table[k, l]
    return val


def cost_function_simple_cummulative(partition, correlations_table, alpha, C_maxsize):
    """
    Calculates the value of the objective function phi on partition.

    :param partition: partition
    :type partition: List[int]

    :param correlations_table: table of all 2-qubit correlation coefficients.
    :type correlations_table: ndarray(dtype=float, ndim=2)

    :param alpha: multiplicative parameter of the cluster size penalty function.
    :type alpha: float

    :param C_maxsize: Maximal allowed size of a cluster (threshold).
    :type C_maxsize: int

    :return value of phi on partition.
    :rtype: float
    """

    no_of_clusters = len(partition)
    val = 0
    for C_i in partition:
        try:
            val = val - alpha * f_clust_square_sharp(len(C_i), C_maxsize)
        except InfinityException:
            raise InfinityException

    for i in range(no_of_clusters):
        C_i = partition[i]
        S_i = get_Sc_i(C_i, correlations_table)
        val = val + S_i
    return val

"""def cost_function_simple_cummulative(partition, correlations_table, alpha, C_maxsize):
    no_of_clusters = len(partition)
    val = 0
    for C_i in partition:
        try:
            val = val - alpha * f_clust_square_sharp(len(C_i), C_maxsize)
        except InfinityException:
            raise InfinityException

    for i in range(no_of_clusters):
        C_i = partition[i]
        S_i = get_Sc_i(C_i, correlations_table)
        val = val + S_i
    return val

"""
# returns the value of cost function (simpler version; no badness) and S table - a symmetric matrix which is defined by S_ij=S_ij_av (off-diagonal) and S_ii=S_i (diagonal). Raises error if minus infinity.
# assumes C_maxsize > 2
def cost_function_simple(partition, correlations_table, alpha, C_maxsize):
    no_of_clusters = len(partition)
    S = np.zeros((no_of_clusters, no_of_clusters))
    val = 0
    for C_i in partition:
        try:
            val = val - alpha * f_clust_square_sharp(len(C_i), C_maxsize)
        except:
            raise InfinityException

    for i in range(no_of_clusters):
        C_i = partition[i]
        S_i = get_S_i(C_i, correlations_table)
        S[i, i] = S_i
        val = val + S_i

    for i in range(no_of_clusters - 1):
        for j in range(i + 1, no_of_clusters):
            C_i = partition[i]
            C_j = partition[j]
            S_ij = get_S_ij_av(C_i, C_j, correlations_table)
            S[i, j] = S_ij
            S[j, i] = S_ij
            val = val - S_ij
    return val, S


def evaluate_move_operation_naive_cummulative(partition, index_k, index_C_i, index_C_j, correlations_table, alpha,
                                              C_maxsize):
    partition_copy = copy.deepcopy(partition)
    val1 = cost_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    make_move_operation(partition_copy, index_k, index_C_i, index_C_j)
    # print(S_1)
    try:
        val2 = cost_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    except InfinityException:
        return 0, False
    diff = val2 - val1
    return diff, True


def evaluate_move_operation_naive(partition, index_k, index_C_i, index_C_j, correlations_table, alpha, C_maxsize, S):
    partition_copy = copy.deepcopy(partition)
    val1, S_1 = cost_function_simple(partition_copy, correlations_table, alpha, C_maxsize)
    # print(S_1)
    if len(partition_copy[index_C_i]) < 2:
        print("deletion")
        S_1 = np.delete(S_1, index_C_i, 0)
        S_1 = np.delete(S_1, index_C_i, 1)

    make_move_operation(partition_copy, index_k, index_C_i, index_C_j)
    # print(S_1)
    try:
        val2, S_2 = cost_function_simple(partition_copy, correlations_table, alpha, C_maxsize)
    except InfinityException:
        return 0, False
    diff = val2 - val1
    dS = S_2 - S_1
    print(S_2)
    return diff, dS, True


def evaluate_swap_operation_naive_cummulative(partition, index_k, index_l, index_C_i, index_C_j, correlations_table,
                                              alpha, C_maxsize):
    partition_copy = copy.deepcopy(partition)
    val1 = cost_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    make_swap_operation(partition_copy, index_k, index_l, index_C_i, index_C_j)
    val2 = cost_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    diff = val2 - val1
    return diff


def evaluate_swap_operation_naive(partition, index_k, index_l, index_C_i, index_C_j, correlations_table, alpha,
                                  C_maxsize, S):
    partition_copy = copy.deepcopy(partition)
    val1, S_1 = cost_function_simple(partition_copy, correlations_table, alpha, C_maxsize)
    make_swap_operation(partition_copy, index_k, index_l, index_C_i, index_C_j)
    val2, S_2 = cost_function_simple(partition_copy, correlations_table, alpha, C_maxsize)
    diff = val2 - val1
    dS = S_2 - S_1
    return diff, dS


def evaluate_move_operation_cummulative(partition, index_k, index_C_i, index_C_j, correlations_table, alpha, C_maxsize):
    C_i = partition[index_C_i]
    C_j = partition[index_C_j]
    c_i = len(C_i)
    c_j = len(C_j)

    diff = 0
    # cluster function part
    dS_clust = 0
    try:
        dS_clust = f_clust_square_sharp(c_i, C_maxsize) + f_clust_square_sharp(c_j, C_maxsize) - f_clust_square_sharp(
            c_i - 1, C_maxsize) - f_clust_square_sharp(c_j + 1, C_maxsize)
        dS_clust = alpha * dS_clust
    except InfinityException:
        return diff, False

    # intra cluster part
    dS_intra = 0
    for index_q in C_j:
        dS_intra = dS_intra + correlations_table[index_q][index_k] + correlations_table[index_k][index_q]
    for index_q in C_i:
        if (index_q == index_k):
            continue
        dS_intra = dS_intra - correlations_table[index_q][index_k] - correlations_table[index_k][index_q]

    diff = dS_clust + dS_intra
    return diff, True


# returns the change of cost function cost_function_simple after a move operation (qubit k from C_i to C_j) on clusters.
def evaluate_move_operation(partition, index_k, index_C_i, index_C_j, correlations_table, alpha, C_maxsize, S):
    no_of_clusters = len(partition)
    dS = np.zeros((no_of_clusters, no_of_clusters))
    diff = 0
    C_i = partition[index_C_i]
    C_j = partition[index_C_j]
    c_i = len(C_i)
    c_j = len(C_j)

    # intra attraction part
    dS_intra = 0

    if c_i > 2:
        psum = 0
        for index_q in C_i:
            if index_q == index_k:
                continue
            psum = psum + correlations_table[index_k][index_q] + correlations_table[index_q][index_k]
        psum = -psum / ((c_i - 1) * (c_i - 2))
        psum = psum + (c_i * (c_i - 1) / ((c_i - 1) * (c_i - 2)) - 1) * S[index_C_i, index_C_i]
    else:
        psum = -S[index_C_i, index_C_i]

    dS[index_C_i, index_C_i] = psum
    dS_intra = dS_intra + psum

    psum = 0
    sum_k_q_C_j = 0
    for index_q in C_j:
        psum = psum + correlations_table[index_k][index_q] + correlations_table[index_q][index_k]
    sum_k_q_C_j = psum
    psum = psum / ((c_j + 1) * c_j)
    psum = psum + (c_j * (c_j - 1) / ((c_j + 1) * c_j) - 1) * S[index_C_j, index_C_j]
    dS[index_C_j, index_C_j] = psum
    dS_intra = dS_intra + psum

    # inter attraction part
    dS_inter = 0
    for index_C_m in range(len(partition)):
        if index_C_m in [index_C_i, index_C_j]:
            continue
        C_m = partition[index_C_m]
        c_m = len(C_m)

        psum = 0
        for index_q in C_m:
            psum = psum + correlations_table[index_k][index_q] + correlations_table[index_q][index_k]
        if c_i > 1:
            psum1 = (-1 / (2 * c_m)) * (1 / (c_i - 1)) * psum + (c_i * c_m / ((c_i - 1) * c_m) - 1) * S[
                index_C_i, index_C_m]
        else:
            psum1 = -S[index_C_i, index_C_m]
        dS[index_C_i, index_C_m] = psum1
        dS[index_C_m, index_C_i] = psum1
        psum2 = (1 / (2 * c_m)) * (1 / (c_j + 1)) * psum + (c_j * c_m / ((c_j + 1) * c_m) - 1) * S[index_C_j, index_C_m]
        dS[index_C_j, index_C_m] = psum2
        dS[index_C_m, index_C_j] = psum2
        dS_inter = dS_inter + psum1 + psum2

    psum = 0
    if c_i > 1:
        for index_q in C_i:
            if index_q == index_k:
                continue
            psum = psum + correlations_table[index_k][index_q] + correlations_table[index_q][index_k]
        psum = (1 / (2 * (c_i - 1) * (c_j + 1))) * (psum - sum_k_q_C_j) + (c_i * c_j / ((c_i - 1) * (c_j + 1)) - 1) * S[
            index_C_i, index_C_j]
    else:
        psum = -S[index_C_i, index_C_j]

    dS[index_C_i, index_C_j] = psum
    dS[index_C_j, index_C_i] = psum
    dS_inter = dS_inter + psum

    # cluster function part
    dS_clust = 0
    try:
        dS_clust = f_clust_square_sharp(len(C_i) - 1, C_maxsize) + f_clust_square_sharp(len(C_j) + 1,
                                                                                        C_maxsize) - f_clust_square_sharp(
            len(C_i),
            C_maxsize) - f_clust_square_sharp(
            len(C_j), C_maxsize)
        dS_clust = alpha * dS_clust
    except InfinityException:
        return diff, dS, False

    diff = dS_intra - dS_inter - dS_clust

    return diff, dS, True


def evaluate_swap_operation_cummulative(partition, index_k, index_l, index_C_i, index_C_j, correlations_table, alpha):
    C_i = partition[index_C_i]
    C_j = partition[index_C_j]

    diff = 0
    # intra cluster part
    dS_intra = 0
    for index_q in C_i:
        if (index_q == index_k):
            continue
        dS_intra = dS_intra + correlations_table[index_q][index_l] + correlations_table[index_l][index_q] - \
                   correlations_table[index_q][index_k] - correlations_table[index_k][index_q]
    for index_q in C_j:
        if (index_q == index_l):
            continue
        dS_intra = dS_intra + correlations_table[index_q][index_k] + correlations_table[index_k][index_q] - \
                   correlations_table[index_q][index_l] - correlations_table[index_l][index_q]
        diff = dS_intra
    return diff


# returns the change of cost function cost_function_simple after a swap operation (qubit k from C_i wilh qubit l from C_j) on clusters.
def evaluate_swap_operation(partition, index_k, index_l, index_C_i, index_C_j, correlations_table, alpha, C_maxsize, S):
    # cluster function part - contrubutes zero
    # intra attraction part
    no_of_clusters = len(partition)
    dS = np.zeros((no_of_clusters, no_of_clusters))
    dS_intra = 0
    C_i = partition[index_C_i]
    C_j = partition[index_C_j]
    c_i = len(C_i)
    c_j = len(C_j)

    if c_i > 1:
        psum = 0
        for index_q in C_i:
            if index_q == index_k:
                continue
            psum = psum + correlations_table[index_q][index_l] + correlations_table[index_l][index_q] - \
                   correlations_table[index_q][index_k] - correlations_table[index_k][index_q]
        psum = psum * (1 / (c_i * (c_i - 1)))
        dS[index_C_i, index_C_i] = psum
        dS_intra = dS_intra + psum

    if c_j > 1:
        psum = 0
        for index_q in C_j:
            if index_q == index_l:
                continue
            psum = psum + correlations_table[index_q][index_k] + correlations_table[index_k][index_q] - \
                   correlations_table[index_q][index_l] - correlations_table[index_l][index_q]
        dS[index_C_j][index_C_j] = psum
        psum = psum * (1 / (c_j * (c_j - 1)))
        dS_intra = dS_intra + psum

    # inter attraction part
    dS_inter = 0
    for index_C_m in range(len(partition)):
        if index_C_m in [index_C_i, index_C_j]:
            continue
        C_m = partition[index_C_m]
        c_m = len(C_m)

        psum = 0
        for index_q in C_m:
            psum = psum + correlations_table[index_l][index_q] + correlations_table[index_q][index_l] - \
                   correlations_table[index_k][index_q] - correlations_table[index_q][index_k]
        psum1 = (1 / (2 * c_m)) * (1 / c_i) * psum
        psum2 = (1 / (2 * c_m)) * (-1 / c_j) * psum
        dS[index_C_i, index_C_m] = psum1
        dS[index_C_m, index_C_i] = psum1
        dS[index_C_j, index_C_m] = psum2
        dS[index_C_m, index_C_j] = psum2
        dS_inter = dS_inter + psum1 + psum2

    psum = 0
    for index_q in C_j:
        if index_q == index_l:
            continue
        psum = psum + correlations_table[index_l][index_q] + correlations_table[index_q][index_l] - \
               correlations_table[index_k][index_q] - correlations_table[index_q][index_k]
    for index_q in C_i:
        if index_q == index_k:
            continue
        psum = psum - (correlations_table[index_l][index_q] + correlations_table[index_q][index_l] -
                       correlations_table[index_k][index_q] - correlations_table[index_q][index_k])
    psum = psum * (1 / (2 * c_i * c_j))
    dS[index_C_i, index_C_j] = psum
    dS[index_C_j, index_C_i] = psum
    dS_inter = dS_inter + psum

    diff = dS_intra - dS_inter
    return diff, dS


def get_initial_partition(correlations_table):
    n = correlations_table.shape[0]
    CL = dict()
    for i in range(n - 1):
        for j in range(i + 1, n):
            val = 0
            val1 = correlations_table[i, j]
            val2 = correlations_table[j, i]
            if (val1 > val2):
                val = val1
            else:
                val = val2
            CL.update({(i, j): val})
    # print(CL)
    partition = []
    qubits = list(range(n))
    while len(qubits) > 1:
        pair = max(CL, key=lambda key: CL[key])
        i = pair[0]
        j = pair[1]
        # print(i,j)
        partition.append([i, j])
        keys = list(CL.keys())
        # print(keys)
        for pair in keys:
            # print(str(pair)+"keys:"+str(i)+str(j))
            if pair[0] in [i, j] or pair[1] in [i, j]:
                # print("popping"+str(pair))
                CL.pop(pair, None)
        qubits.remove(i)
        qubits.remove(j)
    if len(qubits) == 1:
        partition.append([qubits[0]])

    return partition


def return_cluster_index(partition, target_qubit):
    index = 0
    for cluster in partition:
        for qubit in cluster:
            if qubit == target_qubit:
                return index
        index = index + 1
    return index


def make_move_operation(partition, index_k, index_C_i, index_C_j):
    # partition[index_C_i].remove(index_k)
    partition[index_C_j].append(index_k)

    if len(partition[index_C_i]) == 1:
        partition.pop(index_C_i)
    else:
        partition[index_C_i].remove(index_k)

    return


def make_swap_operation(partition, index_k, index_l, index_C_i, index_C_j):
    partition[index_C_i].remove(index_k)
    partition[index_C_i].append(index_l)
    partition[index_C_j].remove(index_l)
    partition[index_C_j].append(index_k)

    return




def choose_nonpos_op(val_of_ops_nonpos, temp):
    """
    Chooses the cluster index for which to move the qubit among all those resulting in nonpositive change of objective function phi.
    Probability of the choice of cluster i is:
    p(i)=exp(change_i/temp)/(no_of_ops_nonpos + epsilon)
    where no_of_ops_nonpos is the number of all operations and epsilon is small number added for numerical stability (here set as 0.001).
    Additionally, no operation can be selected with probability:
    p(reject)= 1- sum_j p(j).

    :param val_of_ops_nonpos: dictionary with keys being cluster indices and values being the changes of phi.
    :type val_of_ops_nonpos: dict

    :param temp: temperature parameter
    :type temp: float

    :return: chosen cluster index.
    :rtype: int
    """

    prob_dict = dict()
    no_of_ops_nonpos = len(val_of_ops_nonpos.keys())
    psum = 0
    for op in val_of_ops_nonpos:
        p = np.exp(val_of_ops_nonpos[op] / temp) / (no_of_ops_nonpos + 0.001)
        prob_dict[op] = p
        psum += p
    prob_dict['reject'] = 1.0 - psum
    chosen_op = random.choices(list(prob_dict.keys()), weights=prob_dict.values(), k=1)[0]
    if chosen_op == 'reject':
        raise RejectionException
    return chosen_op


# requires C_maxsize > 2 (otherwise terminates before execution)
# uses f_clust_square_sharp, cost_function_simple, and both naive non-naive move and swap evaluation (non-cummulative) operations (uses non-naive, naive used only for comparison)
def partition_algorithm_v1(correlations_table, alpha, C_maxsize, N_alg, printing, drawing):
    if (C_maxsize < 3):
        print("Error: max cluster size has to be at least 3!. Algorithm terminated.")
        return
    initial_partition = get_initial_partition(correlations_table)
    if (printing):
        print("initial partition:")
        print(initial_partition)
    if (drawing):
        print_partition(initial_partition)
    init_cf, init_S = cost_function_simple(initial_partition, correlations_table, alpha, C_maxsize)
    if (printing):
        print("initial value: " + str(init_cf))
    no_of_qubits = correlations_table.shape[0]
    # results=dict()
    global_best_parition = initial_partition
    global_best_value = init_cf
    for attempt_no in range(N_alg):
        if (printing):
            print("attempt: " + str(attempt_no + 1))
        partition = copy.deepcopy(initial_partition)
        best_cf = init_cf - 1
        curr_cf = init_cf
        curr_S = init_S
        epoch_no = 0
        while curr_cf > best_cf:
            epoch_no = epoch_no + 1
            best_cf = curr_cf
            if (printing):
                print("starting epoch: " + str(epoch_no))
            pairs = []
            for i in range(no_of_qubits - 1):
                for j in range(i + 1, no_of_qubits):
                    pairs.append([i, j])
            while len(pairs) > 0:
                val_of_ops = dict()
                pair = random.choice(pairs)
                pairs.remove(pair)
                i = pair[0]
                j = pair[1]
                if (printing):
                    print("pair: " + "(" + str(i) + ", " + str(j) + ")")
                index_C_i = return_cluster_index(partition, i)
                index_C_j = return_cluster_index(partition, j)
                if (index_C_i == index_C_j):
                    if (printing):
                        print("WRONG PAIR - SKIP")
                    continue
                index_k = i
                val1, dS1, not_neg_infty = evaluate_move_operation(partition, index_k, index_C_i, index_C_j,
                                                                   correlations_table, alpha, C_maxsize, curr_S)
                # val1_true, dS1_true, not_neg_infty_true = evaluate_move_operation_naive(partition, index_k, index_C_i,index_C_j, correlations_table, alpha, C_maxsize, curr_S)
                if (printing):
                    print("checking move " + str(index_C_i) + "--" + str(index_k) + "->" + str(
                        index_C_j) + "\t, result: " + "{:.5f}".format(val1) + "\t, size_leq_C_max: " + str(
                        not_neg_infty))
                if val1 > 0 and not_neg_infty:
                    if (printing):
                        print("update")
                    val_of_ops.update({'move_ij': val1})
                index_C_i = return_cluster_index(partition, j)
                index_C_j = return_cluster_index(partition, i)
                index_k = j
                val2, dS2, not_neg_infty = evaluate_move_operation(partition, index_k, index_C_i, index_C_j,
                                                                   correlations_table, alpha, C_maxsize, curr_S)
                # val2_true, dS2_true, not_neg_infty_true = evaluate_move_operation_naive(partition, index_k, index_C_i, index_C_j, correlations_table, alpha, C_maxsize, curr_S)
                if (printing):
                    print("checking move " + str(index_C_i) + "--" + str(index_k) + "->" + str(
                        index_C_j) + "\t, result: " + "{:.5f}".format(val2) + "\t, size_leq_C_max: " + str(
                        not_neg_infty))
                if val2 > 0 and not_neg_infty:
                    val_of_ops.update({'move_ji': val2})
                index_C_i = return_cluster_index(partition, i)
                index_C_j = return_cluster_index(partition, j)
                index_k = i
                index_l = j
                val3, dS3 = evaluate_swap_operation(partition, index_k, index_l, index_C_i, index_C_j,
                                                    correlations_table, alpha, C_maxsize, curr_S)
                val3_true, dS3_true = evaluate_swap_operation_naive(partition, index_k, index_l, index_C_i, index_C_j,
                                                                    correlations_table, alpha, C_maxsize, curr_S)
                if (printing):
                    print("checking swap " + str(index_C_i) + "<-" + str(index_k) + "&" + str(index_l) + "->" + str(
                        index_C_j) + "\t, result: " + "{:.5f}".format(val3) + "\t, size_leq_C_max: " + str(
                        not_neg_infty) + "\t, true_res:" + "{:.5f}".format(val3_true))
                if val3 > 0:
                    val_of_ops.update({'swap': val3})

                if len(val_of_ops) > 0:
                    if (printing):
                        print("ACCEPT:")
                    op = max(val_of_ops, key=lambda key: val_of_ops[key])
                    if (op == 'move_ij'):
                        dS = dS1
                        index_C_i = return_cluster_index(partition, i)
                        index_C_j = return_cluster_index(partition, j)
                        index_k = i
                        c_i = len(partition[index_C_i])
                        if (printing):
                            print("MOVE_ij")
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                        curr_S = curr_S + dS
                        if c_i < 2:
                            curr_S = np.delete(curr_S, index_C_i, 0)
                            curr_S = np.delete(curr_S, index_C_i, 1)
                    elif (op == 'move_ji'):
                        dS = dS2
                        index_C_i = return_cluster_index(partition, j)
                        index_C_j = return_cluster_index(partition, i)
                        index_k = j
                        c_i = len(partition[index_C_i])
                        if (printing):
                            print("MOVE_ji")
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                        curr_S = curr_S + dS
                        if c_i < 2:
                            curr_S = np.delete(curr_S, index_C_i, 0)
                            curr_S = np.delete(curr_S, index_C_i, 1)
                    else:
                        dS = dS3
                        index_C_i = return_cluster_index(partition, i)
                        index_C_j = return_cluster_index(partition, j)
                        index_k = i
                        index_l = j
                        if (printing):
                            print("SWAP")
                        make_swap_operation(partition, index_k, index_l, index_C_i, index_C_j)
                        print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                        curr_S = curr_S + dS
                else:
                    if (printing):
                        print("REJECT ALL")
        if (printing):
            print("convergence to: " + str(best_cf) + " after " + str(epoch_no) + " epochs")
        if (best_cf > global_best_value):
            global_best_parition = partition
            global_best_value = best_cf
        # results.update({partition:best_cf})
    # best_attempt=op=max(results, key=lambda key: results[key])
    return global_best_parition, global_best_value


# requires C_maxsize >= 2 (otherwise terminates before execution)
# uses f_clust_square_sharp, cost_function_simple_cummulative, and non-naive move and swap cummulative evaluation operations
def partition_algorithm_v1_cummulative(correlations_table,
                                       alpha,
                                       C_maxsize,
                                       N_alg,
                                       printing,
                                       drawing, disable_pb):
    skip_optimisation = False

    if (C_maxsize < 2):
        print("Error: max cluster size has to be at least 2!. Algorithm terminated.")
        return
    elif (C_maxsize == 2):
        skip_optimisation = True
    initial_partition = get_initial_partition(correlations_table)
    if (printing):
        print("initial partition:")
        print(initial_partition)
    if (drawing):
        print_partition(initial_partition)
    init_cf = cost_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
    if (printing):
        print("initial value: " + str(init_cf))
    no_of_qubits = correlations_table.shape[0]
    # results=dict()
    global_best_parition = initial_partition
    global_best_value = init_cf
    if (skip_optimisation == False):
        for attempt_no in tqdm(range(N_alg), disable=disable_pb):
            if (printing):
                print("attempt: " + str(attempt_no + 1))
            partition = copy.deepcopy(initial_partition)
            best_cf = init_cf - 1
            curr_cf = init_cf
            epoch_no = 0
            while curr_cf > best_cf:
                epoch_no = epoch_no + 1
                best_cf = curr_cf
                if (printing):
                    print("starting epoch: " + str(epoch_no))
                pairs = []
                for i in range(no_of_qubits - 1):
                    for j in range(i + 1, no_of_qubits):
                        pairs.append([i, j])
                while len(pairs) > 0:
                    val_of_ops = dict()
                    pair = random.choice(pairs)
                    pairs.remove(pair)
                    i = pair[0]
                    j = pair[1]
                    if (printing):
                        print("pair: " + "(" + str(i) + ", " + str(j) + ")")
                    index_C_i = return_cluster_index(partition, i)
                    index_C_j = return_cluster_index(partition, j)
                    if (index_C_i == index_C_j):
                        if (printing):
                            print("WRONG PAIR - SKIP")
                        continue
                    index_k = i
                    val1, not_neg_infty = evaluate_move_operation_cummulative(partition, index_k, index_C_i, index_C_j,
                                                                              correlations_table, alpha, C_maxsize)
                    # val1_true, not_neg_infty_true = evaluate_move_operation_naive_cummulative(partition, index_k, index_C_i, index_C_j, correlations_table, alpha, C_maxsize)

                    if (printing):
                        print("checking move " + str(index_C_i) + "--" + str(index_k) + "->" + str(
                            index_C_j) + "\t, result: " + "{:.5f}".format(val1) + "\t, size_leq_C_max: " + str(
                            not_neg_infty))
                    if val1 > 0 and not_neg_infty:
                        val_of_ops.update({'move_ij': val1})
                    index_C_i = return_cluster_index(partition, j)
                    index_C_j = return_cluster_index(partition, i)
                    index_k = j
                    val2, not_neg_infty = evaluate_move_operation_cummulative(partition, index_k, index_C_i, index_C_j,
                                                                              correlations_table, alpha, C_maxsize)
                    # val2_true, not_neg_infty_true = evaluate_move_operation_naive_cummulative(partition, index_k, index_C_i, index_C_j, correlations_table, alpha, C_maxsize)
                    if (printing):
                        print("checking move " + str(index_C_i) + "--" + str(index_k) + "->" + str(
                            index_C_j) + "\t, result: " + "{:.5f}".format(val2) + "\t, size_leq_C_max: " + str(
                            not_neg_infty))
                    if val2 > 0 and not_neg_infty:
                        val_of_ops.update({'move_ji': val2})
                    index_C_i = return_cluster_index(partition, i)
                    index_C_j = return_cluster_index(partition, j)
                    index_k = i
                    index_l = j
                    val3 = evaluate_swap_operation_cummulative(partition, index_k, index_l, index_C_i, index_C_j,
                                                               correlations_table, alpha)
                    # val3_true = evaluate_swap_operation_naive_cummulative(partition, index_k, index_l, index_C_i, index_C_j, correlations_table, alpha, C_maxsize)
                    if (printing):
                        print("checking swap " + str(index_C_i) + "<-" + str(index_k) + "&" + str(index_l) + "->" + str(
                            index_C_j) + "\t, result: " + "{:.5f}".format(val3) + "\t, size_leq_C_max: " + str(
                            not_neg_infty))
                    if val3 > 0:
                        val_of_ops.update({'swap': val3})

                    if len(val_of_ops) > 0:
                        if (printing):
                            print("ACCEPT:")
                        op = max(val_of_ops, key=lambda key: val_of_ops[key])
                        if (op == 'move_ij'):
                            index_C_i = return_cluster_index(partition, i)
                            index_C_j = return_cluster_index(partition, j)
                            index_k = i
                            make_move_operation(partition, index_k, index_C_i, index_C_j)
                            if (printing):
                                print("MOVE_ij")
                                print(partition)
                            curr_cf = curr_cf + val_of_ops[op]
                        elif (op == 'move_ji'):
                            index_C_i = return_cluster_index(partition, j)
                            index_C_j = return_cluster_index(partition, i)
                            index_k = j
                            make_move_operation(partition, index_k, index_C_i, index_C_j)
                            if (printing):
                                print("MOVE_ji")
                                print(partition)
                            curr_cf = curr_cf + val_of_ops[op]
                        else:
                            index_C_i = return_cluster_index(partition, i)
                            index_C_j = return_cluster_index(partition, j)
                            index_k = i
                            index_l = j
                            make_swap_operation(partition, index_k, index_l, index_C_i, index_C_j)
                            if (printing):
                                print("SWAP")
                                print(partition)
                            curr_cf = curr_cf + val_of_ops[op]
                    elif (printing):
                        print("REJECT ALL")
            if (printing):
                print("convergence to: " + str(best_cf) + " after " + str(epoch_no) + " epochs")
            if (best_cf > global_best_value):
                global_best_parition = partition
                global_best_value = best_cf

    global_best_parition_sorted = [sorted(cluster) for cluster in global_best_parition]
    return global_best_parition_sorted, global_best_value


# requires C_maxsize >= 2 (otherwise terminates before execution)
# uses f_clust_square_sharp, cost_function_simple_cummulative, and non-naive move and swap cummulative evaluation operations
# TODO OS: add optional convergence detection (termination after N_rand steps or if convergence detected). For example if a pair is rejected it goes to a list. If pair is accepted the list is erased. If list is full the algorithm terminates current attempt.
def partition_algorithm_v2_cummulative(correlations_table,
                                       alpha,
                                       C_maxsize,
                                       N_alg,
                                       N_rand,
                                       printing,
                                       drawing, disable_pb):
    skip_optimisation = False

    if (C_maxsize < 2):
        print("Error: max cluster size has to be at least 2!. Algorithm terminated.")
        return
    elif (C_maxsize == 2):
        skip_optimisation = True
    initial_partition = get_initial_partition(correlations_table)
    if (printing):
        print("initial partition:")
        print(initial_partition)
    if (drawing):
        print_partition(initial_partition)
    init_cf = cost_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
    if (printing):
        print("initial value: " + str(init_cf))
    no_of_qubits = correlations_table.shape[0]
    # results=dict()
    global_best_parition = initial_partition
    global_best_value = init_cf

    if (skip_optimisation == False):

        pairs = []
        for i in range(no_of_qubits - 1):
            for j in range(i + 1, no_of_qubits):
                pairs.append([i, j])

        for attempt_no in tqdm(range(N_alg), disable=disable_pb):
            if (printing):
                print("attempt: " + str(attempt_no + 1))
            partition = copy.deepcopy(initial_partition)
            best_cf = init_cf - 1
            curr_cf = init_cf
            move_no = 0

            while move_no <= N_rand - 1:
                move_no = move_no + 1
                best_cf = curr_cf
                if (printing):
                    print("starting move: " + str(move_no))

                val_of_ops = dict()
                pair = random.choice(pairs)
                i = pair[0]
                j = pair[1]
                if (printing):
                    print("pair: " + "(" + str(i) + ", " + str(j) + ")")
                index_C_i = return_cluster_index(partition, i)
                index_C_j = return_cluster_index(partition, j)
                if (index_C_i == index_C_j):
                    if (printing):
                        print("WRONG PAIR - SKIP")
                    continue
                index_k = i
                val1, not_neg_infty = evaluate_move_operation_cummulative(partition, index_k, index_C_i, index_C_j,
                                                                          correlations_table, alpha, C_maxsize)
                # val1_true, not_neg_infty_true = evaluate_move_operation_naive_cummulative(partition, index_k, index_C_i, index_C_j, correlations_table, alpha, C_maxsize)

                if (printing):
                    print("checking move " + str(index_C_i) + "--" + str(index_k) + "->" + str(
                        index_C_j) + "\t, result: " + "{:.5f}".format(val1) + "\t, size_leq_C_max: " + str(
                        not_neg_infty))
                if val1 > 0 and not_neg_infty:
                    val_of_ops.update({'move_ij': val1})
                index_C_i = return_cluster_index(partition, j)
                index_C_j = return_cluster_index(partition, i)
                index_k = j
                val2, not_neg_infty = evaluate_move_operation_cummulative(partition, index_k, index_C_i, index_C_j,
                                                                          correlations_table, alpha, C_maxsize)
                # val2_true, not_neg_infty_true = evaluate_move_operation_naive_cummulative(partition, index_k, index_C_i, index_C_j, correlations_table, alpha, C_maxsize)
                if (printing):
                    print("checking move " + str(index_C_i) + "--" + str(index_k) + "->" + str(
                        index_C_j) + "\t, result: " + "{:.5f}".format(val2) + "\t, size_leq_C_max: " + str(
                        not_neg_infty))
                if val2 > 0 and not_neg_infty:
                    val_of_ops.update({'move_ji': val2})
                index_C_i = return_cluster_index(partition, i)
                index_C_j = return_cluster_index(partition, j)
                index_k = i
                index_l = j
                val3 = evaluate_swap_operation_cummulative(partition, index_k, index_l, index_C_i, index_C_j,
                                                           correlations_table, alpha)
                # val3_true = evaluate_swap_operation_naive_cummulative(partition, index_k, index_l, index_C_i, index_C_j, correlations_table, alpha, C_maxsize)
                if (printing):
                    print("checking swap " + str(index_C_i) + "<-" + str(index_k) + "&" + str(index_l) + "->" + str(
                        index_C_j) + "\t, result: " + "{:.5f}".format(val3) + "\t, size_leq_C_max: " + str(
                        not_neg_infty))
                if val3 > 0:
                    val_of_ops.update({'swap': val3})

                if len(val_of_ops) > 0:
                    if (printing):
                        print("ACCEPT:")
                    op = max(val_of_ops, key=lambda key: val_of_ops[key])
                    if (op == 'move_ij'):
                        index_C_i = return_cluster_index(partition, i)
                        index_C_j = return_cluster_index(partition, j)
                        index_k = i
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        if (printing):
                            print("MOVE_ij")
                            print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                    elif (op == 'move_ji'):
                        index_C_i = return_cluster_index(partition, j)
                        index_C_j = return_cluster_index(partition, i)
                        index_k = j
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        if (printing):
                            print("MOVE_ji")
                            print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                    else:
                        index_C_i = return_cluster_index(partition, i)
                        index_C_j = return_cluster_index(partition, j)
                        index_k = i
                        index_l = j
                        make_swap_operation(partition, index_k, index_l, index_C_i, index_C_j)
                        if (printing):
                            print("SWAP")
                            print(partition)
                        curr_cf = curr_cf + val_of_ops[op]
                elif (printing):
                    print("REJECT ALL")
            if (printing):
                print("Obtained: " + str(best_cf) + " after " + str(move_no) + " moves")
            if (best_cf > global_best_value):
                global_best_parition = partition
                global_best_value = best_cf

    global_best_parition_sorted = [sorted(cluster) for cluster in global_best_parition]
    return global_best_parition_sorted, global_best_value


# requires C_maxsize >= 2 (otherwise terminates before execution)
# uses f_clust_square_sharp, cost_function_simple_cummulative, and non-naive move and swap cummulative evaluation operations
# TODO OS: add optional convergence detection (termination after N_rand steps or if convergence detected)
def partition_algorithm_v3_cummulative(correlations_table,
                                       alpha,
                                       C_maxsize,
                                       N_alg,
                                       N_rand,
                                       printing,
                                       drawing, disable_pb):
    skip_optimisation = False

    if (C_maxsize < 2):
        print("Error: max cluster size has to be at least 2!. Algorithm terminated.")
        return
    elif (C_maxsize == 2):
        skip_optimisation = True
    initial_partition = get_initial_partition(correlations_table)
    if (printing):
        print("initial partition:")
        print(initial_partition)
    if (drawing):
        print_partition(initial_partition)
    init_cf = cost_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
    if (printing):
        print("initial value: " + str(init_cf))
    no_of_qubits = correlations_table.shape[0]
    # results=dict()
    global_best_parition = initial_partition
    global_best_value = init_cf

    if (skip_optimisation == False):

        qubits = range(no_of_qubits)

        for attempt_no in tqdm(range(N_alg), disable=disable_pb):
            if (printing):
                print("attempt: " + str(attempt_no + 1))
            partition = copy.deepcopy(initial_partition)
            # best_cf = init_cf - 1
            best_cf = init_cf
            move_no = 0

            while move_no <= N_rand - 1:
                move_no = move_no + 1
                # best_cf = curr_cf
                if (printing):
                    print("starting move: " + str(move_no))

                val_of_ops = dict()
                i = random.choice(qubits)
                if (printing):
                    print("selected qubit: " + str(i))
                index_C_i = return_cluster_index(partition, i)
                index_k = i
                for index_C_j in range(len(partition)):
                    if index_C_j == index_C_i:
                        continue

                    val, not_neg_infty = evaluate_move_operation_cummulative(partition, index_k, index_C_i, index_C_j,
                                                                             correlations_table, alpha, C_maxsize)

                    if (printing):
                        print("checking move of qubit " + str(i) + " to partition " + str(
                            index_C_j) + "\t, result: " + "{:.5f}".format(val) + "\t, size_leq_C_max: " + str(
                            not_neg_infty))
                    if val > 0 and not_neg_infty:
                        val_of_ops.update({index_C_j: val})
                if len(val_of_ops) > 0:
                    best_cluster_index = max(val_of_ops, key=lambda key: val_of_ops[key])
                    best_cf = best_cf + val_of_ops[best_cluster_index]
                    index_C_j = best_cluster_index
                    make_move_operation(partition, index_k, index_C_i, index_C_j)
                    if (printing):
                        print("ACCEPT move to cluster no " + str(best_cluster_index) + ", value=" + str(best_cf))
                        print(partition)
                elif (printing):
                    print("REJECT ALL")

            if (printing):
                print("Obtained: " + str(best_cf) + " after move " + str(move_no))
            if (best_cf > global_best_value):
                global_best_parition = partition
                global_best_value = best_cf

    global_best_parition_sorted = [sorted(cluster) for cluster in global_best_parition]
    return global_best_parition_sorted, global_best_value

#
#We use this algorithm, maybe we just need to keep this
#
def partition_algorithm_v4_cummulative(correlations_table,
                                       alpha,
                                       C_maxsize,
                                       N_alg,
                                       printing,
                                       drawing, disable_pb):
    skip_optimisation = False

    if (C_maxsize < 2):
        print("Error: max cluster size has to be at least 2!. Algorithm terminated.")
        return
    elif (C_maxsize == 2):
        skip_optimisation = True
    initial_partition = get_initial_partition(correlations_table)
    if (printing):
        print("initial partition:")
        print(initial_partition)
    if (drawing):
        print_partition(initial_partition)
    init_cf = cost_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
    if (printing):
        print("initial value: " + str(init_cf))
    no_of_qubits = correlations_table.shape[0]
    # results=dict()
    global_best_parition = initial_partition
    global_best_value = init_cf
    if (skip_optimisation == False):
        for attempt_no in tqdm(range(N_alg), disable=disable_pb):
            if (printing):
                print("attempt: " + str(attempt_no + 1))
            partition = copy.deepcopy(initial_partition)
            best_cf = init_cf - 1
            curr_cf = init_cf
            epoch_no = 0
            while curr_cf > best_cf:
                epoch_no = epoch_no + 1
                best_cf = curr_cf
                if (printing):
                    print("starting epoch: " + str(epoch_no))
                qubits = list(range(no_of_qubits))

                while len(qubits) > 0:
                    val_of_ops = dict()
                    i = random.choice(qubits)
                    qubits.remove(i)

                    if (printing):
                        print("selected qubit: " + str(i))
                    index_C_i = return_cluster_index(partition, i)
                    index_k = i
                    for index_C_j in range(len(partition)):
                        if index_C_j == index_C_i:
                            continue

                        val, not_neg_infty = evaluate_move_operation_cummulative(partition, index_k, index_C_i,
                                                                                 index_C_j, correlations_table, alpha,
                                                                                 C_maxsize)

                        if (printing):
                            print("checking move of qubit " + str(i) + " to partition " + str(
                                index_C_j) + "\t, result: " + "{:.5f}".format(val) + "\t, size_leq_C_max: " + str(
                                not_neg_infty))
                        if val > 0 and not_neg_infty:
                            val_of_ops.update({index_C_j: val})
                    if len(val_of_ops) > 0:
                        best_cluster_index = max(val_of_ops, key=lambda key: val_of_ops[key])
                        curr_cf = curr_cf + val_of_ops[best_cluster_index]
                        index_C_j = best_cluster_index
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        if (printing):
                            print("ACCEPT move to cluster no " + str(best_cluster_index) + ", value=" + str(curr_cf))
                            print(partition)
                    elif (printing):
                        print("REJECT ALL")

            if (printing):
                print("convergence to: " + str(best_cf) + " after " + str(epoch_no) + " epochs")
            if (best_cf > global_best_value):
                global_best_parition = partition
                global_best_value = best_cf

    global_best_parition_sorted = [sorted(cluster) for cluster in global_best_parition]
    return global_best_parition_sorted, global_best_value



# uses f_clust_square_sharp, objective_function_simple_cummulative, and non-naive move and swap cummulative evaluation operations
def partition_algorithm_v4_cummulative_temp(correlations_table,
                                            alpha,
                                            C_maxsize,
                                            N_alg,
                                            printing,
                                            drawing, disable_pb, temp=1.0):
    """
    Partition algorithm using move operations on qubits with termination criterion and metropolis-like acceptance condition (regulated by temp).
    If C_maxsize < 2 then algorithm terminates with exception raised. If C_maxsize = 2 then the algorithm is not executed and the initial partition is returned.
    The algorith is run N_alg times and the best partition is chosen. In each run, the algorithm traverses qubits in epochs.
    An epoch consists of all qubits that are traversed at random and without repetitions.
    The partition is being updated during each epoch, by executing move operations on qubits.
    The algorithm terminates, if after some epoch the value of objective function phi has not improved compared to the end of a previous epoch.

    :param correlations_table: table of all 2-qubit correlation coefficients.
    :type correlations_table: ndarray(dtype=float, ndim=2)

    :param alpha: multiplicative parameter of the cluster size penalty function.
    :type alpha: float

    :param C_maxsize: Maximal allowed size of a cluster (threshold).
    :type C_maxsize: int

    :param N_alg: Number of runs.
    :type N_alg: int

    :param printing: Is printing allowed.
    :type printing: bool

    :param drawing: Is cluster drawing allowed.
    :type drawing: bool

    :param disable_pb: Is progress bar disabled.
    :type disable_pb: bool

    :param temp: Temperture parameter (default is 1.0).
    :type temp: float

    :return best partition found, value of the objective function for this partition
    :rtype:(List[int], float)

    :raises MaxClusterSizeException: returned if max allowed cluster size is < 2
    """

    skip_optimisation = False

    if (C_maxsize < 2):
        print("Error: max cluster size has to be at least 2!. Algorithm terminated.")
        raise MaxClusterSizeException
    elif (C_maxsize == 2):
        skip_optimisation = True
    initial_partition = get_initial_partition(correlations_table)
    if (printing):
        print("initial partition:")
        print(initial_partition)
    if (drawing):
        print_partition(initial_partition)
    init_cf = cost_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
    if (printing):
        print("initial value: " + str(init_cf))
    no_of_qubits = correlations_table.shape[0]
    global_best_parition = initial_partition
    global_best_value = init_cf
    if (skip_optimisation == False):
        for attempt_no in tqdm(range(N_alg), disable=disable_pb):
            if (printing):
                print("attempt: " + str(attempt_no + 1))
            partition = copy.deepcopy(initial_partition)
            best_cf = init_cf - 1
            curr_cf = init_cf
            epoch_no = 0
            while curr_cf > best_cf:
                epoch_no = epoch_no + 1
                best_cf = curr_cf
                if (printing):
                    print("starting epoch: " + str(epoch_no))
                qubits = list(range(no_of_qubits))

                while len(qubits) > 0:
                    val_of_ops_pos = dict()
                    val_of_ops_nonpos = dict()
                    i = random.choice(qubits)
                    qubits.remove(i)

                    if (printing):
                        print("selected qubit: " + str(i))
                    index_C_i = return_cluster_index(partition, i)
                    index_k = i
                    for index_C_j in range(len(partition)):
                        if index_C_j == index_C_i:
                            continue

                        val, not_neg_infty = evaluate_move_operation_cummulative(partition, index_k, index_C_i,
                                                                                 index_C_j, correlations_table, alpha,
                                                                                 C_maxsize)

                        if (printing):
                            print("checking move of qubit " + str(i) + " to partition " + str(
                                index_C_j) + "\t, result: " + "{:.5f}".format(val) + "\t, size_leq_C_max: " + str(
                                not_neg_infty))
                        if val > 0 and not_neg_infty:
                            val_of_ops_pos.update({index_C_j: val})
                        elif val <= 0 and not_neg_infty:
                            val_of_ops_nonpos.update({index_C_j: val})
                    if len(val_of_ops_pos) > 0:
                        chosen_cluster_index = choose_pos_op(val_of_ops_pos, temp)
                        curr_cf = curr_cf + val_of_ops_pos[chosen_cluster_index]
                        index_C_j = chosen_cluster_index
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        if (printing):
                            print("ACCEPT move to cluster no " + str(chosen_cluster_index) + ", value=" + str(curr_cf))
                            print(partition)
                    else:
                        try:
                            chosen_cluster_index = choose_nonpos_op(val_of_ops_nonpos, temp)
                            curr_cf = curr_cf + val_of_ops_nonpos[chosen_cluster_index]
                            index_C_j = chosen_cluster_index
                            make_move_operation(partition, index_k, index_C_i, index_C_j)
                            if (printing):
                                print("ACCEPT move to cluster no " + str(chosen_cluster_index) + ", value=" + str(
                                    curr_cf))
                                print(partition)
                        except RejectionException:
                            if (printing):
                                print("REJECT ALL")

            if (printing):
                print("convergence to: " + str(best_cf) + " after " + str(epoch_no) + " epochs")
            if (best_cf > global_best_value):
                global_best_parition = partition
                global_best_value = best_cf

    global_best_parition_sorted = [sorted(cluster) for cluster in global_best_parition]
    return global_best_parition_sorted, global_best_value

def choose_pos_op(val_of_ops_pos, temp):
    """
    Chooses the cluster index for which to move the qubit among all those resulting in positive change of objective function phi.
    Probability of the choice of cluster i is:
    p(i)=exp((change_i-max_change)/temp)/norm_const
    where norm_const is the normalization constant: sum_j exp((change_j-max_change)/temp.

    :param val_of_ops_pos: dictionary with keys being cluster indices and values being the changes of phi.
    :type val_of_ops_pos: dict

    :param temp: temperature parameter
    :type temp: float

    :return: chosen cluster index.
    :rtype: int
    """

    prob_dict = dict()
    values = np.array(list(val_of_ops_pos.values()))
    max_value = np.max(values)
    exp_args = (values - max_value) / temp
    exp_vals = np.exp(exp_args)
    norm_const = np.sum(exp_vals)
    for op in val_of_ops_pos:
        prob_dict[op] = np.exp((val_of_ops_pos[op] - max_value) / temp) / norm_const
    chosen_op = random.choices(list(prob_dict.keys()), weights=prob_dict.values(), k=1)[0]
    return chosen_op