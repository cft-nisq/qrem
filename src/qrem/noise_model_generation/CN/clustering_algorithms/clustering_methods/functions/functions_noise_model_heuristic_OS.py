"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""

import copy, math, random

import numpy as np
from tqdm import tqdm

from qrem.noise_visualization.functions_data_visualization import *


class InfinityException(Exception):
    """ Class for handling infinite value of cost function  """
    pass


class RejectionException(Exception):
    """ Class for handling rejection of all operations """
    pass


class MaxClusterSizeException(Exception):
    """ Class for handling wrong maximal allowed size of clusters """
    pass


def f_clust_sharp(C_size, C_maxsize):
    """
    Cluster size penalty function f_penalty with sharp threshold.
    f_penalty(C_size) = 0,          if C_size <= C_maxsize,
    f_penalty(C_size) = Infinity,   if C_size > C_maxsize.

    :param C_size: Size of a cluster.
    :type C_size: int

    :param C_maxsize: Maximal allowed size of a cluster (threshold).
    :type C_maxsize: int

    :return: value of the function.
    :rtype: int
    """

    val = 0
    if C_size > C_maxsize:
        raise InfinityException
    return val


def f_clust_square_sharp(C_size, C_maxsize):
    """
    Cluster size penalty function f_penalty with quadratic dependence on C_size and sharp threshold.
    f_penalty(C_size) = (C_size/C_maxsize)^2,   if C_size <= C_maxsize,
    f_penalty(C_size) = Infinity,               if C_size > C_maxsize.

    :param C_size: Size of a cluster.
    :type C_size: int

    :param C_maxsize: Maximal allowed size of a cluster (threshold).
    :type C_maxsize: int

    :return: value of the function.
    :rtype: float
    """

    if C_size > C_maxsize:
        raise InfinityException
    else:
        val = math.pow(C_size, 2)
    return val


def f_clust(C_size, C_maxsize):
    """
    Cluster size penalty function f_penalty with flat beginning, quadratic intermediate values (normalized by C_maxsize) and and sharp threshold.
    f_penalty(C_size) = 0,                      if C_size < 3,
    f_penalty(C_size) = (C_size/C_maxsize)^2,   if 3 <= C_size <= C_maxsize,
    f_penalty(C_size) = Infinity,               if C_size > C_maxsize.

    :param C_size: Size of a cluster.
    :type C_size: int

    :param C_maxsize: Maximal allowed size of a cluster (threshold).
    :type C_maxsize: int

    :return: value of the function.
    :rtype: float
    """

    if C_size > C_maxsize:
        raise InfinityException
    elif C_size < 3:
        val = 0
    else:
        val = math.pow(C_size / C_maxsize, 2)
    return val


def get_Sc_i(C_i, correlations_table):
    """
    Strength of correlaions S_i in cluster i.

    :param C_i: i-th cluster
    :type C_i: List[int]

    :param correlations_table: table of all 2-qubit correlation coefficients.
    :type correlations_table: ndarray(dtype=float, ndim=2)

    :return value of S_i.
    :rtype: float
    """

    if len(C_i) < 2:
        return 0
    val = 0
    for k in C_i:
        for l in C_i:
            if (k == l):
                continue
            val = val + correlations_table[k, l]
    return val


# uses f_clust_square_sharp
def objective_function_simple_cummulative(partition, correlations_table, alpha, C_maxsize):
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


# uses f_clust_square_sharp
def evaluate_move_operation_naive_cummulative(partition, index_k, index_C_i, index_C_j, correlations_table, alpha,
                                              C_maxsize):
    """
    Calculates the change of the objective function phi after specified move operation. Calculations are naive - objective function is calculated twice and the difference is taken. Can be used for debugging.

    :param partition: partition
    :type partition: List[int]

    :param index_k: qubit to be moved
    :type index_k: int

    :param index_C_i: index of a cluster with qubit index_k
    :type index_C_i: int

    :param index_C_j: index of a cluster where index_k is to be moved
    :type index_C_j: int

    :param correlations_table: table of all 2-qubit correlation coefficients.
    :type correlations_table: ndarray(dtype=float, ndim=2)

    :param alpha: multiplicative parameter of the cluster size penalty function.
    :type alpha: float

    :param C_maxsize: Maximal allowed size of a cluster (threshold).
    :type C_maxsize: int

    :return change of phi under operation, if the value is finite
    :rtype:(float, bool)
    """

    partition_copy = copy.deepcopy(partition)
    val1 = objective_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    make_move_operation(partition_copy, index_k, index_C_i, index_C_j)
    try:
        val2 = objective_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    except InfinityException:
        return 0, False
    diff = val2 - val1
    return diff, True


# uses f_clust_square_sharp
def evaluate_move_operation_cummulative(partition, index_k, index_C_i, index_C_j, correlations_table, alpha, C_maxsize):
    """
    Calculates the change of the objective function phi after specified move operation.

    :param partition: partition
    :type partition: List[int]

    :param index_k: qubit to be moved
    :type index_k: int

    :param index_C_i: index of a cluster with qubit index_k
    :type index_C_i: int

    :param index_C_j: index of a cluster where index_k is to be moved
    :type index_C_j: int

    :param correlations_table: table of all 2-qubit correlation coefficients.
    :type correlations_table: ndarray(dtype=float, ndim=2)

    :param alpha: multiplicative parameter of the cluster size penalty function.
    :type alpha: float

    :param C_maxsize: Maximal allowed size of a cluster (threshold).
    :type C_maxsize: int

    :return change of phi under operation, if the value is finite
    :rtype:(float, bool)
    """

    C_i = partition[index_C_i]
    C_j = partition[index_C_j]
    c_i = len(C_i)
    c_j = len(C_j)

    diff = 0
    # cluster function part
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


# uses f_clust_square_sharp
def evaluate_swap_operation_naive_cummulative(partition, index_k, index_l, index_C_i, index_C_j, correlations_table,
                                              alpha, C_maxsize):
    """
    Calculates the change of the objective function phi after specified swap operation. Calculations are naive - objective function is calculated twice and the difference is taken. Can be used for debugging.

    :param partition: partition
    :type partition: List[int]

    :param index_k: qubit to be swapped
    :type index_k: int

    :param index_l: qubit to be swapped
    :type index_l: int

    :param index_C_i: index of a cluster with qubit index_k
    :type index_C_i: int

    :param index_C_j: index of a cluste with qubit index_l
    :type index_C_j: int

    :param correlations_table: table of all 2-qubit correlation coefficients.
    :type correlations_table: ndarray(dtype=float, ndim=2)

    :param alpha: multiplicative parameter of the cluster size penalty function.
    :type alpha: float

    :param C_maxsize: Maximal allowed size of a cluster (threshold).
    :type C_maxsize: int

    :return change of phi under operation, if the value is finite
    :rtype:(float, bool)
    """

    partition_copy = copy.deepcopy(partition)
    val1 = objective_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    make_swap_operation(partition_copy, index_k, index_l, index_C_i, index_C_j)
    val2 = objective_function_simple_cummulative(partition_copy, correlations_table, alpha, C_maxsize)
    diff = val2 - val1
    return diff


def evaluate_swap_operation_cummulative(partition, index_k, index_l, index_C_i, index_C_j, correlations_table, alpha):
    """
   Calculates the change of the objective function phi after specified swap operation.

   :param partition: partition
   :type partition: List[int]

   :param index_k: qubit to be swapped
   :type index_k: int

   :param index_l: qubit to be swapped
   :type index_l: int

   :param index_C_i: index of a cluster with qubit index_k
   :type index_C_i: int

   :param index_C_j: index of a cluste with qubit index_l
   :type index_C_j: int

   :param correlations_table: table of all 2-qubit correlation coefficients.
   :type correlations_table: ndarray(dtype=float, ndim=2)

   :param alpha: multiplicative parameter of the cluster size penalty function.
   :type alpha: float

   :param C_maxsize: Maximal allowed size of a cluster (threshold).
   :type C_maxsize: int

   :return change of phi under operation, if the value is finite
   :rtype:(float, bool)
   """

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


def get_initial_partition(correlations_table):
    """
    Finds the initial partition based on thr correlatons table. Pairs of qubits with largest correlations are put to the same 2-qubit cluster. If number of qubits is odd, one qubit remains unpaired.

    :param correlations_table: table of all 2-qubit correlation coefficients.
    :type correlations_table: ndarray(dtype=float, ndim=2)

    :return: partition
    :rtype: List[List[int]]
    """

    n = correlations_table.shape[0]
    CL = dict()
    for i in range(n - 1):
        for j in range(i + 1, n):
            val1 = correlations_table[i, j]
            val2 = correlations_table[j, i]
            if (val1 > val2):
                val = val1
            else:
                val = val2
            CL.update({(i, j): val})
    partition = []
    qubits = list(range(n))
    while len(qubits) > 1:
        pair = max(CL, key=lambda key: CL[key])
        i = pair[0]
        j = pair[1]
        partition.append([i, j])
        keys = list(CL.keys())
        for pair in keys:
            if pair[0] in [i, j] or pair[1] in [i, j]:
                CL.pop(pair, None)
        qubits.remove(i)
        qubits.remove(j)
    if len(qubits) == 1:
        partition.append([qubits[0]])

    return partition


def return_cluster_index(partition, target_qubit):
    """
    Finds the index of a cluster in a partition in which is the target_qubit.

    :param partition: list of clusters
    :type partition: List[List[int]]

    :param target_qubit: qubit whose cluster's index we are looking
    :type target_qubit: int

    :return: cluster index in a partition.
    :rtype: int
    """

    index = 0
    for cluster in partition:
        for qubit in cluster:
            if qubit == target_qubit:
                return index
        index = index + 1
    return index


def make_move_operation(partition, index_k, index_C_i, index_C_j):
    """
    Realizes the move operation on partition, where qubit index_k is moved from cluster with index index_C_i to cluster with index index_C_j.

    :param partition: list of clusters
    :type partition: List[List[int]]

    :param index_k: qubit to be moved
    :type index_k: int

    :param index_C_i: index of a cluster with qubit index_k
    :type index_C_i: int

    :param index_C_j: index of a cluster where index_k is to be moved
    :type index_C_j: int
    """
    partition[index_C_j].append(index_k)

    if len(partition[index_C_i]) == 1:
        partition.pop(index_C_i)
    else:
        partition[index_C_i].remove(index_k)


def make_swap_operation(partition, index_k, index_l, index_C_i, index_C_j):
    """
    Realizes the swap operation on partition, where qubit index_k from cluster with index index_C_i is swapped with qubit index_l from cluster with index index_C_j

    :param partition: list of clusters
    :type partition: List[List[int]]

    :param index_k: qubit to be swapped
    :type index_k: int

    :param index_l: qubit to be swapped
    :type index_l: int

    :param index_C_i: index of a cluster with qubit index_k
    :type index_C_i: int

    :param index_C_j: index of a cluste with qubit index_l
    :type index_C_j: int
    """

    partition[index_C_i].remove(index_k)
    partition[index_C_i].append(index_l)
    partition[index_C_j].remove(index_l)
    partition[index_C_j].append(index_k)


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


# uses f_clust_square_sharp, objective_function_simple_cummulative, and non-naive move and swap cummulative evaluation operations
def partition_algorithm_v1_cummulative(correlations_table,
                                       alpha,
                                       C_maxsize,
                                       N_alg,
                                       printing,
                                       drawing, disable_pb):
    """
    Partition algorithm using move and swap operations on pairs of qubits with termination criterion.
    If C_maxsize < 2 then algorithm terminates with exception raised. If C_maxsize = 2 then the algorithm is not executed and the initial partition is returned.
    The algorith is run N_alg times and the best partition is chosen. In each run, the algorithm traverses pairs of qubits in epochs.
    An epoch consists of all pairs of qubits that are traversed at random and without repetitions.
    The partition is being updated during each epoch, by executing move and swap operations on pairs of qubits.
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
    init_cf = objective_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
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


# TODO OS: add optional convergence detection (termination after N_rand steps or if convergence detected). For example if a pair is rejected it goes to a list. If pair is accepted the list is erased. If list is full the algorithm terminates current attempt.
# uses f_clust_square_sharp, objective_function_simple_cummulative, and non-naive move and swap cummulative evaluation operations
def partition_algorithm_v2_cummulative(correlations_table,
                                       alpha,
                                       C_maxsize,
                                       N_alg,
                                       N_rand,
                                       printing,
                                       drawing, disable_pb):
    """
    Partition algorithm using move and swap operations on pairs of qubits at random.
    If C_maxsize < 2 then algorithm terminates with exception raised. If C_maxsize = 2 then the algorithm is not executed and the initial partition is returned.
    The algorith is run N_alg times and the best partition is chosen. In each run, the algorithm traverses pairs of qubits.
    Pairs of qubits are being traversed at random with repetitions and a total of N_rand pairs is chosen.
    The partition is being updated by executing move and swap operations on pairs of qubits.
    The algorithm terminates after N_runs. Each run terminates after considering N_rand pairs.

    :param correlations_table: table of all 2-qubit correlation coefficients.
    :type correlations_table: ndarray(dtype=float, ndim=2)

    :param alpha: multiplicative parameter of the cluster size penalty function.
    :type alpha: float

    :param C_maxsize: Maximal allowed size of a cluster (threshold).
    :type C_maxsize: int

    :param N_alg: Number of runs.
    :type N_alg: int
    
    :param N_rand: Number of pairs to be chosen at random on each run.
    :type N_rand: int

    :param printing: Is printing allowed.
    :type printing: bool

    :param drawing: Is cluster drawing allowed.
    :type drawing: bool

    :param disable_pb: Is progress bar disabled.
    :type disable_pb: bool

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
    init_cf = objective_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
    if (printing):
        print("initial value: " + str(init_cf))
    no_of_qubits = correlations_table.shape[0]
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
            best_cf = init_cf
            move_no = 0

            while move_no <= N_rand - 1:
                move_no = move_no + 1
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
                        best_cf = best_cf + val_of_ops[op]
                    elif (op == 'move_ji'):
                        index_C_i = return_cluster_index(partition, j)
                        index_C_j = return_cluster_index(partition, i)
                        index_k = j
                        make_move_operation(partition, index_k, index_C_i, index_C_j)
                        if (printing):
                            print("MOVE_ji")
                            print(partition)
                        best_cf = best_cf + val_of_ops[op]
                    else:
                        index_C_i = return_cluster_index(partition, i)
                        index_C_j = return_cluster_index(partition, j)
                        index_k = i
                        index_l = j
                        make_swap_operation(partition, index_k, index_l, index_C_i, index_C_j)
                        if (printing):
                            print("SWAP")
                            print(partition)
                        best_cf = best_cf + val_of_ops[op]
                elif (printing):
                    print("REJECT ALL")
            if (printing):
                print("Obtained: " + str(best_cf) + " after " + str(move_no) + " moves")
            if (best_cf > global_best_value):
                global_best_parition = partition
                global_best_value = best_cf

    global_best_parition_sorted = [sorted(cluster) for cluster in global_best_parition]
    return global_best_parition_sorted, global_best_value


# TODO OS: add optional convergence detection (termination after N_rand steps or if convergence detected)
# uses f_clust_square_sharp, objective_function_simple_cummulative, and non-naive move and swap cummulative evaluation operations
def partition_algorithm_v3_cummulative(correlations_table,
                                       alpha,
                                       C_maxsize,
                                       N_alg,
                                       N_rand,
                                       printing,
                                       drawing, disable_pb):
    """
   Partition algorithm using move operations on qubits at random.
   If C_maxsize < 2 then algorithm terminates with exception raised. If C_maxsize = 2 then the algorithm is not executed and the initial partition is returned.
   The algorith is run N_alg times and the best partition is chosen. In each run, the algorithm traverses qubits.
   Qubits are being traversed at random with repetitions and a total of N_rand qubits chosen.
   The partition is being updated by executing move operations on qubits.
   The algorithm terminates after N_runs. Each run terminates after considering N_rand qubits.

   :param correlations_table: table of all 2-qubit correlation coefficients.
   :type correlations_table: ndarray(dtype=float, ndim=2)

   :param alpha: multiplicative parameter of the cluster size penalty function.
   :type alpha: float

   :param C_maxsize: Maximal allowed size of a cluster (threshold).
   :type C_maxsize: int

   :param N_alg: Number of runs.
   :type N_alg: int

   :param N_rand: Number of qubits to be chosen at random on each run.
   :type N_rand: int

   :param printing: Is printing allowed.
   :type printing: bool

   :param drawing: Is cluster drawing allowed.
   :type drawing: bool

   :param disable_pb: Is progress bar disabled.
   :type disable_pb: bool

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
    init_cf = objective_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
    if (printing):
        print("initial value: " + str(init_cf))
    no_of_qubits = correlations_table.shape[0]
    global_best_parition = initial_partition
    global_best_value = init_cf

    if (skip_optimisation == False):

        qubits = range(no_of_qubits)

        for attempt_no in tqdm(range(N_alg), disable=disable_pb):
            if (printing):
                print("attempt: " + str(attempt_no + 1))
            partition = copy.deepcopy(initial_partition)
            best_cf = init_cf
            move_no = 0

            while move_no <= N_rand - 1:
                move_no = move_no + 1
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


# uses f_clust_square_sharp, objective_function_simple_cummulative, and non-naive move and swap cummulative evaluation operations
def partition_algorithm_v4_cummulative(correlations_table,
                                       alpha,
                                       C_maxsize,
                                       N_alg,
                                       printing,
                                       drawing, disable_pb):
    """
    Partition algorithm using move and swap operations on qubits with termination criterion.
    If C_maxsize < 2 then algorithm terminates with exception raised. If C_maxsize = 2 then the algorithm is not executed and the initial partition is returned.
    The algorith is run N_alg times and the best partition is chosen. In each run, the algorithm traverses qubits in epochs.
    An epoch consists of all qubits that are traversed at random and without repetitions.
    The partition is being updated during each epoch, by executing move and swap operations on qubits.
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
    init_cf = objective_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
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
    init_cf = objective_function_simple_cummulative(initial_partition, correlations_table, alpha, C_maxsize)
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
