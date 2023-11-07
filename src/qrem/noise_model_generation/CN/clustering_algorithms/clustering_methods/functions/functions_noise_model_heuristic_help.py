from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.functions.functions_noise_model_heuristic import *
from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.child_classes.heuristic_clustering_algorithm import HeuristicClusteringAlgorithmBase
import numpy as np


class AlphaScanError(Exception):
    """Class for handling situation where no primising plateaus are detected during alpha scan"""
    pass


def get_avg_correlation(correlations_table):
    sum = 0
    qubits = len(correlations_table)
    no_of_corr = qubits * qubits
    for row in correlations_table:
        for corr in row:
            sum += corr
    return sum / no_of_corr


# C_max has to be at least 2
def get_avg_large_corr(correlations_table, C_maxsize):
    qubits = correlations_table.shape[0]

    no_of_large_corrs = qubits * (C_maxsize - 1)
    flat = correlations_table.flatten()
    flat[::-1].sort(kind='heapsort')
    # print(flat)
    largest_corrs = flat[:no_of_large_corrs]

    sum = 0
    for corr in largest_corrs:
        sum += corr

    return sum / no_of_large_corrs


def compare_clusterings(clustering_1, clustering_2):
    isEqual = False

    los1 = set([frozenset(x) for x in clustering_1])
    los2 = set([frozenset(x) for x in clustering_2])

    # print(los1)
    # print(los2)

    if los1 == los2:
        isEqual = True
    return isEqual


'''
cl1=[[1,2],[3,4,5]]
cl2=[[4,3,5],[2,1]]
print(compare_clusterings(cl1,cl2))
'''


def plateaux_detection(hca: HeuristicClusteringAlgorithmBase):
    avg_large_corr = get_avg_large_corr(hca.correlations_table, hca.C_maxsize)
    # print(avg_large_corr)

    first = True
    c_old = []
    incr = 0.01
    stability_intervals = []
    on_plateaux = False
    for coeff in tqdm(np.arange(0, 2 + incr, incr)):
        alpha = coeff * avg_large_corr
        c, cost = hca.clusterize(alpha=alpha)
        if (not first):
            if (compare_clusterings(c, c_old)):
                if (not on_plateaux):
                    on_plateaux = True
                    plateaux_start = coeff - incr
                # print("Stability on v detected at: "+str(coeff))
                # print(c)
            else:
                if (on_plateaux):
                    on_plateaux = False
                    plateaux_end = coeff - incr
                    stability_intervals.append((plateaux_start, plateaux_end))

        c_old = c
        first = False

    return avg_large_corr, stability_intervals


def suggest_alpha_via_scan(hca: HeuristicClusteringAlgorithmBase, values_to_return=None):
    avg_large_corr, stability_intervals = plateaux_detection(hca)
    print("avg_large_corr: "+str(avg_large_corr)+"stability_intervals: "+str(stability_intervals))
    if (len(stability_intervals) < 2):
        raise AlphaScanError("No promising plateas detected")

    promising_plateaux = tuple([avg_large_corr*x for x in stability_intervals[1]])
    if values_to_return is None:
        return promising_plateaux
    else:
        alpha_min = promising_plateaux[0]
        alpha_max = promising_plateaux[1]
        interval = (alpha_max - alpha_min) / (values_to_return - 1)
        return np.arange(alpha_min, alpha_max + interval, interval)
