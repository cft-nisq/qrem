"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""

from qrem.functions_qrem.functions_noise_model_heuristic import partition_algorithm_v1_cummulative
from qrem.functions_qrem.functions_noise_model_heuristic import partition_algorithm_v2_cummulative
from qrem.functions_qrem.functions_noise_model_heuristic import partition_algorithm_v3_cummulative
from qrem.functions_qrem.functions_noise_model_heuristic import partition_algorithm_v4_cummulative


class HeuristicClusteringAlgorithmBase:

    def __init__(self, version=None, correlations_table=None, alpha=None, C_maxsize=None, N_alg=None, N_rand=None,
                 printing=False, drawing=False, disable_pb=False):
        self.version = version
        self.correlations_table = correlations_table
        self.alpha = alpha
        self.C_maxsize = C_maxsize
        self.N_alg = N_alg
        self.N_rand = N_rand
        self.printing = printing
        self.drawing = drawing
        self.disable_pb = disable_pb
        self.global_best_parition_sorted = None
        self.global_best_value = None

    def clusterize(self, correlations_table=None, alpha=None):
        if correlations_table is not None:
            self.correlations_table = correlations_table
        elif self.correlations_table is None:
            raise ValueError("Correlations_table not specified.")
        if alpha is not None:
            self.alpha = alpha
        elif self.alpha is None:
            raise ValueError("Alpha not specified.")
        if self.C_maxsize is None:
            raise ValueError("C_maxsize not specified.")
        if self.N_alg is None:
            raise ValueError("N_alg not specified.")

        if self.version == 'v1':
            self.global_best_parition_sorted, self.global_best_value = partition_algorithm_v1_cummulative(
                self.correlations_table,
                self.alpha,
                self.C_maxsize,
                self.N_alg,
                self.printing,
                self.drawing, self.disable_pb)
        elif self.version == 'v2':
            if self.N_rand is None:
                raise ValueError("N_rand not specified.")

            self.global_best_parition_sorted, self.global_best_value = partition_algorithm_v2_cummulative(
                self.correlations_table,
                self.alpha,
                self.C_maxsize,
                self.N_alg,
                self.N_rand,
                self.printing,
                self.drawing, self.disable_pb)

        elif self.version == 'v3':
            if self.N_rand is None:
                raise ValueError("N_rand not specified.")

            self.global_best_parition_sorted, self.global_best_value = partition_algorithm_v3_cummulative(
                self.correlations_table,
                self.alpha,
                self.C_maxsize,
                self.N_alg,
                self.N_rand,
                self.printing,
                self.drawing, self.disable_pb)
        elif self.version == 'v4':

            self.global_best_parition_sorted, self.global_best_value = partition_algorithm_v4_cummulative(
                self.correlations_table,
                self.alpha,
                self.C_maxsize,
                self.N_alg,
                self.printing,
                self.drawing, self.disable_pb)
        else:
            raise ValueError("Implemented versions are: v1, v2, v3, v4")

        return self.global_best_parition_sorted, self.global_best_value
