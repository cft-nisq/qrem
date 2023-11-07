#JT: We have a separate file in functions_qrem where clustering algoithms are defined (functions_noise_model_heuristic file). Is this redundant?

#JT There are two heuristic_clustering_algorithm_base calsses (the other in cluststering_algorithms/clustering_methods/child_classes). Is this redundant

from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.functions.functions_noise_model_heuristic import partition_algorithm_v1_cummulative
from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.functions.functions_noise_model_heuristic import partition_algorithm_v2_cummulative
from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.functions.functions_noise_model_heuristic import partition_algorithm_v3_cummulative
from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.functions.functions_noise_model_heuristic import partition_algorithm_v4_cummulative

#JT Here we import a new clustering algorithm. The import is from qrem.functions_qrem, functions_noise_model_heuristic file (see above)
from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.functions.functions_noise_model_heuristic import partition_algorithm_v4_cummulative_temp


class HeuristicClusteringAlgorithmBase:

    def __init__(self,
                 version=None,
                 correlations_table=None,
                 alpha=None,
                 C_maxsize=None, N_alg=None, N_rand=None,
                 printing=False, drawing=False, disable_pb=None):

        if disable_pb is None:
            # if correlations_table.shape[0]>=100:
            #     disable_pb = False
            # else:
            disable_pb = True

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

        #JT: We add this to use probabilistic clustering algorithm

        elif self.version == 'v4_temp':

            self.global_best_parition_sorted, self.global_best_value = partition_algorithm_v4_cummulative_temp(
                self.correlations_table,
                self.alpha,
                self.C_maxsize,
                self.N_alg,
                self.printing,
                self.drawing, self.disable_pb,temp=1.0)

        else:
            raise ValueError("Implemented versions are: v1, v2, v3, v4")

        return self.global_best_parition_sorted, self.global_best_value
