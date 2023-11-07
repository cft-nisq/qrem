"""
Created on 23.08.2021

@author: Oskar Słowik
@contact: osslowik@gmail.com
"""
from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.child_classes.heuristic_clustering_algorithm_base import HeuristicClusteringAlgorithmBase
from qrem.noise_model_generation.CN.clustering_algorithms.clustering_methods.functions.functions_noise_model_heuristic_help import get_avg_large_corr, suggest_alpha_via_scan


#JT class used when performing benchmarks to set clustering algorithm, probably can be joined with HeuristicClusteringAlgorithmBase
class HeuristicClusteringAlgorithm(HeuristicClusteringAlgorithmBase):

    def __init__(self, **kwargs):
        super(HeuristicClusteringAlgorithm, self).__init__(**kwargs)

    def suggest_alpha_heuristic(self):
        if self.correlations_table is None:
            raise ValueError("correlations_table not specified.")
        if self.C_maxsize is None:
            raise ValueError("C_maxsize not specified.")
        self.alpha = 0.5 * get_avg_large_corr(self.correlations_table, self.C_maxsize)

    def suggest_alpha_scan(self, values_to_return=None):
        self.alpha = suggest_alpha_via_scan(self, values_to_return)
