"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""

from base_classes.heuristic_clustering_algorithm_base import HeuristicClusteringAlgorithmBase
from qrem.functions_qrem.functions_noise_model_heuristic_help import get_avg_large_corr, suggest_alpha_via_scan

class HeuristicClusteringAlgorithmAuto(HeuristicClusteringAlgorithmBase):

    def __init__(self, **kwargs):
        super(HeuristicClusteringAlgorithmAuto, self).__init__(**kwargs)

    def suggest_alpha_heuristic(self):
        if self.correlations_table is None:
            raise ValueError("correlations_table not specified.")
        if self.C_maxsize is None:
            raise ValueError("C_maxsize not specified.")
        return 0.5 * get_avg_large_corr(self.correlations_table, self.C_maxsize)

    def suggest_alpha_scan(self, values_to_return=None):
        return suggest_alpha_via_scan(self, values_to_return)
