"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""
"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""
# TODO OS: finish
import abc

class HeuristicClusteringAlgorithmBase(abc.ABC):

    @property
    @abc.abstractmethod
    def clusterize(self, **kwargs) -> Tuple[list, number]:
        # dictionary of experimental results
        raise NotImplementedError
