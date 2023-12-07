"""
The *qtypes* subpackage provides classes for representing and manipulating data in *qrem* package.

Submodules
----------
circuit_collection
    Contains the *CircuitCollection* class, which represents a collection of quantum circuits and all information necessary to run characterisationexperiment on them.
experiment_results
    Contains the *ExperimentResults* class, which represents the results of a quantum characterisation experiment.
cn_noise_model
    Contains the *CNNoiseModel* class, which represents a noise model based on Clusters and Neighbors (CN) noise model.
ctmp_noise_model
    Contains the *CTMPNoiseModel* class, which represents a noise model based on continuous-time Markov processes.
datastructure_base
    Contains the *DataStructureBase* class, which represents a base class for data holding with helpful import/export functions.

"""


from qrem.qtypes.circuit_collection import CircuitCollection
from qrem.qtypes.cn_noise_model import CNModelData
from qrem.qtypes.ctmp_noise_model import CTMPModelData
from qrem.qtypes.experiment_results import ExperimentResults
from qrem.qtypes.datastructure_base import DataStructureBase
from qrem.qtypes.characterization_data import CharacterizationData
__all__ = [
    'CircuitCollection',
    'CNModelData',
    'CTMPModelData',
    'ExperimentResults',
    'DataStructureBase',
    'CharacterizationData'
]
