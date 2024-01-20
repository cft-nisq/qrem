from typing import Dict,List, Tuple, Optional, Type
import numpy as np
import numpy.typing as npt



from qrem.qtypes.datastructure_base import DataStructureBase

class MitigationData(DataStructureBase):

    """
    Class storing results of mitigation 
    
    """

    def __init__(self):
        
        super().__init__()

    noise_models_mitigation_results_dictionary: Dict[str,Dict[Tuple[int],Dict[int,float]]] = None 


    noise_models_mitigated_energy_dictionary_error: Dict[Tuple[int],Dict[int,float]] = None


    noise_models_mitigated_energy_dictionary_error_statistics: Dict[str,Dict[Tuple[int],Dict[int,float]]] = None 
