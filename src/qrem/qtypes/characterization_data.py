from typing import Dict,List, Tuple, Optional, Type
import numpy as np
import numpy.typing as npt



from qrem.qtypes.datastructure_base import DataStructureBase
from qrem.qtypes.cn_noise_model import CNModelData

class CharacterizationData(DataStructureBase):


    """
    The class contains data about with characterization of readout noise results.

 
    Parameters
    ----------
    marginals_dictionary: dict
        The key is the label of a circuit (str), the value is a tuple containing all information about counts, according to following schematic:

        

        counts = {"<circuit_label>": Dict[Tuple[int], npt.NDArray[np.float_]]}

        <circuit_label>: str, BigEndian-0 on left
        Tuple[int]: tuple encoding subset 
        npt.NDArray[np.float_]]: np.array(1 x 2**(subset_length)) - array storing probability of obtaining a particular result for a measurement obtained on that subset


    POVMs_dictionary: dict

        They key is tuple with qubits subset, value numpy array storing POVM effects
        
    
    noise_matrices_dictionary: dict
        The key is tuple with qubits subset, value a dictionary storing noise matrices.
        
        In the noise matrices dictionary the possible keys are: 
            - the string 'averaged'
            - a tuple of ints encoding the neighborhood state. The qubits forming the neighborhood are stored in clusters_neighbors_sets_dictionary

    POMVs_errors_dictionary: dict
        A nested dictionary, with the structure [distance_name][distance_type][subset] where:
            distance_name: 'averaged_case' or 'worst_case'
            distance_type: 'classical' or 'quantum'
            subset: tuple of ints e.g. (0,1)
        
            The value corresponds to the value of a specified distance and reconstructed POVM to ideal computational basis  measurement 
    
    correlation_coefficients_dictionary: dict
        A nested dictionary, with the structure [distance_name][distance_type]:
            distance_name: 'averaged_case' or 'worst_case'
            distance_type: 'classical' or 'quantum'
        The value is a numpy array with entries corresponding to the value of a correlation coefficient between qubits #TODO check direction of influence 
        


            


    probabilities_list (List): TODO: find in code and comment what this is
    shots_results (List[List[str]] or np.ndarray(c, s, q)): measured outcome for each shot in each circuit, where
            c is # of circuits, s is # of shots in each circuit, q is # of qubits. Outcome saved as bitstring.
            Example: [['01001', '00100', '10101'],
                      ['01011', '11101', '01100']] is a result of 2 circuits on 5 qubits, 3 shots each circuit.
    tasks_IDs (str): device-specific ID for each circuit in the experiment, ordered #TODO - order how?
    datetime_obtained_utc: date of obtaining experiment results

    """

    def __init__(self):
        #counts
        super().__init__()

        
        self.marginals_dictionary: Dict[str, Dict[Tuple[int], npt.NDArray[np.float_]]] = None

        
        self.benchmark_marginals_dictionary = None

        self.results_dictionary = None

        self.benchmark_results_dictionary = None 
        
        self.POVMs_dictionary: Dict[Tuple[int], npt.NDArray[np.float_]] = None

        self.noise_matrices_dictionary: Dict[Tuple[int], Dict[ str | Tuple[int] , npt.NDArray[np.float_]]] = None

        self.POMVs_errors_dictionary: Dict[str, Dict[str, Dict[ Tuple[int], float ]]] = None 

        self.correlation_coefficients_dictionary: Dict[str, Dict[str, Dict[ Tuple[int], float ]]] = None 

        self.clusters_neighbors_sets_dictionary: Dict[Tuple[Tuple[int]], Dict[Tuple[Tuple[int]], Tuple[Tuple[int]]]] = None 

        self.coherence_bound_dictionary: Dict[Tuple[Tuple[int]]:List[(float,str)]] = None 

        self.noise_model_list: List[Type[CNModelData]] = None



     
    pass