#===================================================
# Constants
#===================================================
import numpy as np
#Why 9 is the number as of now? 
SIGNIFICANT_DIGITS = 9
SIGNIFICANT_DIGITS_POVM = 7

EXPERIMENT_TYPE_SYMBLOS = {"qdot": 6, "ddot": 2, "coh": 2}
CIRCUIT_DATA_TYPE = np.dtype(np.uint8) 


#For Circuits total count estimation:
CIRC_EFFECT_DISTANCE =0.1
CIRC_PROBABILITY_OF_FAIL = 0.5