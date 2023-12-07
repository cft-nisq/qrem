"""
Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""
import qutip
import numpy as np
from typing import Optional, Union, List, Tuple,Dict
 

def _embed_1q_operator(number_of_qubits:int,
                       local_operator:np.ndarray,
                       global_qubit_index:int):
    if global_qubit_index == 0:
        embed_operator = np.kron(local_operator, np.eye(int(2 ** (number_of_qubits - 1))))
        return embed_operator
    else:
        first_eye = np.eye(2 ** (global_qubit_index))
        second_eye = np.eye(2 ** (number_of_qubits - global_qubit_index - 1))

        embeded_operator = np.kron(np.kron(first_eye, local_operator), second_eye)

        return embeded_operator


def embed_operator_in_bigger_hilbert_space(number_of_qubits:int,
                                           local_operator:np.ndarray,
                                           global_indices:Optional[Union[List[int],Tuple[int]]]=[0, 1]):

    N_small = int(np.log2(local_operator.shape[0]))

    # print
    if N_small == 1:
        return _embed_1q_operator(number_of_qubits, local_operator, global_indices[0])


    qutip_object = qutip.Qobj(local_operator,
                              dims=[[2 for _ in range(N_small)],
                                    [2 for _ in range(N_small)]])


    return qutip.qip.operations.gates.expand_operator(qutip_object,
                                     number_of_qubits,
                                     global_indices).full()


# #(PP) TODO_MO This function is defined around 4-5 times, with differing implementations. We need to unpack it together and simplify
# def get_energy_from_bitstring_diagonal(bitstring: List[str],
#                                        weights_dict: Dict[Tuple[int], float]
#                                        ):
#     energy = 0
#     for qubits_tuple, hamiltonian_coefficient in weights_dict.items():
#         marginal_bitstring = [int(bitstring[q]) for q in qubits_tuple]
#         parity = (-1)**(np.count_nonzero(marginal_bitstring))
#         energy += parity * hamiltonian_coefficient

#     return energy


# def get_energy_from_bitstring_diagonal(bitstring: str,
#                                        weights_dict: Dict[str, float]
#                                        ):
#     energy = 0
#     for qubits_string, hamiltonian_coefficient in weights_dict.items():
#         qubit_indices = [int(q) for q in re.findall(r'\d+', qubits_string)]
#         marginal_bitstring = [int(bitstring[q]) for q in qubit_indices]
#         parity = (-1) ** (np.count_nonzero(marginal_bitstring))
#         energy += parity * hamiltonian_coefficient

#     return energy


# #(PP) Hamiltonian w stylu Isinga
# def get_energy_from_bitstring_diagonal_local(bitstring: str,
#                                              weights_dictionary: Dict[Tuple[int], float],
#                                              additional_multipliers=None
#                                              ):
#     """Calculates the energy corresponding to a state (encoded in bistring) and a Hamiltonian
#     (encoded in weights and additional_multipliers)"""
#     if isinstance(bitstring,list):
#         bitstring = ''.join(bitstring)

#     bitstring_getitem = bitstring.__getitem__

#     parities = {qubit_indices: (-1) ** list(map(bitstring_getitem, qubit_indices)).count('1') for qubit_indices in weights_dictionary.keys()}

#     energy = 0
#     for qubit_indices, hamiltonian_coefficient in weights_dictionary.items():
#         # marginal_bitstring = [int(bitstring[q]) for q in qubit_indices]
#         # parity = (-1) ** (np.count_nonzero(marginal_bitstring))
#         local_energy = parities[qubit_indices] * hamiltonian_coefficient
#         if additional_multipliers is not None:
#             local_energy *= additional_multipliers[qubit_indices]
#         energy += local_energy

#     return energy

#(PP) not used - commented out
# def get_plus_state(N):
#     return np.full((2**N,1),(1/np.sqrt(2)**N))


#(PP) not used - commented out
# def get_bistring_parity(bitstring:List[Union[int,str]]):
#     if isinstance(bitstring[0],str):
#         bitstring = [int(x) for x in bitstring]

#     return (-1) ** (np.count_nonzero(bitstring))


#(PP) not used - commented out
# def get_energy_from_spectrum_and_probability(spectrum:Union[np.ndarray,List[float]],
#                                              probability:Union[np.ndarray,List[float]]):
#     return sum([spectrum[i] * probability[i] for i in range(len(probability))])