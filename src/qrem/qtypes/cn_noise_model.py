from qrem.qtypes.datastructure_base import DataStructureBase
from qrem.common import math
from typing import Dict, Tuple, List
import operator
from functools import reduce
import numpy as np
import itertools


from qrem.common import math, probability




#TODO JS: TUZJAN, prepare full local test
#TODO PP: Find a way of exporting to jason dictionary, in which key is a tuple. This is needed to export CNModelData to json, e.g. noise matrices are stored in a dictionary with tupels as keys.  
class CNModelData(DataStructureBase):
    """
    Handles data for the Clusters and Neighbors (CN) noise model in quantum systems.

    This class is used for storing results from the CN noise model reconstruction algorithm and for mitigating readout errors
    on marginal probability distributions.

    Parameters
    ----------
    number_of_qubits : int
        The total number of qubits in the quantum system.

    Attributes
    ----------
    clusters_tuple : Tuple[Tuple]
        Information about clusters of qubits, where each tuple contains indices of qubits belonging to a cluster  e.g. ((0,2),(1,3).
    noise_matrices : Dict[Tuple,NDArray]
        Noise matrices associated with each cluster, indexed by tuples of qubit indices. Eg. (0:2) : [[0.8,0,0,0], [0.2,1,0.1,0],[0,0,0.9,0],[0,0,0,1]].
    inverse_noise_matrices : dict
        Inverse of the noise matrices, computed and stored for each cluster.  This property is filled automatically when noise_matrices property is set via set_noise_matrices_dictionary method.
    qubit_in_cluster_membership : Dict[Tuple,Tuple]
        Maps individual qubits to the cluster they belong to. Key in the dictionary is a tuple constructed form a qubit index. Corresponding value is a cluster to which the qubit belongs.
        E.g. (2,) : (0,2,5). This property is filled automatically when noise_matrices property is set via set_noise_matrices_dictionary method
    composite_inverse_noise_matrices : dict
        Stores the composite inverse noise matrices for different qubit combinations.
    clusters_neighborhoods : dict
        Information about the neighborhoods of each cluster.

    Methods
    -------
    set_noise_model :
        Sets up the noise model by initializing clusters, noise matrices, and their inverses.
    compute_extended_inverse_noise_matrices :
        Computes the extended inverse noise matrix for a given marginal.
    get_clusters_in_marginal_list :
        Returns the list of clusters involved in a specified marginal.
    ...

    Examples
    --------
    >>> model = CNModelData(number_of_qubits=5)
    >>> model.set_noise_model({_qubits_in_cluster_1: _noise_matrix_1, _qubits_in_cluster_2: _noise_matrix_2})
    >>> print(model.clusters_tuple)
    ((_qubits_in_cluster_1,), (_qubits_in_cluster_2,))
    """ 

 
    def __init__(self, number_of_qubits: int):
        """
        Initializes the CNModelData instance with the specified number of qubits.

        Parameters
        ----------
        number_of_qubits : int
            The total number of qubits in the quantum system.
        """
        super().__init__()
        self.number_of_qubits = number_of_qubits
        self.locality = None  #integer
        self.clusters_tuple=()
        self.noise_matrices = {}
        self.inverse_noise_matrices ={}
        self.qubit_in_cluster_membership ={}
        self.composite_inverse_noise_matrices = {}
        self.clusters_neighborhoods = {}
    

    def _establish_qubit_in_cluster_dictionary(self):
        """
        Method used to establish membership of qubits in clusters. 
        It constructs a dictionary with key - tuple consisting of a qubit index e.g. (0,) and value - tuple storing corresponding cluster e.g. (0,3)
        It is automatically called when set_clusters_tuple method is used.     

        """
        for cluster in self.clusters_tuple:
            for qubit in cluster:
                self.qubit_in_cluster_membership[(qubit,)]= cluster
    
    def _compute_inverse_noise_matrices(self):
        """
        Method that creates inverse noise matrix dictionary for an instance of CN noise model. 
        In the dictionary key corresponds to a tuple storing a cluster, and value to an numpy array storing inverse noise matrix e.g. (0,5): 4 by 4 numpy array.
        It is automatically called when set_noise_matrices_dictionary is used.     

        """

        for cluster, noise_matrix in self.noise_matrices.items():
            
            if bool(self.clusters_neighborhoods):

                try:
                    self.inverse_noise_matrices[(cluster)] = np.linalg.inv(noise_matrix['averaged'])
                except:
                    print(f"MATRIX FOR SUBSET: {(cluster)} is not invertible, computing pseudinverse",'','red')
                    self.inverse_noise_matrices[(cluster)] = np.linalg.pinv(noise_matrix['averaged'])


            else:
                try:
                    self.inverse_noise_matrices[(cluster)] = np.linalg.inv(noise_matrix)
                except:
                    print(f"MATRIX FOR SUBSET: {(cluster)} is not invertible, computing pseudinverse",'','red')
                    self.inverse_noise_matrices[(cluster)] = np.linalg.pinv(noise_matrix)
    

    def set_clusters_tuple(self,clusters_tuple: tuple):
        """
        Method fills clusters_tuple property. It is called when properties of a CN noise model are established via set_noise_model method.   
        In addition the method calls _establish_qubit_in_cluster_dictionary, which fills qubit_in_cluster_membership property 

        """
        if isinstance(clusters_tuple,tuple) and isinstance(clusters_tuple[0],tuple):
            self.clusters_tuple = clusters_tuple
            self._establish_qubit_in_cluster_dictionary()
        else:
             raise Exception("Wrong data format. Cluster list should be a tuple of tuples")
        
    def set_clusters_neighborhoods(self,clusters_neighborhoods: Dict[tuple,tuple]):
        """
        Method fills clusters_neighborhoods property. It is called when properties of a CN noise model are established via set_noise_model method.   
        In addition the method calls _establish_qubit_in_cluster_dictionary, which fills qubit_in_cluster_membership property 

        """
        if isinstance(clusters_neighborhoods,dict):
            self.clusters_neighborhoods = clusters_neighborhoods
            self.clusters_tuple = tuple(clusters_neighborhoods.keys())
            self._establish_qubit_in_cluster_dictionary()

        elif isinstance(clusters_neighborhoods,tuple) and isinstance(clusters_neighborhoods[0],tuple):
            self.clusters_neighborhoods = {cluster :None  for cluster in clusters_neighborhoods}
            self.clusters_tuple = clusters_neighborhoods
            self._establish_qubit_in_cluster_dictionary()
        else:
             raise Exception("Wrong data format. Cluster list should be a tuple of tuples")
    
    
        
    def set_noise_matrices_dictionary(self, noise_matrices_dictionary: Dict):

        """
        Method fills noise_matrices property. It is called when properties of a CN noise model are established via set_noise_model method.   
        In addition the method calls _compute_inverse_noise_matrices, which fills inverse_noise_matrices property.

        Parameters
        ----------
        noise_matrices_dictionary : dictionary (key - tuple of integers, value - array)
            Dictionary storing noise matrices for a CN noise model. Key in the dictionary is a tuple with indices of qubits belonging to a cluster. 
            Corresponding value is an array storing noise matrix associated with the cluster. Eg. (0:2) : [[0.8,0,0,0], [0.2,1,0.1,0],[0,0,0.9,0],[0,0,0,1]].
        
        

        """


        if isinstance(noise_matrices_dictionary,dict):
            self.noise_matrices =  noise_matrices_dictionary
            self._compute_inverse_noise_matrices()
        else:
             raise Exception("Wrong data format. Cluster list should be a list of tuples")
        
    def set_noise_model(self,noise_matrices_dictionary):

        """
        Main method used to fill properties of CN noise model. The following properties are filled when the method is called:    
        1) clusters_tuple
        2) qubit_in_cluster_membership
        3) noise_matrices 
        4) inverse_noise_matrices 
          

        Parameters
        ----------
        noise_matrices_dictionary : dictionary (key - tuple of integers, value - array)
            Dictionary storing noise matrices for a CN noise model. Key in the dictionary is a tuple with indices of qubits belonging to a cluster. 
            Corresponding value is an array storing noise matrix associated with the cluster. Eg. (0:2) : [[0.8,0,0,0], [0.2,1,0.1,0],[0,0,0.9,0],[0,0,0,1]].

        """

        self.set_clusters_tuple(clusters_tuple=tuple(noise_matrices_dictionary.keys()))
        self.set_noise_matrices_dictionary(noise_matrices_dictionary=noise_matrices_dictionary)

    def compute_extended_inverse_noise_matrices(self, clusters_in_marginal_list:List[Tuple])-> np.array:

        """
        Function determines inverse noise matrix for a given marginal. It is used as a step in mitigation_marginal function.
        The matrix is constructed as a tensor product on inverse noise matrices involved in a marginal, when it is necessary
        the matrix is permuted to ensure proper ordering of qubits indices. E.g. for a cluster_in_marginal_list 
        [(0,2),(1,3)], the returned matrix corresponds to qubits ordered as (0,1,2,3).   


        Parameters
        ----------
        noise_model : object of CNModelData class
            An object of CNModelData class 

        clusters_in_marginal_list:
            A list of tuples with clusters involved in the marginal   
            

        Returns
        -------

            An inverse noise matrix for qubits specified in clusters_in_marginal_list, qubits are sorted in ascending order 
            (e.g. for a clusters_in_marginal_list =[(0,4),(1,8)], indices of the inverse noise matrix indices correspond to qubits in the order (0,1,4,8) )
        
        """ 

        # a list of qubits is established
        unordered_qubits_in_marginal_list = list(reduce(operator.concat, clusters_in_marginal_list))

        #the inverse noise matrix is initialized as a unit numpy array 
        marginal_inverse_noise_matrix = np.array([1])
        
        # a loop over clusters in clusters_in_marginal_list
        for cluster in clusters_in_marginal_list:
        
            # total noise matrix is updated by taking tensor product with a inverse noise matrix of the currect cluster
            marginal_inverse_noise_matrix = np.kron(marginal_inverse_noise_matrix, self.inverse_noise_matrices[cluster])
    
        #final noise matrix is returned, permuted to preserve ascending order of qubit indices if necessary 
        marginal_inverse_noise_matrix = math.permute_composite_matrix(qubits_list=unordered_qubits_in_marginal_list,noise_matrix=marginal_inverse_noise_matrix)

        unordered_qubits_in_marginal_list.sort()

        key = tuple(unordered_qubits_in_marginal_list)

        self.inverse_noise_matrices[key] = marginal_inverse_noise_matrix 

        return marginal_inverse_noise_matrix 




#MOVE TO CNModelData class   

    def get_clusters_in_marginal_list(self, marginal : Tuple[int]) -> List[Tuple]:
        
        """
        Function creates a list of clusters that are involved in a marginal. Used in mitigation routines.
        For a given marginal inspects provided noise model and checks clusters membership of qubits form
        marginal.
        
        Parameters
        ----------
        marginal : tuple
            A tuple specifying marginal 

        noise_model : object of CNModelData class
            An object of CNModelData class  
            

        Returns
        -------

        clusters_in_marginal_list
            A list of tuples involved in the input marginal 
            
        
        """  

        #list storing clusters involved in the marginal, empty for start
        clusters_in_marginal_list=[]
        
        #a loop over qubits in the marginal
        for qubit in marginal:
            
            #a cluster to which the qubit belongs is determined 
            cluster = self.qubit_in_cluster_membership[(qubit,)]
            
            #this cluster is appended to clusters_in_marginal_list if it is not there 
            if cluster not in clusters_in_marginal_list:
                clusters_in_marginal_list.append(self.qubit_in_cluster_membership[(qubit,)])
        
        #results are returned
        return clusters_in_marginal_list
    

    def get_clusters_and_neighborhoods_in_marginal_dictionary(self, marginal : Tuple[int]) -> Dict[Tuple[int], Tuple[int]]:
        
        """
        Function creates a list of clusters that are involved in a marginal. Used in mitigation routines.
        For a given marginal inspects provided noise model and checks clusters membership of qubits form
        marginal.
        
        Parameters
        ----------
        marginal : tuple
            A tuple specifying marginal 

        noise_model : object of CNModelData class
            An object of CNModelData class  
            

        Returns
        -------

        clusters_in_marginal_list
            A list of tuples involved in the input marginal 
            
        
        """  

        #list storing clusters involved in the marginal, empty for start
        clusters_and_neighbors_in_marginal_dictionary={}
        
        if bool(self.clusters_neighborhoods):
        #a loop over qubits in the marginal
            for qubit in marginal:
                
                #a cluster to which the qubit belongs is determined 
                cluster = self.qubit_in_cluster_membership[(qubit,)]

                
                
                #this cluster is appended to clusters_in_marginal_list if it is not there 
                if cluster not in clusters_and_neighbors_in_marginal_dictionary.keys():
                    clusters_and_neighbors_in_marginal_dictionary[cluster] = self.clusters_neighborhoods[cluster]
            
            #results are returned
        return clusters_and_neighbors_in_marginal_dictionary
    


    def get_neighbors_in_marginal(self, marginal : Tuple[int]) -> Dict[Tuple[int], Tuple[int]]:
        
        """
        Function creates a dictionary with clusters and neighbors that belong to a marginal (it can happen that clusters' neighbors do not belong to other clusters that form a marginal .
        For a given marginal, and clusters and neighbors, membership of neighbors in the marginal is checked  
        
        Parameters
        ----------
        marginal : tuple
            A tuple specifying marginal 

        noise_model : object of CNModelData class
            An object of CNModelData class  
            

        Returns
        -------

        clusters_in_marginal_list
            A list of tuples involved in the input marginal 
            
        
        """  
        
        #here a dictionary is created to store neighbors that take part in a marginal 

        #dictionary with keys corresponding to clusters and values corresponding to neighbors that are in a marginal, and their position relative to a neighborhood 
        neighbors_coinciding_with_clusters_dictionary = {}

        clusters_and_neighbors_in_marginal_dictionary = self.get_clusters_and_neighborhoods_in_marginal_dictionary(marginal=marginal)

        qubits_in_marginal_list = list(reduce(operator.concat, list(clusters_and_neighbors_in_marginal_dictionary.keys())))


        #a loop over clusters and neighbors 
        for cluster, neighbors in clusters_and_neighbors_in_marginal_dictionary.items():

            # a temporary list storing neighbors
            neighbor_in_marginal = []

            #if there are neighbors for a cluster the check is performed 
            if neighbors!= None:

                #qubits in neighborhood are enumerated to store their relative position
                for index, qubit in enumerate(neighbors):

                    #if a qubit form this neighborhood is present in the marginal, its index and relative position is stored in a list 
                    if qubit in qubits_in_marginal_list:

                        neighbor_in_marginal.append((index,qubit))
                    
                #if after a previous loop neighbors were identified a dictionary entry is cerated
                if len(neighbor_in_marginal) > 0:

                    neighbors_coinciding_with_clusters_dictionary[cluster] = tuple(neighbor_in_marginal)
            
        return neighbors_coinciding_with_clusters_dictionary 
    

    def get_noise_matrices_indexes(self, marginal : Tuple[int] ,state):
        
        """
        Function creates a dictionary with indices of state dependent noise matrices that should be used to compute noise matrix inverse, 

        It is used during error  mitigation of CN noise model with neighbors.  
        
        Parameters
        ----------
        marginal : tuple
            A tuple specifying marginal 

        noise_model : object of CNModelData class
            An object of CNModelData class  
            

        Returns
        -------

        neighbors_indices_to_compute_average 
            A dictionary with a key - cluster and value - tuple of indicies to be averaged over or a string 'averaged' 
            
              
            
        
        """  
        
        
        
        #a dictionary to store information about noise matrices needed to compute an inverse for a given cluster is created 
        neighbors_indices_to_compute_average ={} 

        # this loop goes over all clusters that take part in a marginal, and their neighbors 

        neighbors_coinciding_with_clusters_dictionary = self.get_neighbors_in_marginal(marginal=marginal)

        clusters_and_neighbors_in_marginal_dictionary = self.get_clusters_and_neighborhoods_in_marginal_dictionary(marginal=marginal) 

        
        for cluster, neighbors in clusters_and_neighbors_in_marginal_dictionary.items():
            
            #this conditions verify whether given cluster has a neighbor that actual coincides with the marginal    
            if cluster in neighbors_coinciding_with_clusters_dictionary.keys():

                #here all possible states of the clusters neighbors are created 
                possible_states_list = list(itertools.product([0,1], repeat = len(neighbors)))
                
                #here we check, which states of neighbors are needed to compute inverse noise matrix 
                for neighbor_tuple in neighbors_coinciding_with_clusters_dictionary[cluster]:

                #possible_state is ordered according to position of a qubit in its neighborhood (relative index), while the position of a qubit in state is just its index
                    possible_states_list = [ possible_state for possible_state in possible_states_list if possible_state[neighbor_tuple[0]] == int(state[neighbor_tuple[1]]) ]
                
                neighbors_indices_to_compute_average[cluster] = tuple(possible_states_list)
            
            #if there are no neighbors in the marginal averaged noise matrix is used 
            else:

                neighbors_indices_to_compute_average[cluster] = 'averaged'
        
        return neighbors_indices_to_compute_average 


    def compute_extended_inverse_noise_matrices_state_dependent(self, clusters_in_marginal_list:List[Tuple],marginal,state)-> np.array:

        """
        Function determines inverse noise matrix for a given marginal. It is used as a step in mitigation_marginal function.
        The matrix is constructed as a tensor product on inverse noise matrices involved in a marginal, when it is necessary
        the matrix is permuted to ensure proper ordering of qubits indices. E.g. for a cluster_in_marginal_list 
        [(0,2),(1,3)], the returned matrix corresponds to qubits ordered as (0,1,2,3).   


        Parameters
        ----------
        noise_model : object of CNModelData class
            An object of CNModelData class 

        clusters_in_marginal_list:
            A list of tuples with clusters involved in the marginal   
            

        Returns
        -------

            An inverse noise matrix for qubits specified in clusters_in_marginal_list, qubits are sorted in ascending order 
            (e.g. for a clusters_in_marginal_list =[(0,4),(1,8)], indices of the inverse noise matrix indices correspond to qubits in the order (0,1,4,8) )
        
        """ 

        # a list of qubits is established
        unordered_qubits_in_marginal_list = list(reduce(operator.concat, clusters_in_marginal_list))

        #indicies of needed noise matrix are established
        noise_matrices_indicies = self.get_noise_matrices_indexes(marginal=marginal,state=state)

        #the inverse noise matrix is initialized as a unit numpy array 
        marginal_inverse_noise_matrix = np.array([1])
        
        # a loop over clusters in clusters_in_marginal_list
        for cluster in clusters_in_marginal_list:
            
            if noise_matrices_indicies[cluster]  == 'averaged':      
                
                # total noise matrix is updated by taking tensor product with a inverse noise matrix of the currect cluster
                marginal_inverse_noise_matrix = np.kron(marginal_inverse_noise_matrix, self.inverse_noise_matrices[cluster])

            else:

                current_noise_matrix = self.compute_specific_state_dependent_noise_matrix(cluster=cluster, indicies=noise_matrices_indicies[cluster])

                current_inverse_noise_matrix = np.linalg.inv(current_noise_matrix)

                marginal_inverse_noise_matrix = np.kron(marginal_inverse_noise_matrix, current_inverse_noise_matrix)




    
        #final noise matrix is returned, permuted to preserve ascending order of qubit indices if necessary 
        marginal_inverse_noise_matrix = math.permute_composite_matrix(qubits_list=unordered_qubits_in_marginal_list,noise_matrix=marginal_inverse_noise_matrix)

        unordered_qubits_in_marginal_list.sort()

        key = tuple(unordered_qubits_in_marginal_list)

        #self.inverse_noise_matrices[key] = marginal_inverse_noise_matrix 

        return marginal_inverse_noise_matrix 
    

    def compute_specific_state_dependent_noise_matrix(self, cluster: Tuple[int], indicies: Tuple[Tuple[int]]) -> np.array:

        noise_matrix = self.noise_matrices[cluster]
        
        normalization_factor = len(indicies)
        
        dim = len(noise_matrix['averaged'] )

        averaged_noise_matrix = np.zeros((dim,dim))

        for index in indicies:

            averaged_noise_matrix += noise_matrix[index]

        averaged_noise_matrix = 1/normalization_factor * averaged_noise_matrix

        return averaged_noise_matrix

        



    
   

def _t1(test_model:type[CNModelData]):
    # Test of json export

    
    json_dict_test = test_model.to_json()

    new_model =CNModelData(number_of_qubits=5)

    new_model.import_json(json_dict_test)

def _t2(test_model:type[CNModelData]):
    # Test of pickle export

 
    pickle_data = test_model.export_pickle("test_data")

    new_model =CNModelData(number_of_qubits=5)

    new_model.import_pickle("test_data")

    print(new_model.clusters_tuple)

    


if __name__=='__main__':
   #set clusters tuple
    _qubits_in_cluster_1=(0,3)
    _qubits_in_cluster_2=(1,2,4)

    #generation noise matrice for clusters with built in function probability.random_stochastic_matrix
    _noise_matrix_1= probability.random_stochastic_matrix(2**2)
    _noise_matrix_2= probability.random_stochastic_matrix(2**3)
    
    _noise_matrices_dictionary_test = {_qubits_in_cluster_1: _noise_matrix_1, _qubits_in_cluster_2: _noise_matrix_2}

    #create noise model 
    _test_model= CNModelData(number_of_qubits=5)
    _test_model.set_noise_model(noise_matrices_dictionary=_noise_matrices_dictionary_test) 

    _t2(_test_model)
 

        


    #}



   # test_model.load_from_dict(dictionary_to_load)
   # print('after loading:\n', test_model.get_dict_format())

    #NOTE: Is it possible to have a nested dictionary with keys that are not string? 
    #json_dict_test = test_model.to_json()
 
    #pickle_dict_test = test_model.get_pickle()
    #test_model.export_pickle('exported_pickle', overwrite=True)
    #print('after change:\n', test_model.get_dict_format())
    #test_model.import_pickle('exported_pickle')