import copy
from typing import Optional, List, Dict, Union, Tuple

import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import rc

from qrem.functions_qrem import functions_distances as fun_dist
from qrem.functions_qrem import povmtools
from qrem.functions_qrem.povmtools import get_enumerated_rev_map_from_indices
from qrem.noise_characterization.tomography_design.overlapping.DOTMarginalsAnalyzer import \
    DOTMarginalsAnalyzer

from qrem.common.printer import qprint

class InitialNoiseAnalyzer(DOTMarginalsAnalyzer):
    # TODO FBM, JM: think whether graphical features should be here or in some child class
    """

    """
#Mocomm - a bit wierd that som many optional POVMs
    def __init__(self,
                 results_dictionary: Dict[str, Dict[str, int]],
                 bitstrings_right_to_left: Optional[bool] = False,
                 marginals_dictionary: Optional[Dict[str, Dict[Tuple[int], np.ndarray]]] = None,
                 noise_matrices_dictionary: Optional[
                     Dict[Tuple[int], Dict[Union[str, Tuple[int]], Dict[str, np.ndarray]]]] = None,
                 clusters_list: Optional[List[Tuple[int]]] = None,
                 neighborhoods: Dict[Tuple[int], Tuple[int]] = None,
                 correlations_data: Optional[Dict[str, np.ndarray]] = None,
                 POVM_dictionary: Optional[Dict[Tuple[int], List[np.ndarray]]] = {},
                 single_qubit_errors = None,
                 qubits_indices_mapping_for_plots=None,
                 coherent_errors_1q=None,
                 correlations_pairs=None,
                 errors_data_POVMs=None,
                 coherence_data=None
                 ) -> None:

        # TODO FBM, JT: clean this up, perhaps too many arguments for initial noise analyzer

        super().__init__(results_dictionary_ddot=results_dictionary,
                         bitstrings_right_to_left=bitstrings_right_to_left,
                         marginals_dictionary=marginals_dictionary,
                         noise_matrices_dictionary=noise_matrices_dictionary
                         )

        self._correlations_data = correlations_data
        self._POVM_dictionary = POVM_dictionary
        self._errors_data_POVMs = errors_data_POVMs
        self._coherence_data = coherence_data

        if single_qubit_errors is None:
            single_qubit_errors = {(qi,): {} for qi in self._qubit_indices}

        self._single_qubit_errors = single_qubit_errors

        if clusters_list is None:
            clusters_list = []

        if neighborhoods is None:
            neighborhoods = {}

        self._clusters_list = clusters_list

        self._neighborhoods = neighborhoods

        self._coherent_errors_1q = coherent_errors_1q
        self._correlations_pairs = correlations_pairs

        fontsize_labels = 32
        fontsize_legend = 56
        fontsize_ticks = 50
        linewidth_plots = 6
        xticks_size = 20

        self._plotting_properties = {'fontsize_labels': fontsize_labels,
                                     'fontsize_legend': fontsize_legend,
                                     'fontsize_ticks': fontsize_ticks,
                                     'linewidth_plots': linewidth_plots,
                                     'xticks_size': xticks_size,
                                     'yticks_size': xticks_size
                                     # 'font_dict':{'font.size': fontsize_labels},
                                     }

        rc('text', usetex=True)
        rc('font', **{'family': 'serif',
                      'serif': ['Computer Modern'],
                      # 'weight':'bold'
                      })

        if qubits_indices_mapping_for_plots is None:
            qubits_indices_mapping_for_plots = {(qi,): qi for qi in self._qubit_indices}

        self._qubits_indices_mapping_for_plots = qubits_indices_mapping_for_plots

    @property
    def correlations_data(self):
        return self._correlations_data

    @correlations_data.setter
    def correlations_data(self, correlations_data: np.ndarray) -> None:
        self._correlations_data = correlations_data

    @property
    def errors_data(self):
        return self._errors_data_POVMs

    @errors_data.setter
    def errors_data(self, errors_data) -> None:
        self._errors_data_POVMs = errors_data


    @property
    def coherence_data(self):
        return self._coherence_data

    @coherence_data.setter
    def coherence_data(self, coherence_data) -> None:
        self._coherence_data = coherence_data







    @property
    def single_qubit_errors(self) -> dict:
        return self._single_qubit_errors

    @single_qubit_errors.setter
    def single_qubit_errors(self, single_qubit_errors: dict) -> None:
        self._single_qubit_errors = single_qubit_errors
#TODO not dectribed what this function concerns (MO) 
    def get_conservative_bound_on_statistical_errors_correlations(self,
                                                                  error_probability=0.05,
                                                                  number_of_DDOT_experiments=None):

        DDOT_keys = list(self._results_dictionary.keys())
        number_of_shots_per_setting = sum(list(self._results_dictionary[DDOT_keys[0]].values()))

        if number_of_DDOT_experiments is None:
            number_of_DDOT_experiments = len(DDOT_keys)

        number_of_qubits = self._number_of_qubits
        delta_part = np.log(1 / error_probability)
        outcomes_part = np.log(2 ** 2 - 2)
        multiple_marginals_part = np.log(number_of_qubits * (number_of_qubits - 1) * 2)

        shots_part = 2 * number_of_shots_per_setting * number_of_DDOT_experiments / 4

        return np.sqrt((delta_part + outcomes_part + multiple_marginals_part) / shots_part)

#Mocomm this method is never used - I suggest to delate it (for now I commented)


    # def _get_other_boudn_test(self,
    #                           error_probability=0.05):
    #     DDOT_keys = list(self._results_dictionary.keys())
    #     number_of_shots_per_setting = sum(list(self._results_dictionary[DDOT_keys[0]].values()))
    #     number_of_DDOT_experiments = len(DDOT_keys)

    #     shots_part = number_of_shots_per_setting * number_of_DDOT_experiments / 2

    #     outcomes_part = (1 + 1) * np.log(2)
    #     marginals_part = np.log(self._number_of_qubits)

    #     delta_part = np.log(1 / error_probability)

    #     fraction_together = -2 * (outcomes_part + marginals_part + delta_part) / shots_part
    #     # print(fraction_together, 1-fraction_together)
    #     # hej = np.
    #     epsilon = np.sqrt(np.log(1 - fraction_together))

    #     return epsilon

    # def compute_correl
#This method changes the state of "correlations data", does not return any Disctionary
    def compute_correlations_data_pairs(self,
                                       qubit_indices,
                                        distances_types=[('ac', 'classical')],
                                        chopping_threshold: Optional[float] = 0) -> Dict[
        str, Dict[str, np.array]]:

        qubit_pairs = [(qi, qj) for qi in qubit_indices for qj in qubit_indices if qj > qi]

        number_of_qubits = len(qubit_indices)

        if np.max(qubit_indices) > number_of_qubits:
            mapping = get_enumerated_rev_map_from_indices(qubit_indices)
        else:
            mapping = {qi: qi for qi in qubit_indices}

        correlations_data = {distance_tuple[0]: {} for distance_tuple in distances_types}

        for distance_tuple in distances_types:
            correlations_data[distance_tuple[0]][distance_tuple[1]] = np.zeros((number_of_qubits,
                                                                                number_of_qubits),
                                                                               dtype=float)

        for distance_tuple in distances_types:
            qprint("\nCalculating correlations of type:", distance_tuple, 'red')

            (distance_type, correlations_type) = distance_tuple

            classical_correlations = False
            if correlations_type.lower() in ['classical', 'diagonal']:
                classical_correlations = True

            for (qi, qj) in tqdm(qubit_pairs):
                qi_mapped, qj_mapped = mapping[qi], mapping[qj]

                povm_2q_now = self._POVM_dictionary[(qi, qj)]

                c_i_j, c_j_i = fun_dist.find_correlations_coefficients(povm_2q=povm_2q_now,
                                                                       distance_type=distance_type,
                                                                       classical=classical_correlations,
                                                                       direct_optimization=True)

                if c_i_j >= chopping_threshold:
                    correlations_data[distance_type][correlations_type][qi_mapped, qj_mapped] = c_i_j

                if c_j_i >= chopping_threshold:
                    correlations_data[distance_type][correlations_type][qj_mapped, qi_mapped] = c_j_i
            qprint("DONE")

        if self._correlations_data is None:
            self._correlations_data = correlations_data
        else:
            self._correlations_data = {**self._correlations_data, **correlations_data}


#JT: This function is used to compute a distance between an ideal POVM and the reconstructed one
#qubits_subsets - a list of qubits for which the computation is to be performed, for 1 qunits POVMs entries of the list consits of a tuple contaning one integer
# for two qubit POVMs tuple contains two integers
#distance types a list of tuples, contaning strings, that specifies distance(s) to be computed
#chopping_treshold - a threshold for computed distances
# resulting data structure distance_name :{distance_type:{(qubits_subset): distance_value}}


#

    def compute_errors_POVMs(self,
                             qubits_subsets: List[Tuple[int]],
                             distances_types=[('ac', 'classical')],
                             chopping_threshold: Optional[float] = 0) -> Dict[
        str, Dict[str, Dict[Tuple[int], float]]]:

        #JT: a list contaning uniqe sizes of qubits subsets passed to the class

        unique_subsets_sizes = list(np.unique([len(x) for x in qubits_subsets]))

        #JT: a dictionary contaning all ideal, projetive measurements in the computational basis
        # for dimensions corresponding to the unique dimensions passed to the input

        POVMs_compuational_basis = {x: povmtools.computational_projectors(d=2, n=x) for x in
                                    unique_subsets_sizes}

        #JT: errors_data is a dictionary that stores results of computation
        #the keys are the first entry of a tuple corresponding to chosen distances (ac/wc)
        #this is a nested dictionary - values will be also dictionaries

        errors_data = {distance_tuple[0]: {} for distance_tuple in distances_types}

        #JT: the second key in errors data is a string classical/quantum"
        #JT: We need to discuess if we want to store it this way


        for distance_tuple in distances_types:
            errors_data[distance_tuple[0]][distance_tuple[1]] = {}

        #JT: a loop to compute distances

        for distance_tuple in distances_types:



            #JT: information about computation is printed

            qprint("\nCalculating errors of type:", distance_tuple, 'red')

            #JT: an internal loop over subsets

            for subset in tqdm(qubits_subsets):

                #JT: Povm corresponding to subset is loaded into povm_now

                povm_now = self._POVM_dictionary[subset]

                #JT: function calculating specified distances, detailed comments in function distances and povmtools
                #functions involved include average_distance_POVMs and  operational_distance_POVMs

                distance_now = fun_dist.calculate_distance_between_POVMs(POVM_1=povm_now,
                                                                         POVM_2=
                                                                         POVMs_compuational_basis[
                                                                             len(subset)],
                                                                         distance_type_tuple=distance_tuple)

                #JT: treshold check

                if distance_now >= chopping_threshold:
                    errors_data[distance_tuple[0]][distance_tuple[1]][subset] = distance_now

            qprint("DONE")

        #JT: If no distances are computed the resulting dictionary is assigned to _errors_data_POVMs

        if self._errors_data_POVMs is None:
            self._errors_data_POVMs = errors_data

        #JT: a union of dictionaries is performed when some distances are computed

        else:
            self._errors_data_POVMs = {**self._errors_data_POVMs, **errors_data}

    #JT: this method is used to compute distances between coherent and their diagonal parts
    #inputs: substes_of_qubits - determines POVMs to be computed, distances_types -types of distances to be computed
    def compute_coherences_POVMs(self,
                                 qubits_subsets: List[Tuple[int]],
                                 distances_types=[('ac', 'quantum')],
                                 chopping_threshold: Optional[float] = 0) -> Dict[
        str, Dict[str, Dict[Tuple[int], float]]]:



        #JT: diagonal part of POVM is taken

        POVMs_diagonal = {subset: povmtools.get_diagonal_povm_part(self._POVM_dictionary[subset]) for
                          subset in qubits_subsets}

        #JT a nested dictionary is initialized


        coherences_data = {distance_tuple[0]: {} for distance_tuple in distances_types}

        for distance_tuple in distances_types:
            coherences_data[distance_tuple[0]][distance_tuple[1]] = {}

        #JT: start of calculation message is printed

        for distance_tuple in distances_types:
            qprint("\nCalculating errors of type:", distance_tuple, 'red')

            #JT: warning message when classical type of distance is chosen as results are trivial

            if distance_tuple[1].lower() in ['classical']:
                qprint("Warning:",
                               f"Coherence for classical part of POVM is by definition zero.", 'red')

            for subset in tqdm(qubits_subsets):
                povm_now = self._POVM_dictionary[subset]

                #JT: calculation of distances is performed in the same way as in compute_errors_POVMs

                distance_now = fun_dist.calculate_distance_between_POVMs(POVM_1=povm_now,
                                                                         POVM_2=POVMs_diagonal[subset],
                                                                         distance_type_tuple=distance_tuple)

                if distance_now >= chopping_threshold:
                    coherences_data[distance_tuple[0]][distance_tuple[1]][subset] = distance_now

            qprint("DONE")

        #JT: update of class property storing data

        if self._coherence_data is None:
            self._coherence_data = coherences_data
        else:
            self._coherence_data = {**self._coherence_data, **coherences_data}
#MOcomm - this is an old function - do we need to use it - it is used 
    def _compute_correlations_data_pairs_old(self,
                                             qubit_indices: Optional[List[int]] = None,
                                             chopping_threshold: Optional[float] = 0.) -> np.ndarray:
        """From marginal noise matrices, get correlations between pairs of qubits.
           Correlations are defined as:

           c_{j -> i_index} =
                           1/2 * || Lambda_{i_index}^{Y_j = '0'} - Lambda_{i_index}^{Y_j = '0'}||_{l1}

           Where Lambda_{i_index}^{Y_j} is an effective noise matrix on qubit "i_index"
           (averaged over all other of qubits except "j"), provided that input state
           of qubit "j" was "Y_j". Hence, c_{j -> i_index} measures how much
           noise on qubit "i_index" depends on the input state of qubit "j".

           :param qubit_indices: list of integers labeling the qubits we want to consider
                  if not provided, uses class property self._qubit_indices

           :param chopping_threshold: numerical value, for which correlations lower than
                  chopping_threshold are set to 0. If not provided, does not chop.
                  In general, it is advisable to set such cluster_threshold that
                  cuts off values below expected statistical fluctuations.

           :return: correlations_table_quantum (ARRAY):
                    element correlations_table_quantum[i_index,j] =
                    how qubit "j" AFFECTS qubit "i_index"
                    [= how noise on qubit "i_index" depends on "j"]
           """

        add_property = False
        if qubit_indices is None:
            add_property = True
            qubit_indices = self._qubit_indices

        number_of_qubits = len(qubit_indices)
        correlations_table_average = np.zeros((number_of_qubits, number_of_qubits))
        correlations_table_worst = np.zeros((number_of_qubits, number_of_qubits))

        if np.max(qubit_indices) > number_of_qubits:
            mapping = get_enumerated_rev_map_from_indices(qubit_indices)
        else:
            mapping = {qi: qi for qi in qubit_indices}

        for qi in qubit_indices:
            for qj in qubit_indices:
                qi_mapped, qj_mapped = mapping[qi], mapping[qj]
                if qj > qi:

                    # TODO FBM: ADJUST FOR FULL QDT
                    lam_i_j = self.get_noise_matrix_dependent((qi,),
                                                              (qj,))
                    lam_j_i = self.get_noise_matrix_dependent((qj,),
                                                              (qi,))

                    # how "i" is affected by "j"
                    diff_i_j = lam_i_j['0'] - lam_i_j['1']

                    # how "j" is affected by "i"
                    diff_j_i = lam_j_i['1'] - lam_j_i['0']

                    lam2q = self.get_noise_matrix_averaged(subset=(qi, qj))
                    povm2q_diagonal = povmtools.get_povm_from_stochastic_map(stochastic_map=lam2q)

                    correlation_i_j_worst = 1 / 2 * np.linalg.norm(diff_i_j, ord=1)
                    correlation_j_i_worst = 1 / 2 * np.linalg.norm(diff_j_i, ord=1)

                    # test_ij, test_ji =fun_dist.find_correlations_coefficients(povm_2q=povm2q_diagonal,
                    #                                                  distance_type='worst-case',
                    #                                                           classical=True)
                    # print()
                    # print(abs(correlation_i_j_worst-test_ij)<=10**(-8), abs(correlation_j_i_worst- test_ji)<=10**(-8))
                    #

                    correlation_i_j_average, correlation_j_i_average = fun_dist.find_correlations_coefficients(
                        povm_2q=povm2q_diagonal,
                        distance_type='average-case',
                        classical=True)

                    if correlation_i_j_worst >= chopping_threshold:
                        correlations_table_worst[qi_mapped, qj_mapped] = correlation_i_j_worst

                    if correlation_j_i_worst >= chopping_threshold:
                        correlations_table_worst[qj_mapped, qi_mapped] = correlation_j_i_worst

                    if correlation_i_j_average >= chopping_threshold:
                        correlations_table_average[qi_mapped, qj_mapped] = correlation_i_j_average

                    if correlation_j_i_average >= chopping_threshold:
                        correlations_table_average[qj_mapped, qi_mapped] = correlation_j_i_average

        if add_property:
            if 'diagonal' not in self._correlations_data.keys():
                self._correlations_data['diagonal'] = {}
            self._correlations_data['diagonal']['average-case'] = correlations_table_average
            self._correlations_data['diagonal']['worst-case'] = correlations_table_worst
        # raise KeyboardInterrupt
        return {'diagonal': {'average-case': correlations_table_average,
                             'worst-case': correlations_table_worst}}
#Mocomm we never use this method, I suggest to delate it
    def compute_1q_errors(self,
                          qubit_indices=None):
        # TODO FBM: add statistical errors
        if qubit_indices is None:
            qubit_indices = self._qubit_indices

        for qi in qubit_indices:
            noise_matrix = self.get_noise_matrix_averaged(subset=(qi,))

            p_10 = noise_matrix[1, 0]
            p_01 = noise_matrix[0, 1]

            self._single_qubit_errors[(qi,)]['1|0'] = p_10
            self._single_qubit_errors[(qi,)]['0|1'] = p_01

        return self._single_qubit_errors


#MOcomm - not sure if needed I suggest delating
#    def print_properties(self):
#        # TODO FBM, OS: add this
#
#        return None

#MOcomm - not sure if needed I suggest delating
#    def draw_noise_model(self):
#        # TODO FBM, JM: add this
#
#        return None



#####  BELOW WE HAWE SOME CRAZY DROWING TOOLS - THEY SHOULD NOT BE HERE!!!! (MO)
# ORGANIZE - everything below is research-specific and we don't use it anymore. A basic histogram is implemented in
# visualisation.ResultsForPlotting method draw_histogram, the research scripts are in use_cases. This can be deleted.

    def __prepare_drawing_1q_errors_barplot(self,
                                            axis,
                                            errors_list,
                                            which_error: str):

        if which_error == "p(0|1)":
            error_insert_string = r"p(0$\vert$1)"
            color = 'red'
        elif which_error == "p(1|0)":
            error_insert_string = r"p(1$\vert$0)"
            color = 'green'

        ylabel = fr"Error prob. {error_insert_string} [$\%$]"

        qubits_labels = [f"{self._qubits_indices_mapping_for_plots[(qi,)]}" for qi in
                         self._qubit_indices]
        bar_width = self._number_of_qubits / 10 * 0.6
        labelpad_x = bar_width * 1.5

        axis.set_xlabel('Qubit label', fontsize=self._plotting_properties['fontsize_labels'])
        axis.set_ylabel(ylabel,
                        fontsize=self._plotting_properties['fontsize_labels'])
        axis.bar(x=self._qubit_indices,
                 height=errors_list,
                 width=bar_width,
                 color=color)
        axis.xaxis.labelpad = labelpad_x
        axis.set_xticks(self._qubit_indices)
        axis.set_xticklabels(qubits_labels,
                             rotation=30,
                             fontsize=self._plotting_properties['xticks_size'])

        mean_errors = np.mean(errors_list)
        position_right_mean = self._number_of_qubits - 0.5

        axis.set_xlim(-1, self._number_of_qubits)

        max_ytick = max([np.round(x, 1) for x in errors_list])
        axis.set_ylim(0, max_ytick * 1.05)

        axis.hlines(y=mean_errors,
                    xmin=-0.5,
                    xmax=position_right_mean,
                    linestyles='--',
                    colors='black',
                    label='mean',
                    linewidth=3)

        axis.annotate(f"mean = {np.round(mean_errors, 1)}",
                      xy=(position_right_mean - 1.25, mean_errors + 0.05 * max_ytick),
                      fontsize=self._plotting_properties['fontsize_labels'])

        #

        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(self._plotting_properties['yticks_size'])

        return axis
    #ORGANIZE - delete
    def __draw_1q_errors_barplot(self,
                                 additional_string_title=None):

        # qubits_mapping = self._qubits_indices_mapping_for_plots

        errors_10 = [self._single_qubit_errors[(qi,)]['1|0'] * 100
                     for qi in self._qubit_indices]
        errors_01 = [self._single_qubit_errors[(qi,)]['0|1'] * 100
                     for qi in self._qubit_indices]
        figure_pyplot = plt.figure()
        axis_10 = figure_pyplot.add_subplot(2, 1, 1)
        axis_10 = self.__prepare_drawing_1q_errors_barplot(axis=axis_10,
                                                           which_error="p(1|0)",
                                                           errors_list=errors_10)

        axis_01 = figure_pyplot.add_subplot(2, 1, 2)
        axis_01 = self.__prepare_drawing_1q_errors_barplot(axis=axis_01,
                                                           which_error="p(0|1)",
                                                           errors_list=errors_01)

        figure_title = "1q errors"

        if additional_string_title is not None:
            figure_title = figure_title + f"{additional_string_title}"
        figure_pyplot.suptitle(figure_title, fontsize=self._plotting_properties['fontsize_labels'],
                               y=0.98)
        figure_pyplot.tight_layout()

        return figure_pyplot

#ORGANIZE MO - for sure plotting does not belong here (MO)! ###########
#ORGANIZE JM - I agree, let's delete this
    def __prepare_drawing_1q_errors_histogram(self,
                                              axis,
                                              errors_list,
                                              which_error: str,
                                              cumulative=False):

        if which_error == "p(0|1)":
            error_insert_string = r"p(0$\vert$1)"
            color = 'red'
        elif which_error == "p(1|0)":
            error_insert_string = r"p(1$\vert$0)"
            color = 'green'
        elif which_error.upper() == 'AVERAGE-CASE DISTANCE':
            color = 'green'
            error_insert_string = r"coherent average-case distance"

        elif which_error.upper() == 'WORST-CASE DISTANCE':
            color = 'red'
            error_insert_string = r"coherent worst-case distance"

        # ylabel = fr"Error prob. {error_insert_string} [$\%$]"
        #
        #
        # qubits_labels = [f"{self._qubits_indices_mapping_for_plots[(qi,)]}" for qi in
        #                  self._qubit_indices]

        #
        # bar_width = self._number_of_qubits / 10 * 0.6
        # labelpad_x = bar_width * 1.5
        # error_insert_string = 'coherent'
        axis.set_xlabel(f'Error {error_insert_string} [\%]',
                        fontsize=self._plotting_properties['fontsize_labels'])
        # TODO FBM: putting by hand
        # axis.set_xlabel(f'Coherent error [\%]', fontsize=self._plotting_properties['fontsize_labels'])

        if cumulative:
            axis.hist(
                x=errors_list,
                bins=int(self._number_of_qubits),
                color=color,
                cumulative=cumulative,
                density=True,
                histtype='step',
                # stacked=True
                # x=self._qubit_indices,
                #      height=errors_list,
                #      width=bar_width,
                #      color=color
            )

            axis.set_ylabel("CDF",
                            fontsize=self._plotting_properties['fontsize_labels'])

            max_error = np.max(errors_list)

            axis.vlines(x=max_error,
                        ymin=0.,
                        ymax=1,
                        linestyles='-',
                        colors='white',
                        # label='mean',
                        linewidth=3)






        else:

            axis.set_ylabel("Amount of qubits",
                            fontsize=self._plotting_properties['fontsize_labels'])

            axis.hist(
                x=errors_list,
                bins=int(self._number_of_qubits),
                color=color,
                cumulative=cumulative
                # x=self._qubit_indices,
                #      height=errors_list,
                #      width=bar_width,
                #      color=color
            )
        # axis.xaxis.labelpad = labelpad_x

        min, max, mean = int(np.floor(np.min(errors_list))), int(
            np.floor(np.max(errors_list))), np.mean(errors_list)

        axis.set_xticks(list(np.round(np.linspace(min, mean, 4), 1)) +
                        list(np.round(np.linspace(mean, max * 1.4, 8), 1)))

        # axis.set_xticks([0.5, 1, 2, 4, 5] + list(range(10,max_error+1,5)))
        # axis.set_xticklabels(qubits_labels,
        #                      rotation=30,
        #                      fontsize=self._plotting_properties['xticks_size'])

        # mean_errors = np.mean(errors_list)
        # position_right_mean = mean_errors

        # axis.set_xlim(-1, self._number_of_qubits)

        # max_xtick = max([np.round(x, 1) for x in errors_list])
        # axis.set_xlim(0, max_xtick * 1.05)

        # axis.hlines(y = mean_errors,
        #             xmin= -0.5,
        #             xmax = max_xtick,
        #             linestyles='--',
        #             colors='black',
        #             label='mean',
        #             linewidth=3)
        #
        # axis.annotate(f"mean = {np.round(mean_errors,1)}",
        #               xy=(position_right_mean-1.25, mean_errors+0.05*max_xtick),
        #               fontsize=self._plotting_properties['fontsize_labels'])
        #
        #
        # #

        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(self._plotting_properties['yticks_size'])

        return axis
    #ORGANIZE - delete
    def __draw_errors_histogram(self,
                                additional_string_title=None,
                                errors_type='classical',
                                histogram_type=None):

        if histogram_type is None:
            if self._number_of_qubits > 30:
                histogram_type = 'cumulative'
            else:
                histogram_type = 'bar'

        if histogram_type.upper() == 'BAR':
            cumulative_histogram = False
        elif histogram_type.upper() == 'CUMULATIVE':
            cumulative_histogram = True

        # qubits_mapping = self._qubits_indices_mapping_for_plots

        if errors_type.upper() == 'CLASSICAL':
            errors_type_0_list = [self._single_qubit_errors[(qi,)]['1|0'] * 100
                                  for qi in self._qubit_indices]
            errors_type_1_list = [self._single_qubit_errors[(qi,)]['0|1'] * 100
                                  for qi in self._qubit_indices]

            errors_type_0_name = "p(1|0)"
            errors_type_1_name = "p(0|1)"

        elif errors_type.upper() == 'COHERENT':
            errors_type_0_list = [self._coherent_errors_1q[(qi,)]['average_case'] * 100
                                  for qi in self._qubit_indices]
            errors_type_1_list = [self._coherent_errors_1q[(qi,)]['worst_case'] * 100
                                  for qi in self._qubit_indices]

            errors_type_0_name = 'average-case distance'
            errors_type_1_name = 'worst-case distance'

        # elif err

        figure_pyplot = plt.figure()
        axis_10 = figure_pyplot.add_subplot(1, 1, 1)
        axis_10 = self.__prepare_drawing_1q_errors_histogram(axis=axis_10,
                                                             which_error=errors_type_0_name,
                                                             errors_list=errors_type_0_list,
                                                             cumulative=cumulative_histogram)

        # axis_01 = figure_pyplot.add_subplot(2, 1, 2)
        # axis_01 = self.__prepare_drawing_1q_errors_histogram(axis=axis_01,
        #                                                    which_error=errors_type_1_name,
        #                                                    errors_list=errors_type_1_list,
        #                                                      cumulative=cumulative_histogram)

        figure_title = "1q errors"

        if additional_string_title is not None:
            figure_title = figure_title + f"{additional_string_title}"
        # figure_pyplot.suptitle(figure_title, fontsize=self._plotting_properties['fontsize_labels'], y=0.98)
        figure_pyplot.tight_layout()

        return figure_pyplot
    #ORGANIZE - delete
    def draw_1q_errors(self,
                       additional_string_title=None,
                       plottype=None,
                       errors_type='classical'):

        additional_string_title = additional_string_title.replace('_', '-')

        if plottype is None:
            if self._number_of_qubits <= 25:
                figure_pyplot = self.__draw_1q_errors_barplot(
                    additional_string_title=additional_string_title,
                    errors_type=errors_type)

            else:
                figure_pyplot = self.__draw_errors_histogram(additional_string_title=
                                                             additional_string_title,
                                                             errors_type=errors_type)
        else:
            # TODO FBM: make it smarter
            if plottype.upper() == 'BARPLOT':
                figure_pyplot = self.__draw_1q_errors_barplot(
                    additional_string_title=additional_string_title,
                    errors_type=errors_type)
            elif plottype.upper() == "HISTOGRAM-BAR":
                figure_pyplot = self.__draw_errors_histogram(
                    additional_string_title=additional_string_title,
                    errors_type=errors_type,
                    histogram_type='BAR')
            elif plottype.upper() == "HISTOGRAM-CUMULATIVE":
                figure_pyplot = self.__draw_errors_histogram(
                    additional_string_title=additional_string_title,
                    errors_type=errors_type,
                    histogram_type='CUMULATIVE')

        return figure_pyplot

    #ORGANIZE: this can be deleted, it's only used in draw_correlations below, which can be deleted
    def __prepare_correlations_drawing_heatmap_classical(self,
                                                         additional_string_title=None,
                                                         ):

        title_heatmap = f"Correlations table"

        if additional_string_title is not None:
            title_heatmap = title_heatmap + f"{additional_string_title}"

        correlations_table = copy.deepcopy(self._correlations_table)

        for i in range(self._number_of_qubits):
            for j in range(self._number_of_qubits):
                if correlations_table[i, j] != 0:
                    correlations_table[i, j] *= 100

        statistics_bound = self.get_conservative_bound_on_statistical_errors_correlations() * 100
        max_value = correlations_table.max()

        percentage_cutoff = statistics_bound / max_value
        percentage_notcutoff = 1 - percentage_cutoff

        number_of_points_cutoff = int(np.floor(percentage_cutoff * 128))
        number_of_points_notcutoff = int(np.ceil(percentage_notcutoff * 128))
        if number_of_points_cutoff == 0:
            number_of_points_cutoff += 1
            number_of_points_notcutoff += -1

        color_zero = plt.cm.nipy_spectral_r([0])
        colors_list_zero = np.array([list(color_zero[0]) for _ in range(number_of_points_cutoff)])
        colors_correlations = plt.cm.gist_heat_r(np.linspace(0, 1, number_of_points_notcutoff))[
                              int(number_of_points_notcutoff * 0.1):]

        colors_merged = np.vstack((colors_list_zero, colors_correlations))

        color_map_merged = mcolors.LinearSegmentedColormap.from_list('colormap_merged', colors_merged)

        heatmap = sns.heatmap(correlations_table,
                              cmap=color_map_merged,
                              # norm=divnorm,
                              linewidths=1,
                              linecolor='white',
                              cbar=True,
                              # cbar_kws={'label': r"$c_{i \rightarrow j}\ [\%]$"}
                              )

        ticks_fontsize = self._plotting_properties['fontsize_labels']
        if self._number_of_qubits > 30:
            ticks_fontsize *= 1 / 3

        # if self._number_of_qubits > 80:
        #     ticks_fontsize*=1/2

        y_label = fr"Affected qubit label"
        x_label = fr"Affecting qubit label"

        qubits_labels = [f"{self._qubits_indices_mapping_for_plots[(qi,)]}" for qi in
                         self._qubit_indices]

        ###########################################################################################
        ticks = [i + 0.5 for i in range(len(qubits_labels))]
        heatmap.set_yticks(ticks)
        heatmap.set_xticks(ticks)

        heatmap.set_xticklabels(qubits_labels,
                                rotation=90,
                                fontsize=ticks_fontsize)

        heatmap.set_yticklabels(qubits_labels,
                                fontsize=ticks_fontsize,
                                rotation=0)

        ###########################################################################################

        bar_width = self._number_of_qubits / 10 * 0.6
        # labelpad_x = bar_width * 1.5

        heatmap.set_xlabel(x_label, fontsize=ticks_fontsize)
        heatmap.set_ylabel(y_label,
                           fontsize=ticks_fontsize)

        cbar = heatmap.collections[0].colorbar
        cbar.set_label(r"$c_{i \rightarrow j}\ [\%]$",
                       labelpad=50,
                       rotation=0,
                       fontsize=self._plotting_properties['fontsize_ticks'])
        # cbar.set_annotation("HEJKA")

        cbar.ax.tick_params(labelsize=self._plotting_properties['fontsize_labels'])
        yticks = cbar.ax.get_yticks()
        yticks_labels = list(cbar.ax.get_yticklabels())

        cbar.ax.annotate(r"$\leftarrow$ cutoff",
                         (1., percentage_cutoff * 1.02),
                         fontsize=self._plotting_properties['fontsize_labels'])

        #
        #
        # print(yticks)
        # print(yticks_labels)
        # raise KeyboardInterrupt

        # cbar.ax.set_yticklabels()

        heatmap.set_title(title_heatmap,
                          fontsize=self._plotting_properties['fontsize_labels'],
                          y=0.98)
        # heatmap.xaxis.tick_top()
        # heatmap.xaxis.set_label_position('top')

        return heatmap
    #ORGANIZE: this can be deleted, it's only used in draw_correlations below, which can be deleted
    def __prepare_correlations_drawing_heatmap_nonclassical(self,
                                                            additional_string_title=None,
                                                            correlations_type='COHERENT-AVERAGE'
                                                            ):

        title_heatmap = f"Correlations table, {correlations_type}"

        if additional_string_title is not None:
            title_heatmap = title_heatmap + f"\n{additional_string_title}"

        correlations_table = np.zeros((self._number_of_qubits, self._number_of_qubits))

        for i in range(self._number_of_qubits):
            for j in range(i + 1, self._number_of_qubits):

                if correlations_type.upper() == 'COHERENT-AVERAGE':
                    corr_now = self._correlations_pairs['coherent'][(i, j)]['average']
                elif correlations_type.upper() == 'COHERENT-WORST':
                    corr_now = self._correlations_pairs['coherent'][(i, j)]['worst']

                if corr_now != 0:
                    correlations_table[i, j] = 100 * corr_now

        max_value = correlations_table.max()
        percentage_cutoff = 0
        percentage_notcutoff = 1 - percentage_cutoff
        number_of_points_cutoff = int(np.floor(percentage_cutoff * 128))
        number_of_points_notcutoff = int(np.ceil(percentage_notcutoff * 128))
        if number_of_points_cutoff == 0:
            number_of_points_cutoff += 1
            number_of_points_notcutoff += -1

        color_zero = plt.cm.nipy_spectral_r([0])
        colors_list_zero = np.array([list(color_zero[0]) for _ in range(number_of_points_cutoff)])
        colors_correlations = plt.cm.gist_heat_r(np.linspace(0, 1, number_of_points_notcutoff))[
                              int(number_of_points_notcutoff * 0.1):]

        colors_merged = np.vstack((colors_list_zero, colors_correlations))

        color_map_merged = mcolors.LinearSegmentedColormap.from_list('colormap_merged', colors_merged)

        heatmap = sns.heatmap(correlations_table,
                              cmap=color_map_merged,
                              # norm=divnorm,
                              linewidths=5,
                              linecolor='white',
                              cbar=True,
                              # cbar_kws={'label': r"$c_{i \rightarrow j}\ [\%]$"}
                              )

        y_label = fr"qubit label"
        x_label = fr"qubit label"

        qubits_labels = [f"{self._qubits_indices_mapping_for_plots[(qi,)]}" for qi in
                         self._qubit_indices]
        ###########################################################################################
        ticks = [i + 0.5 for i in range(len(qubits_labels))]
        heatmap.set_yticks(ticks)
        heatmap.set_xticks(ticks)

        ticks_fontsize = self._plotting_properties['fontsize_labels']

        if self._number_of_qubits > 30:
            ticks_fontsize *= 1 / 3

        heatmap.set_xticklabels(qubits_labels,
                                rotation=30,
                                fontsize=ticks_fontsize)

        heatmap.set_yticklabels(qubits_labels,
                                fontsize=ticks_fontsize,
                                rotation=0)
        ###########################################################################################

        bar_width = self._number_of_qubits / 10 * 0.6
        # labelpad_x = bar_width * 1.5

        heatmap.set_xlabel(x_label, fontsize=self._plotting_properties['fontsize_labels'])

        heatmap.set_ylabel(y_label,
                           fontsize=self._plotting_properties['fontsize_labels'])

        cbar = heatmap.collections[0].colorbar
        cbar.set_label(r"$[\%]$",
                       labelpad=50,
                       rotation=0,
                       fontsize=self._plotting_properties['fontsize_ticks'])

        cbar.ax.tick_params(labelsize=self._plotting_properties['fontsize_labels'])

        heatmap.set_title(title_heatmap,
                          fontsize=self._plotting_properties['fontsize_labels'],
                          y=0.98)

        return heatmap
    #ORGANIZE: same functionality is implemented in visualisation.ResultsForPlotting method draw_heatmap
    #This can be deleted
    def draw_correlations(self,
                          additional_string_title=None,
                          correlations_type='classical',
                          plottype='heatmap'):

        additional_string_title = additional_string_title.replace('_', '-')

        # TODO FBM, JM: Think whether to do some more plot options

        if plottype.lower() in ['heatmap']:

            if correlations_type.upper() == 'CLASSICAL':
                plot = self.__prepare_correlations_drawing_heatmap_classical(
                    additional_string_title=additional_string_title)
            else:
                plot = self.__prepare_correlations_drawing_heatmap_nonclassical(
                    additional_string_title=additional_string_title,
                    correlations_type=correlations_type)


        elif plottype.lower() in ['histogram']:
            if correlations_type.upper() == 'CLASSICAL':
                raise KeyError()
                histogram = self.draw_1q_errors(
                    additional_string_title=None,
                    plottype=None,
                    errors_type='classical')

            else:
                raise ValueError()
                plot = self.__prepare_correlations_drawing_heatmap_nonclassical(
                    additional_string_title=additional_string_title,
                    correlations_type=correlations_type)

        return plot
