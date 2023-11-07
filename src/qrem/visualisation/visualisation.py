"""
Classes for visualisation of detector tomography results.
"""
from pathlib import Path
import copy
import math

from manim import Scene, Graph, Arrow, NumberLine, Rectangle, VGroup, Tex, Text
from manim import tempconfig
from manim import BLACK, DEGREES, LEFT
import matplotlib.pyplot as plt
import numpy as np

from qrem.visualisation.plotting_constants import *

#==============================================================
# Support classes
#==============================================================
class DrawOnLayout(Scene):
    """
    Class inheriting from manim class Scene (for details on this convention see manim documentation).
    Images are rendered when this class' instance is created in the method 'draw_on_layout'
    of the class ResultsForPlotting.

    Parameters
    ----------
        plotting_data (dict):
            dictionary with data about the device and processed experimental results, necessary for manim graph
            (for details see manim graph documentation), with values:
                vertices (list[int]) : list of qubits in device, from 0 to #qubits - 1.
                edges (list[tuple[int, int]]) : list of edges (i, j) to be plotted, where i and j are in vertices.
                labels (bool|dict): dictionary whose keys are the vertices, and whose values are the corresponding
                                    vertex labels (rendered via, e.g., Text or manim Tex). Alternatively can be bool -
                                    if True, vertices are labelled with the integers from the list 'vertices'.
                vertex_config (dict): dictionary with keyword arguments which apply to manim mobject used for vertices.
                edge_config (dict): dictionary with keyword arguments which apply to manim mobject used for edges.
                layout (dict[list]): dictionary where keys are vertices and values are list of coordinates [x, y, z].
                (Optional) max_for_legend (float): maximal value to be displayed on the legend bar;
                                                   if not in keys, no legend is displayed.
    """

    def __init__(self, plotting_data):
        super().__init__()
        self.plotting_data = plotting_data

    def construct(self):
        layout_graph = Graph(self.plotting_data['vertices'],
                             self.plotting_data['edges'],
                             labels=self.plotting_data['labels'],
                             label_fill_color=WHITE,
                             vertex_config=self.plotting_data['vertex_config'],
                             edge_type=Arrow,
                             edge_config=self.plotting_data['edge_config'],
                             layout=self.plotting_data['layout']).scale(self.plotting_data['scale'])

        layout_graph.center()
        self.add(layout_graph)

        if 'max_for_legend' in self.plotting_data.keys():
            legend = NumberLine(x_range=[0, self.plotting_data['max_for_legend'], 0.05],
                                length=legend_bar_height,
                                include_tip=False,
                                include_numbers=True,
                                include_ticks=True,
                                rotation=90 * DEGREES,
                                color=BLACK,
                                stroke_opacity=0.5,
                                label_direction=LEFT,
                                font_size=20)
            legend.align_on_border([1, 0, 0], 0.5 + legend_bar_width / 2 + 0.05)
            legend.numbers.set_color(BLACK)
            legend_bar = Rectangle(
                height=legend_bar_height,
                width=legend_bar_width,
                fill_color=list(colors),
                fill_opacity=1,
                stroke_opacity=0
            )
            legend_bar.align_on_border([1, 0, 0], 0.5)
            self.add(legend_bar)
            self.add(legend)

class DrawClusters(Scene):
    """
    Class inheriting from manim class Scene (for details on this convention see manim documentation).
    Images are rendered when this class' instance is created in the method 'draw_clusters'
    of the class ResultsForPlotting.

    Parameters
    ----------
        clusters_list (list(tuples[ints])):
            list of tuples, each containing indices of qubits assigned to one cluster
    """

    def __init__(self, clusters_list):
        super().__init__()
        self.clusters_list = clusters_list

    def construct(self):
        graphs = []
        for cluster in self.clusters_list:
            vertices = list(cluster)
            edges = []
            for idx in range(len(vertices)):
                jdx = idx + 1
                while jdx < len(vertices):
                    edges.append((vertices[idx], vertices[jdx]))
                    jdx += 1
            graphs.append(Graph(vertices, edges, layout="circular", labels=True, label_fill_color=WHITE,
                                vertex_config={"fill_color": to_hex(color_map(0.5)), "radius": radius_size},
                                edge_config={"stroke_color": to_hex(color_map(0.5))}).scale(0.7))
        self.add(VGroup(*graphs).arrange())


#==============================================================
# Main Class
#==============================================================device_data_divctionary
class ResultsForPlotting:
    """
    Class whose attributes contain experimental tomography data necessary for plotting. Its method prepare the data in
    a format native to manim and create an instance of its class, which results in an image being rendered and saved.
    TODO: add remark about indexing over all vs used qubits
    """

    def __init__(self,
                 device_data_dictionary,
                 experiment_data_dictionary):
        """
        Parameters
        ----------
        device_data_dictionary (dict): device data, can be provided from file device_constants or manually. Keys:
            no_qubits (int) : number of qubits in device.
            edges_in_layout (list[tuple[int, int]]) : list of edges (i, j) present in device layout,
                                                      where 0 <= i, j < no_qubits.
            layout (dict[list]): dictionary where keys are vertices and values are list of coordinates [x, y, z].
            vertex_labels (list[int]): list of length no_qubits with qubits labels (e.g. in Rigetti devices 9th qubit
                                       has label 10) in same order as layout dictionary
            scale (float): fraction by which the whole layout graph will be scaled to fit the default manim window
        experiment_data_dictionary (dict) : experiment data. Keys:
            used_qubits (list[int]) : indices of qubits which were used in tomography (not discarded due to noise).
            correlations (ndarray(used_qubits, used_qubits)) : array of correlation coefficients between qubits,
                                                               where [i][j] indicates the influence of qubit j on i.
        Attributes
        ----------
            correlations_threshold (float) : threshold of correlations coefficient above which the corresponding edges
                                             should be plotted as coloured arrows.
            edges_with_correlations_above_threshold (list[tuple[int, int]])
            clusters_list (list[tuple[int, ..., int]]) : list of clusters into which qubits were divided during
                                                          characterisation


        """
        config.tex_dir = Path('Tex')  # manim configuration - path where tex files created for plot will be saved

        self.no_qubits = device_data_dictionary['no_qubits']
        self.edges_in_layout = device_data_dictionary['edges_in_layout']
        self.layout = device_data_dictionary['layout']
        self.scale = device_data_dictionary['scale']
        self.all_qubits = list(range(self.no_qubits))
        self.all_edges = [val for sublist in [[(i, j) for i in range(self.no_qubits)]
                                              for j in range(self.no_qubits)]
                          for val in sublist]
        self.labels = {}
        for qubit_idx in self.all_qubits:
            self.labels[qubit_idx] = Tex(str(device_data_dictionary['vertex_labels'][qubit_idx]))
        self.used_qubits = experiment_data_dictionary['used_qubits']
        # transpose correlations array, so that correlations[i][j] indicates the influence of qubit i on j:
        self.correlations = np.transpose(experiment_data_dictionary['correlations'])
        self.correlations_threshold = None
        self.edges_above_threshold = []
        self.max_correlation_coefficient = np.amax(self.correlations)
        self.clusters_list = None

    # def __identify_max_correlations(self):
    #     self.edges_directed_by_max_correlations = {}
    #     for qubit0 in range(len(self.used_qubits)):
    #         for qubit1 in range(qubit0 + 1, len(self.used_qubits)):
    #             # map indices of used qubits to indices in device layout
    #             mapped_qubit_pair = (self.used_qubits[qubit0], self.used_qubits[qubit1])
    #             if self.correlations[qubit0][qubit1] >= self.correlations[qubit1][qubit0]:
    #                 self.edges_directed_by_max_correlations[mapped_qubit_pair] = self.correlations[qubit0][qubit1]
    #             else:
    #                 self.edges_directed_by_max_correlations[tuple(reversed(mapped_qubit_pair))] = \
    #                     self.correlations[qubit1][qubit0]

    def __identify_correlations_above_threshold(self, threshold):
        self.edges_above_threshold = []
        self.correlations_threshold = threshold
        for qubit0 in range(len(self.used_qubits)):
            for qubit1 in range(len(self.used_qubits)):
                if self.correlations[qubit0][qubit1] >= threshold:
                    self.edges_above_threshold.append((qubit0, qubit1))

    def __prepare_basic_edge_config(self):
        basic_edge_config = {}
        for idx, e in enumerate(self.edges_in_layout):
            if e[0] in self.used_qubits and e[1] in self.used_qubits:
                edge_color = medium_gray
            else:
                edge_color = lightest_gray
            basic_edge_config[e] = {"stroke_color": edge_color, "stroke_width": stroke_width_thin, "buff": buff_val,
                                    "max_tip_length_to_length_ratio": 0.0, }
        return basic_edge_config

    def __prepare_basic_vertex_config(self):
        basic_vertex_config = {}
        for idx, v in enumerate(self.all_qubits):
            basic_vertex_config[v] = {"color": lightest_gray, "radius": radius_size, "z_index": -1}
            if v in self.used_qubits:
                basic_vertex_config[v] = {"color": medium_gray, "radius": radius_size}
        return basic_vertex_config

    def __prepare_edge_config_for_correlations_above_threshold(self, correlations_threshold):
        self.__identify_correlations_above_threshold(correlations_threshold)
        edge_config = self.__prepare_basic_edge_config()
        for edge in self.edges_above_threshold:
            qubit0 = edge[0]
            qubit1 = edge[1]
            mapped_qubit0 = self.used_qubits[qubit0]
            mapped_qubit1 = self.used_qubits[qubit1]
            stroke_color = to_hex(color_map(self.correlations[qubit0][qubit1]
                                            / self.max_correlation_coefficient))
            # map indices of used qubits to indices in device layout
            mapped_qubit_pair = (mapped_qubit0, mapped_qubit1)
            edge_config[mapped_qubit_pair] = {"stroke_color": stroke_color, "stroke_width": stroke_width_thick,
                                              "max_stroke_width_to_length_ratio": max_stroke_width_to_length_ratio,
                                              "max_tip_length_to_length_ratio": arrow_tip_size / (
                                                      0.01 + math.dist(self.layout[mapped_qubit0],
                                                                       self.layout[mapped_qubit1])) ** 1.4}
        return edge_config

    def __prepare_vertex_config_for_correlations_above_threshold(self):
        vertices_with_correlations_above_threshold = []

        for edge in self.edges_above_threshold:
            vertex_affecting = self.used_qubits[edge[0]]
            vertex_affected = self.used_qubits[edge[1]]
            if vertex_affected not in vertices_with_correlations_above_threshold:
                vertices_with_correlations_above_threshold.append(vertex_affected)
            if vertex_affecting not in vertices_with_correlations_above_threshold:
                vertices_with_correlations_above_threshold.append(vertex_affecting)

        vertex_config = self.__prepare_basic_vertex_config()
        for idx, v in enumerate(vertices_with_correlations_above_threshold):
            vertex_config[v] = {"color": darkest_gray, "radius": radius_size, "z_index": 0}
        return vertex_config

    def __prepare_data_for_draw_basic_layout(self):
        plotting_data = {'vertices': self.all_qubits,
                         'edges': self.edges_in_layout,
                         'labels': self.labels,
                         'layout': self.layout,
                         'scale': self.scale,
                         'vertex_config': self.__prepare_basic_vertex_config(),
                         'edge_config': self.__prepare_basic_edge_config()}
        return plotting_data

    def __prepare_data_for_draw_correlations_on_layout(self, threshold):
        labels = copy.deepcopy(self.labels)
        for qubit_idx in range(self.no_qubits):
            if qubit_idx not in self.used_qubits:
                # the following makes the label of unused qubit a transparent dot, using manim class Text :
                labels[qubit_idx] = Text('.', fill_opacity=0.0)

        edge_config = self.__prepare_edge_config_for_correlations_above_threshold(threshold)

        # create a list with edges where the edges from basic layout are below those above threshold:
        mapped_edges_above_threshold = [(self.used_qubits[unmapped_edge[0]], self.used_qubits[unmapped_edge[1]])
                                        for unmapped_edge in self.edges_above_threshold]
        edges = copy.deepcopy(self.edges_in_layout)
        for edge in mapped_edges_above_threshold:
            if edge in edges:
                edges.remove(edge)
            if tuple(reversed(edge)) in edges:
                edges.remove(tuple(reversed(edge)))
        edges = edges + mapped_edges_above_threshold

        plotting_data = {'vertices': self.all_qubits,
                         'edges': edges,
                         'labels': labels,
                         'layout': self.layout,
                         'scale': self.scale,
                         'edge_config': edge_config,
                         'vertex_config': self.__prepare_vertex_config_for_correlations_above_threshold(),
                         'max_for_legend': self.max_correlation_coefficient}
        return plotting_data

    def set_clusters_list(self, clusters_list):
        self.clusters_list = clusters_list

    def draw_on_layout(self, what_to_draw='basic_layout', file_name=None, quality='high_quality', preview=True,
                       correlations_threshold=0.01, verbosity='CRITICAL'):
        """
        Method called by user to plot the experimental data on the device's layout.

        Parameters
        ----------
            what_to_draw (str) : specifies what data should be plotted. Options:
                                    'basic_layout' : qubits with labels and edges in layout are plotted.
                                    'correlations_above_threshold' : on basic layout also edges with correlations
                                                                     coefficient above a given threshold are plotted and
                                                                     colored, with a color legend.
            quality (str) : quantifies the quality of rendered image. For details see manim docs.
                            [fourk_quality|production_quality|high_quality|medium_quality|low_quality|example_quality].
            preview (bool) : specifies whether the rendered image should be displayed in a user's default file viewer.
                             Regardless of this parameter's value, the image is saved.
            file_name (str): name of the output image file. By default, it's set to 'what_to_draw'
            correlations_threshold (float) : parameter for what_to_draw set to 'correlations_above_threshold',
                                             in range [0, 1].
            verbosity (str) : Verbosity of CLI output. [DEBUG|INFO|WARNING|ERROR|CRITICAL]
        """
        if what_to_draw == 'basic_layout':
            plotting_data = self.__prepare_data_for_draw_basic_layout()
        elif what_to_draw == 'correlations_above_threshold':
            plotting_data = self.__prepare_data_for_draw_correlations_on_layout(correlations_threshold)
        if file_name is None:
            file_name = what_to_draw
        # run manim class:
        with tempconfig({"quality": quality, "preview": preview, "output_file": file_name, "verbosity": verbosity}):
            scene = DrawOnLayout(plotting_data)
            scene.render(preview=preview)

    def draw_histogram(self, file_name='histogram', directory='', plot_title='Correlations histogram',
                       plot_label='Distance name'):
        """
            Method called by user to plot correlation coefficients on a histogram.
            Saves the plot with 'file_name' in 'directory'.

        """
        plt.rc('text', usetex=True)
        plt.rcParams.update({"font.family": "Computer Modern Roman"})
        correlations_for_hist = copy.deepcopy(self.correlations)
        # delete diagonal elements (corresponding to qubit correlation with itself) and flatten:
        correlations_for_hist = np.delete(correlations_for_hist,
                                          [k*len(self.correlations) + k for k in range(len(self.correlations))], None)
        fig, ax = plt.subplots(figsize=(8, 5))
        n, bins, patches = ax.hist(correlations_for_hist, bins=10, density=False, label=plot_label,
                                   color=color_map(0.1), linewidth=1.6)
        ax.legend(loc='upper right')
        ax.set_title(plot_title)
        ax.set_xlabel('Correlation coefficient')
        ax.set_ylabel('Number of qubit pairs')
        plt.savefig(directory + file_name + '.png', bbox_inches='tight', dpi=300)

    def draw_heatmap(self, file_name='heatmap', directory='', plot_title='Correlation coefficients'):
        """
            Method called by user to plot correlation coefficients on a heatmap.
            Saves the plot with 'file_name' in 'directory'.

        """
        plt.rc('text', usetex=True)
        plt.rcParams.update({"font.family": "Computer Modern Roman"})
        fig, ax = plt.subplots()
        im = ax.imshow(self.correlations, cmap=color_map)
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.spines[:].set_visible(False)
        cbar.outline.set_visible(False)
        ax.set_title(plot_title)
        ax.set_xticks(np.arange(0, self.correlations.shape[1], 1))
        ax.set_yticks(np.arange(0, self.correlations.shape[0], 1))
        ax.set_xticklabels(self.used_qubits)
        ax.set_yticklabels(self.used_qubits)
        ax.tick_params(which="minor", bottom=False, left=False)
        plt.savefig(directory + file_name + '.png', bbox_inches='tight', dpi=300)

    def draw_clusters(self, file_name='clusters', quality='high_quality', preview=True, verbosity='CRITICAL'):
        """
            Method called by user to plot the clusters in abstract form.
        Parameters
        ----------
            quality (str) : quantifies the quality of rendered image. For details see manim docs.
                            [fourk_quality|production_quality|high_quality|medium_quality|low_quality|example_quality].
            preview (bool) : specifies whether the rendered image should be displayed in a user's default file viewer.
                             Regardless of this parameter's value, the image is saved.
            file_name (str): name of the output image file. By default, it's set to 'what_to_draw'
            verbosity (str) : Verbosity of CLI output. [DEBUG|INFO|WARNING|ERROR|CRITICAL]
        """
        if self.clusters_list is not None:
            with tempconfig({"quality": quality, "preview": preview, "output_file": file_name, "verbosity": verbosity}):
                scene = DrawClusters(self.clusters_list)
                scene.render(preview=preview)
        else:
            raise ValueError('First set value of clusters_list attribute')
