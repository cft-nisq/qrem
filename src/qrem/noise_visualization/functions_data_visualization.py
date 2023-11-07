"""
@authors: Filip Maciejewski, Oskar Słowik, Tomek Rybotycki
@contact: filip.b.maciejewski@gmail.com
"""
# ORGANIZE from this file only function print_partition is used in non-deprecated code (we can discuss deleting that use);
# print_partition also uses create_undirected_nxgraph_from_partition; other functions can be deleted
from itertools import chain

import matplotlib.pyplot as plt
import networkx as nx
from qrem.functions_qrem import ancillary_functions as anf

# ORGANIZE - out of the non deprecated/ for deletion functions this is only used below in print_partition
def create_undirected_nxgraph_from_partition(partition,
                                             vertices_weights=None,
                                             edges_weights=None):
    # if vertices_labels is None:
    # vertices_labels
    q_list = [q for C in partition for q in C]

    G = nx.Graph()
    for qi in q_list:
        if vertices_weights is None:
            G.add_node(qi)
        else:
            G.add_node(qi, weight=vertices_weights[qi])

    # G.add_nodes_from(q_list)
    for C in partition:
        for i in range(len(C) - 1):
            for j in chain(range(i), range(i + 1, len(C))):
                if edges_weights is None:
                    G.add_edge(C[i], C[j])
                else:
                    G.add_edge(C[i], C[j],
                               weight=edges_weights[(C[i], C[j])])
    return G

# ORGANIZE - delete, this function is not used anywhere
def print_graph(graph):
    nx.draw_circular(graph, with_labels=True)
    plt.show()

# ORGANIZE - this is only used in functions_noise_model_heuristic(_OS), when an the parameter 'drawing' is set to True
# prints clusters with edges only within a cluster
def print_partition(partition):
    G = create_undirected_nxgraph_from_partition(partition)
    nx.draw_circular(G, with_labels=True)
    plt.show()
    # plt.savefig("path.png")


def __combine_colors(bad_color,
                     good_color,
                     bad_weight):
    bad_weight = bad_weight ** (1 / 2.5)

    new_color = [bad_weight * bad_color[i] + (1 - bad_weight) * good_color[i] for i in
                 range(3)] + [1]

    return new_color

# ORGANIZE - to delete; this function is used nowhere; similar functionality implemented in method draw_clusters in
# visualisation.ResultsForPlotting
def create_clusters_graph(clusters_list,
                          correlations_table,
                          true_qubits,
                          Title='',
                          errors_1q=None,

                          ):
    number_of_qubits = len(true_qubits)
    import numpy as np
    C_max = int(np.max([len(x) for x in clusters_list]))

    qubits_indices_map = dict(enumerate(true_qubits))
    # qubits_indices_rev_map = utils.map_index_to_order(true_qubits)
    # colors_list = [[256 / 256 * (i / number_of_qubits),
    #                 256 / 256 * (1 - i / number_of_qubits),
    #                 0,
    #                 1] for i in range(number_of_qubits)]
    #
    # original_color = (220 / 256, 1, 0)

    from matplotlib import pyplot as plt
    from matplotlib import rc
    import networkx as nx

    # fontname= 'Comic Sans MS'
    fontname = 'fantasy'
    plt.rcParams.update({'font.family': fontname})
    rc('font', **{'family': fontname})
    rc('text', usetex=True)
    #
    import numpy as np

    bad_color = (220 / 255, 20 / 255, 60 / 255)

    bad_color = (1, 0, 0)
    good_color = (0, 1, 0)


    node_size = 4000

    if number_of_qubits>30:
        node_size*=1/4

    # print(colors_list)
    # raise KeyError

    vertices_weights_list = []
    vertices_weights_dict = {}

    edges_weights = {}

    for i in range(number_of_qubits):
        if errors_1q is None:
            err_1q_now = 0.1
        else:
            err_1q_now = (errors_1q['q%s' % i][0, 1] +
                               errors_1q['q%s' % i][1, 0]) / 2

        vertices_weights_dict[qubits_indices_map[i]] = err_1q_now
        vertices_weights_list.append((qubits_indices_map[i], err_1q_now))

        for j in range(number_of_qubits):
            edges_weights[(qubits_indices_map[i], qubits_indices_map[j])] \
                = 1 / 2 * (correlations_table[i, j] + correlations_table[j, i])

    # print(vertices_weights_dict)
    # raise KeyError
    # vertices_weights_list = sorted(vertices_weights_list, key=lambda x: x[1])
    # # print(vertices_weights_list)
    if errors_1q is None:
        weights_colors = {u:'cornflowerblue' for u, v in vertices_weights_list}

    else:
        weights_colors = {u: __combine_colors(bad_color=bad_color,
                                              good_color=good_color,
                                              bad_weight=v / np.max(
                                                 list(vertices_weights_dict.values())))
                          for u, v in vertices_weights_list}

    # print(weights_colors)
    # raise KeyError

    clusters_mapped = [[qubits_indices_map[qi] for qi in cluster] for cluster in clusters_list]
    graph = create_undirected_nxgraph_from_partition(clusters_mapped,
                                                     vertices_weights=vertices_weights_dict,
                                                     edges_weights=edges_weights)

    positions = {}

    if C_max == 5:
        x_pos, y_pos = 0, 1.5
    else:
        x_pos, y_pos = 0, 1

    clusters_mapped = sorted(clusters_mapped, key=lambda x: len(x))

    for cluster in clusters_mapped:
        graph_cluster = create_undirected_nxgraph_from_partition([cluster],
                                                                 edges_weights=edges_weights)

        graph_design = nx.drawing.layout.circular_layout(G=graph_cluster,
                                                         # k=0.3,

                                                         # pos=positions
                                                         )

        for qi, position_now in graph_design.items():
            positions[qi] = (
                position_now[0] / len(clusters_list) + x_pos, position_now[1] / 10 + y_pos)

        if C_max == '5':
            x_pos += 0.75
            if x_pos >= 1.5:
                x_pos -= 1.5

                y_pos -= 0.3

        else:
            x_pos += 0.3

            if x_pos >= 1:
                x_pos -= 1
                y_pos -= 0.3

    if C_max == '5':
        fig = plt.figure(figsize=(25, 25))
    else:
        fig = plt.figure(figsize=(15, 15))

    # Title = f"Cmax = {C_max}, version = {version}, alpha = {alpha}"

    plt.title(Title, fontdict={'fontsize': 60})
    nx.draw(graph,
            with_labels=True,
            pos=positions,
            node_size=node_size,
            font_size=40,
            width=[graph[u][v]['weight'] / np.max(list(edges_weights.values())) * 10 + 0.5 for
                   u, v in
                   graph.edges],
            edge_color=[0, 0, 0, 1],
            node_color=[weights_colors[qi] for qi in graph.nodes]
            # width
            )

    # figname = f"Cmax = {C_max}, version = {version}, {counter}_alpha = {alpha}"
    # directory_with_figures = directory_to_open + f'figures/clusters_visualization/C_max{C_max}/'
    plt.tight_layout()
    plt.show()
