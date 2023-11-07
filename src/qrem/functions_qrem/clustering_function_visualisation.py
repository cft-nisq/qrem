import matplotlib.pyplot as plt

#TODO_PP ask Asia which functionalities are covered by her module
# ORGANIZE: these functions can all be deleted; they are either copied to functions_benchmarks or not used anywhere
plt.rcParams['font.family'] = 'sans-serif'

# ORGANIZE - this can be deleted, it's only used in function plot_clustering_results below, which can be deleted
def create_clustering_function_values_list(clustering_data,locality):
    clustering_results=[[] for x in range(1,locality)]
    for clusters, value in clustering_data.items():
        if type(clustering_data[clusters])==list:
            for item in clustering_data[clusters]:
                clustering_results[item[0]-2].append(item[1:3])
    return clustering_results

# ORGANIZE - this can be deleted, as it's only used in visualise_clustering_function below, which can be deleted
def plot_clustering_results(clustering_results_list):
    for item in clustering_results_list:
        alpha_list = []
        clustering_function_values_list = []
        for element in item:
            alpha_list.append(element[0])
            clustering_function_values_list.append(element[1])

        plt.plot(alpha_list, clustering_function_values_list ,'ro')
        plt.show()
        #print(alpha_list)
        #print(clustering_function_values_list)

# ORGANIZE - this isn't used anywhere
def visualise_clustering_function(clustering_data,locality):
    plot_clustering_results( create_clustering_function_values_list(clustering_data,locality))

# ORGANIZE - the same function is copied into functions_benchmarks and is only used in code as an import from there
# the function can be deleted here
# function that plots value of benchmark vs values of alpha for a given set of test/traning data
def create_plots(plot_x_list, plot_y_list, separable_value, label):
    legend = []
    for i in range(len(plot_x_list)):

        # this if exludes plotting results for uncorelated noise model, they are added as a line below
        if len(plot_x_list[i]) > 1:
            plt.plot(plot_x_list[i], plot_y_list[i], marker='o', linestyle='none')
            title = f"Mitigation " + label
            legend.append(f'Locality {i + 1}')
            min_val = min(plot_y_list[i])
            if min_val < 10 ** (-2):
                min_x = plot_x_list[i][plot_y_list[i].index(min_val)]
                plt.annotate(str(np.round(min_val, 5)), xy=(min_x, min_val))

            plt.title(title)
            fname = title + ".png"
    plt.ylim(-0.001, 0.075)

    legend.append("Uncorrelated")

    plt.plot([plot_x_list[1][0], plot_x_list[1][-1]], [separable_value, separable_value])
    plt.annotate(str(np.round(separable_value, 5)), xy=(2.5, separable_value))
    plt.legend(legend)
    plt.savefig(fname)

# ORGANIZE - the same function is copied into functions_benchmarks and is only used in code as an import from there
# the function can be deleted here
#function that prepares data for plots
#input: clusters dictionary: dictionary with clusters 2) mitigation results: dictionary with clusters and mitigation reults
#output: three 2d lists with the first index encoding locality of cluster and the second one: 1) values of alpha 2) values of median for a cluster corresponding to agiven cluster 3) values of mean for a cluster corresponding to agiven cluster
#eg plot_median_list[2][0] encodes clustering for locality 2+1=3 corresponding to the alpha value plot_alpha_list[2][0]
def cerate_data_alpha_plot(clusters_dictionary, mitigation_data):
    parameters_dictionary = {}
    # loop over cluster assigments from clustering algorithm
    for cluster, clustering_params in clusters_dictionary.items():
        # this dictionary will store localities and corresponding alphas for a given cluster as well as median and mean for clustering
        locality_dic = {}
        max_locality = 0

        if clustering_params != None:
            # this is a loop over different configurations of localities and alpha for a given cluster structure
            for item in clustering_params:
                # data is rewritten as dictionary of a form locality : list of alphas
                if item[0] > max_locality:
                    max_locality = item[0]
                if item[0] in locality_dic.keys():
                    temp_list = locality_dic[item[0]]
                    temp_list.append(item[1])
                    locality_dic[item[0]] = temp_list

                else:

                    locality_dic[item[0]] = [item[1]]

        else:
            locality_dic[1] = [1]

        # to a given cluster value of meadian and mean of mitigation benchmark is added
        locality_dic['meadian'] = mitigation_data[cluster]['median']
        locality_dic['mean'] = mitigation_data[cluster]['mean']
        parameters_dictionary[cluster] = locality_dic
        # print(parameters_dictionary[cluster])

    plot_alpha_list = [[] for i in range(2, max_locality + 2)]
    plot_median_list = [[] for i in range(2, max_locality + 2)]
    plot_mean_list = [[] for i in range(2, max_locality + 2)]

    for clusters, parameters in parameters_dictionary.items():
        for i in range(1, max_locality + 1):
            if i in parameters.keys():
                for item in parameters[i]:
                    plot_alpha_list[i - 1].append(item)
                    plot_median_list[i - 1].append(parameters['meadian'])
                    plot_mean_list[i - 1].append(parameters['mean'])

    return plot_alpha_list, plot_median_list, plot_mean_list


# function calculates mean and median of mitigation benchmarks over a test/traning set
# input: benchmarks_results_mitigation - dictionary with mitigation results, traning_set - indicies of hamiltonians that form the traning set

#ORGANIZE - this function can be deleted; it is copied into functions_benchmarks and is only used as an import from there
def calculate_results_test_set(benchmark_results_mitigation, hamiltonian_set):
    # loop over differnt noise models (keys of benchmark_results_mitigation dictionary) and benchmark results for individual hamiltonians
    traning_median_mean = {}
    for noise_model, result in benchmark_results_mitigation.items():
        median, mean = calculate_median_mean(result['errors_list'], hamiltonian_set)
        traning_median_mean[noise_model] = {'median': median, 'mean': mean}
    return traning_median_mean

#ORGANIZE - this function can be deleted; it is copied into functions_benchmarks and is only used there
# helper function that calculates median and mean for a single noise model
def calculate_median_mean(errors_dictionary, hamiltonian_set):
    median_list = []
    mean = 0.0
    for index in hamiltonian_set:
        error = errors_dictionary[index]
        mean = mean + error
        median_list.append(error)
    mean = mean / len(hamiltonian_set)
    return np.median(median_list), mean