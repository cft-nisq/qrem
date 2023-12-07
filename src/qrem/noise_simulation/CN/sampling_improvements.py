from collections import Counter
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from qrem.functions_qrem import functions_data_analysis as fda

from qrem.common.printer import qprint
from qrem.common import convert, utils


#MOcomm - porely documented file -learn what is inside


def __format_counts_from_clusters(counts_dict,
                                  clusters_list):
    number_of_qubits = np.max([np.max(cl) for cl in clusters_list]) + 1
    # print(number_of_qubits)

    number_of_clusters = len(clusters_list)
    counts_dict_formatted = {}
    for sample_list, number_of_times in counts_dict.items():
        output_bitstring_now = ["-" for _ in range(number_of_qubits)]
        for cluster_index in range(number_of_clusters):
            cluster = clusters_list[cluster_index]
            cluster_sample = sample_list[cluster_index]
            local_bitstring_now = convert.integer_to_bitstring(
                integer=int(np.argmax(cluster_sample)),
                number_of_bits=len(cluster))
            # print(local_bitstring_now,cluster,len(cluster))
            for qubit_index in range(len(cluster)):
                output_bitstring_now[cluster[qubit_index]] = local_bitstring_now[qubit_index]

        counts_dict_formatted["".join(output_bitstring_now)] = number_of_times
    return counts_dict_formatted


def sample_from_column_of_stochastic_map(column_bitstring,
                                         stochastic_map,
                                         number_of_samples=1):
    number_of_qubits = len(list(column_bitstring))
    distro = stochastic_map[:, int(column_bitstring, 2)]
    samples = np.random.multinomial(pvals=distro,
                                    n=number_of_samples)

    counts_dictionary = {}
    for integer in range(len(samples)):
        counts_dictionary[convert.integer_to_bitstring(integer=integer,
                                                   number_of_bits=number_of_qubits)] = integer

    return samples


def sample_from_multiple_input_states(input_states_dictionary: Dict[str, int],
                                      local_stochastic_matrices,
                                      print_progres_bar=False):
    counts_dicts_list = []

    items_range = input_states_dictionary.items()

    if print_progres_bar:
        items_range = tqdm(items_range)
    #
    # local_columns_big_matrices = {input_bitstring: {cluster:stochastic_map[:, int("".join([input_bitstring[x] for x in cluster]), 2)] for cluster, stochastic_map in
    #                               local_noise_matrices.items()} for input_bitstring in input_states_dictionary.keys()}

    for input_state, number_of_samples in items_range:
        counts_dicts_list.append(sample_from_product_noise_model(input_bitstring=input_state,
                                                                 number_of_samples=number_of_samples,
                                                                 local_noise_matrices=local_stochastic_matrices,
                                                                 # local_columns_big_dictionary=local_columns_big_matrices
                                                                 ))

    total_counts_dictionary = fda.merge_multiple_counts_dictionaries(counts_dicts_list)
    return total_counts_dictionary


def __format_counts_temporary(all_samples,
                              clusters_tuple,
                              number_of_qubits,
                              counts_dictionary_now):


    samples_pairs = list(zip(*all_samples))
    counting = Counter(samples_pairs)
    for tuple_outcome, amount_of_ticks in counting.items():
        global_output_state = ['9'] * number_of_qubits
        for clust_index in range(len(clusters_tuple)):
            cluster_local = clusters_tuple[clust_index]
            for qubit_index in range(len(cluster_local)):
                global_output_state[cluster_local[qubit_index]] = \
                    tuple_outcome[clust_index][1][qubit_index]

        global_output_state = ''.join(global_output_state)
        if global_output_state not in counts_dictionary_now.keys():
            counts_dictionary_now[global_output_state] = defaultdict(float)

        counts_dictionary_now[global_output_state][
            tuple([x[0] for x in tuple_outcome])] += amount_of_ticks

    return counts_dictionary_now

# def sample_from_multiple_input_states_local(input_states_dictionary):



def sample_from_multiple_input_states_alternative_broken(input_states_dictionary: Dict[str, int],
                                                  local_stochastic_matrices,
                                                  print_progres_bar=False,
                                                  old_sampling=False):
    items_range = input_states_dictionary.items()
    clusters_list = list(local_stochastic_matrices.keys())
    clusters_tuple = tuple(clusters_list)

    number_of_clusters = len(clusters_list)
    number_of_qubits = sum([len(cluster) for cluster in clusters_list])


    local_columns_big_matrices = {input_bitstring: {
        cluster: stochastic_map[:, int("".join([input_bitstring[x] for x in cluster]), 2)] for
        cluster, stochastic_map in
        local_stochastic_matrices.items()} for input_bitstring in input_states_dictionary.keys()}

    if old_sampling:
        all_samples = {cluster: [] for cluster in clusters_list}
    counts_dictionary_now = {}


    rng = np.random.default_rng()

    # all_samples = []
    for input_bitstring, number_of_samples in  tqdm(items_range,disable= not print_progres_bar):
        local_columns = local_columns_big_matrices[input_bitstring]

        samples_pairs = []
        for cluster in clusters_list:
            local_distro_now = local_columns[cluster]
            local_output_states = rng.multinomial(n=number_of_samples,
                                                  pvals=local_distro_now)

            local_samples = []
            for further_index in range(len(local_output_states)):
                local_samples += [local_registers_dictionary[len(cluster)][further_index]] * local_output_states[further_index]

            rng.shuffle(local_samples)
            # print(local_samples)

            if old_sampling:
                all_samples[cluster] += local_samples
            else:
                samples_pairs.append(local_samples)
        # all_samples.append(samples_pairs)




        if old_sampling:
            counts_dict = defaultdict(int)
            for i in range(sum(list(input_states_dictionary.values()))):
                counts_dict[tuple([all_samples[cluster][i] for cluster in clusters_list])] += 1
            # t1=time.time()

            formatted_counts_dict = {}
            for global_output_state, number_of_counts in counts_dict.items():
                formatting_state = [None for _ in range(number_of_qubits)]

                for cluster_index in range(number_of_clusters):
                    cluster_now = clusters_list[cluster_index]
                    cluster_state_bitstring = convert.integer_to_bitstring(global_output_state[cluster_index],
                                                                       len(cluster_now))

                    for qi in range(len(cluster_now)):
                        formatting_state[cluster_now[qi]] = cluster_state_bitstring[qi]
        else:
            # print(samples_pairs[0:5])
            print(samples_pairs)
            samples_pairs = list(zip(*samples_pairs))

            print(samples_pairs[0])
            print(len(samples_pairs[0]))

            # raise KeyboardInterrupt
            # print(samples_pairs[0])
            # print(samples_pairs[1])
            counting = Counter(samples_pairs)
            for tuple_outcome, amount_of_ticks in counting.items():
                # print(tuple_outcome)
                global_output_state = ['9'] * number_of_qubits
                for clust_index in range(len(clusters_tuple)):
                    cluster_local = clusters_tuple[clust_index]
                    for qubit_index in range(len(cluster_local)):
                        global_output_state[cluster_local[qubit_index]] = tuple_outcome[clust_index][1][qubit_index]
                            # tuple_outcome[clust_index][1][qubit_index]


                global_output_state = ''.join(global_output_state)
                if global_output_state not in counts_dictionary_now.keys():
                    counts_dictionary_now[global_output_state] = defaultdict(float)

                counts_dictionary_now[global_output_state][
                    tuple([x[0] for x in tuple_outcome])] += amount_of_ticks

    formatted_counts_dict = dict(counts_dictionary_now)
    return formatted_counts_dict


def sample_from_multiple_input_states_alternative(input_states_dictionary: Dict[str, int],
                                                  local_stochastic_matrices,
                                                  print_progres_bar=False,
                                                  old_sampling=False,):
    items_range = input_states_dictionary.items()

    clusters_tuple = tuple(list(local_stochastic_matrices.keys()))

    if print_progres_bar:
        items_range = tqdm(items_range)

    local_columns_big_matrices = {input_bitstring: {
        cluster: stochastic_map[:, int("".join([input_bitstring[x] for x in cluster]), 2)] for
        cluster, stochastic_map in
        local_stochastic_matrices.items()} for input_bitstring in input_states_dictionary.keys()}

    clusters_list = list(local_stochastic_matrices.keys())
    number_of_clusters = len(clusters_list)
    number_of_qubits = sum([len(cluster) for cluster in clusters_list])

    all_samples = {cluster: [] for cluster in clusters_list}
    counts_dictionary_now = {}

    local_registers_dictionary = {len(cluster):
                                      [tuple(convert.integer_to_bitstring(index_outcome, len(cluster)))
                                       for index_outcome in range(int(2 ** len(cluster)))]
                                  for cluster in clusters_list}


    rng = np.random.default_rng()
    counts_dictionary_now = {}
    import time
    t0 = time.time()

    counting_all = {}
    for input_bitstring, number_of_samples in items_range:
        local_columns = local_columns_big_matrices[input_bitstring]

        samples_pairs = []
        for cluster in clusters_list:
            local_distro_now = local_columns[cluster]
            local_output_states = rng.multinomial(n=number_of_samples,
                                                  pvals=local_distro_now)

            local_samples = []
            for further_index in range(len(local_output_states)):
                local_samples += [local_registers_dictionary[len(cluster)][further_index]] * local_output_states[further_index]

            rng.shuffle(local_samples)
            # print(local_samples)

            if old_sampling:
                all_samples[cluster] += local_samples
            else:
                samples_pairs.append(local_samples)
            # print(len(samples_pairs))
        # print(samples_pairs[0])

        if old_sampling:
            counts_dict = defaultdict(int)
            for i in range(sum(list(input_states_dictionary.values()))):
                counts_dict[tuple([all_samples[cluster][i] for cluster in clusters_list])] += 1
            # t1=time.time()

            formatted_counts_dict = {}
            for global_output_state, number_of_counts in counts_dict.items():
                formatting_state = [None for _ in range(number_of_qubits)]

                for cluster_index in range(number_of_clusters):
                    cluster_now = clusters_list[cluster_index]
                    cluster_state_bitstring = convert.integer_to_bitstring(global_output_state[cluster_index],
                                                                       len(cluster_now))

                    for qi in range(len(cluster_now)):
                        formatting_state[cluster_now[qi]] = cluster_state_bitstring[qi]
        else:
            # print(samp    les_pairs[0])
            # print(clusters_list)
            # print(number_of_clusters)
            # print(samples_pairs[0][0])
            # print([samples_pairs[k][0] for k in range(number_of_clusters)])
            samples_pairs = list(zip(*samples_pairs))
            # print(len(samples_pairs))
            #
            # print(samples_pairs[0])


            counting = Counter(samples_pairs)
            for tuple_outcome, amount_of_ticks in counting.items():
                # print(tuple_outcome)
                global_output_state = ['9'] * number_of_qubits
                for clust_index in range(len(clusters_tuple)):
                    cluster_local = clusters_tuple[clust_index]
                    for qubit_index in range(len(cluster_local)):
                        global_output_state[cluster_local[qubit_index]] = \
                            tuple_outcome[clust_index][qubit_index]

                global_output_state = ''.join(global_output_state)
                if global_output_state not in counts_dictionary_now.keys():
                    counts_dictionary_now[global_output_state] = 0

                counts_dictionary_now[global_output_state] += amount_of_ticks

    t1= time.time()
    qprint("This took:",t1-t0)
    formatted_counts_dict = dict(counts_dictionary_now)
    return formatted_counts_dict

def _get_clusters_samples(counts_dictionary,
                            local_stochastic_matrices,
                            local_columns_big_matrices,
                            local_registers_dictionary
                          ):
    rng = np.random.default_rng()
    items_range = counts_dictionary.items()

    clusters_list = list(local_stochastic_matrices.keys())

    counting_all = []
    for input_bitstring, number_of_samples in items_range:
        local_columns = local_columns_big_matrices[input_bitstring]

        samples_pairs = []
        for cluster in clusters_list:
            local_distro_now = local_columns[cluster]
            local_output_states = rng.multinomial(n=number_of_samples,
                                                  pvals=local_distro_now)

            local_samples = []
            for further_index in range(len(local_output_states)):
                local_samples += [local_registers_dictionary[len(cluster)][further_index]] * \
                                 local_output_states[further_index]

            rng.shuffle(local_samples)

            samples_pairs.append(local_samples)

        samples_pairs = list(zip(*samples_pairs))

        counting_all = counting_all + samples_pairs
    return counting_all

import time
def _get_clusters_samples_mp(items_range,
                             other_stuff
                          ):
    local_stochastic_matrices = other_stuff['local_stochastic_matrices']
    local_columns_big_matrices = other_stuff['local_columns_big_matrices']
    local_registers_dictionary = other_stuff['local_registers_dictionary']


    rng = np.random.default_rng()

    clusters_list = list(local_stochastic_matrices.keys())

    counting_all = []
    # counting_all = {}
    for input_bitstring, number_of_samples in items_range:
        local_columns = local_columns_big_matrices[input_bitstring]

        samples_pairs = []
        for cluster in clusters_list:
            local_distro_now = local_columns[cluster]
            local_output_states = rng.multinomial(n=number_of_samples,
                                                  pvals=local_distro_now)

            local_samples = []
            for further_index in range(len(local_output_states)):
                local_samples += [local_registers_dictionary[len(cluster)][further_index]] * \
                                 local_output_states[further_index]

            rng.shuffle(local_samples)

            samples_pairs.append(local_samples)

        samples_pairs = list(zip(*samples_pairs))

        # counting_all = counting_all + samples_pairs

        counting_all = counting_all+samples_pairs

    return {f"{time.time()}, {np.random.randint(0,1000)}":counting_all}


def _count_batches(counting_all,
                   number_of_batches,
                   number_of_qubits,
                   clusters_tuple):


    total_length = len(counting_all)
    batches_size = int(total_length//number_of_batches)

    counts_dicts = []
    for batch_index in range(number_of_batches):
        if batch_index<number_of_batches-1:
            counts_now = counting_all[batch_index*batches_size:(batch_index+1)*batches_size]
        else:
            counts_now = counting_all[batch_index * batches_size:]

        counts_dictionary_now = {}
        counting = Counter(counts_now)
        for tuple_outcome, amount_of_ticks in counting.items():
            global_output_state = ['9'] * number_of_qubits
            for clust_index in range(len(clusters_tuple)):
                cluster_local = clusters_tuple[clust_index]
                for qubit_index in range(len(cluster_local)):
                    global_output_state[cluster_local[qubit_index]] = \
                        tuple_outcome[clust_index][qubit_index]
            global_output_state = ''.join(global_output_state)
            if global_output_state not in counts_dictionary_now.keys():
                counts_dictionary_now[global_output_state] = 0

            counts_dictionary_now[global_output_state] += amount_of_ticks
        counts_dicts.append(counts_dictionary_now)
    return counts_dicts


def _count_batches_mp(batches_tuples,
                   other_stuff):
    # number_of_batches = other_stuff['number_of_batches']
    number_of_qubits = other_stuff['number_of_qubits']
    clusters_tuple = other_stuff['clusters_tuple']

    counts_dicts = []
    for batch_of_counts in batches_tuples:
        for counts_now in batch_of_counts:
            counts_dictionary_now = {}
            counting = Counter(counts_now)
            for tuple_outcome, amount_of_ticks in counting.items():
                global_output_state = ['9'] * number_of_qubits
                for clust_index in range(len(clusters_tuple)):
                    cluster_local = clusters_tuple[clust_index]
                    for qubit_index in range(len(cluster_local)):
                        global_output_state[cluster_local[qubit_index]] = \
                            tuple_outcome[clust_index][qubit_index]
                global_output_state = ''.join(global_output_state)
                if global_output_state not in counts_dictionary_now.keys():
                    counts_dictionary_now[global_output_state] = 0

                counts_dictionary_now[global_output_state] += amount_of_ticks
            counts_dicts.append(counts_dictionary_now)

    return {f"{time.time()}, {np.random.randint(0,100)}":counts_dicts}


def sample_from_multiple_input_states_alternative_batches(input_states_dictionary: Dict[str, int],
                                                  local_stochastic_matrices,
                                                          number_of_batches,
                                                  print_progres_bar=False,
                                                  old_sampling=False,
                                                          multiprocessing=False):
    items_range = input_states_dictionary.items()
    clusters_list = list(local_stochastic_matrices.keys())
    clusters_tuple = tuple(list(local_stochastic_matrices.keys()))

    # if print_progres_bar:
    #     items_range = tqdm(items_range)

    local_columns_big_matrices = {input_bitstring: {
        cluster: stochastic_map[:, int("".join([input_bitstring[x] for x in cluster]), 2)] for
        cluster, stochastic_map in
        local_stochastic_matrices.items()} for input_bitstring in input_states_dictionary.keys()}


    # number_of_clusters = len(clusters_list)
    number_of_qubits = sum([len(cluster) for cluster in clusters_list])

    # all_samples = {cluster: [] for cluster in clusters_list}
    # counts_dictionary_now = {}

    local_registers_dictionary = {len(cluster):
                                      [tuple(convert.integer_to_bitstring(index_outcome, len(cluster)))
                                       for index_outcome in range(int(2 ** len(cluster)))]
                                  for cluster in clusters_list}

    rng = np.random.default_rng()
    # counts_dictionary_now = defaultdict(float)
    import time
    t0 = time.time()

    # if multiprocessing:

    t0 = time.time()
    if multiprocessing:

        main_tuples = list(items_range)

        res_mp = utils.wrapped_multiprocessing_function(tuple_of_main_arguments=main_tuples,
                                                        additional_kwargs={'local_stochastic_matrices':local_stochastic_matrices,
                                                                           'local_registers_dictionary':local_registers_dictionary,
                                                                           'local_columns_big_matrices':local_columns_big_matrices},
                                                        function_to_multiprocess=_get_clusters_samples_mp,
                                                        printing=True)
        print(time.time()-t0)
        counting_all = []
        for value in res_mp.values():
            counting_all = counting_all+value
        # print(time.time()-t0)
        #
        print(len(list(res_mp.values())))



    else:
        counting_all = _get_clusters_samples(counts_dictionary=input_states_dictionary,
                                             local_stochastic_matrices=local_stochastic_matrices,
                                             local_registers_dictionary=local_registers_dictionary,
                                             local_columns_big_matrices=local_columns_big_matrices)
    rng.shuffle(counting_all)
    rng.shuffle(counting_all)
    rng.shuffle(counting_all)
    t1 = time.time()

    if multiprocessing:
        total_length = len(counting_all)
        batches_size = int(total_length // number_of_batches)

        batches_mp = []
        for batch_index in range(number_of_batches):
            if batch_index < number_of_batches - 1:
                counts_now = counting_all[batch_index * batches_size:(batch_index + 1) * batches_size]
            else:
                counts_now = counting_all[batch_index * batches_size:]

            batches_mp.append((counts_now,))


        res_mp = utils.wrapped_multiprocessing_function(tuple_of_main_arguments=batches_mp,
                                                        additional_kwargs={'number_of_qubits':number_of_qubits,
                                                                           'clusters_tuple':clusters_tuple,
                                                                           },
                                                        function_to_multiprocess=_count_batches_mp,
                                                        printing=True)

        # print(len(list(res_mp.values())))

        counts_dicts = []
        for value in res_mp.values():
            counts_dicts = counts_dicts+value

    else:
        counts_dicts = _count_batches(counting_all,
                       number_of_batches,
                       number_of_qubits,
                       clusters_tuple)

    t2 = time.time()
    qprint("Sampling took:", t1 - t0)
    qprint("Counting took:", t2 - t1)
    # formatted_counts_dict = dict(counts_dictionary_now)
    return counts_dicts
           # normalizations


def sample_from_product_noise_model(input_bitstring: str,
                                    local_noise_matrices: Dict[Tuple[int], np.ndarray],
                                    number_of_samples: int,
                                    local_columns_big_dictionary=None):
    clusters_list = list(local_noise_matrices.keys())
    number_of_clusters = len(clusters_list)
    number_of_qubits = len(input_bitstring)

    if local_columns_big_dictionary is not None:
        if not isinstance(input_bitstring,str):
            input_bitstring = ''.join(input_bitstring)
            # print(local_columns_big_dictionary)
        local_columns = local_columns_big_dictionary[input_bitstring]
    else:
        local_columns = {
            cluster: stochastic_map[:, int("".join([input_bitstring[x] for x in cluster]), 2)]
            for cluster, stochastic_map in local_noise_matrices.items()}

    new_sampling = True

    counts_dict = defaultdict(int)

    if new_sampling:
        all_samples = {}
        # t0 = time.time()
        rng = np.random.default_rng()
        for cluster in clusters_list:
            # t2 =time.time()
            local_distro_now = local_columns[cluster]
            # rng = np.random.default_rng()
            local_output_states = rng.multinomial(n=number_of_samples,
                                                  pvals=local_distro_now)

            local_samples = []
            for index_outcome in range(len(local_output_states)):
                local_samples += [convert.integer_to_bitstring(index_outcome, len(cluster))] * \
                                 local_output_states[index_outcome]

            #         list(np.full(shape=local_output_states[index_outcome],
            #                                   fill_value=convert.integer_to_bitstring(index_outcome,
            #                                                                       len(cluster))
            #                                   ))
            # # t3 = time.time()

            rng.shuffle(local_samples)
            all_samples[cluster] = local_samples

        # t1 =time.time()
        # t5 = time.time()
        for i in range(number_of_samples):
            global_output_state = [None for __ in range(number_of_qubits)]
            for cluster in clusters_list:
                local_output_state = all_samples[cluster][i]
                for qubit_index in range(len(cluster)):
                    global_output_state[cluster[qubit_index]] = local_output_state[qubit_index]

            global_output_state = ''.join(global_output_state)
            counts_dict[global_output_state] += 1

        counts_dict_formatted = counts_dict
        # t6 = time.time()
        # if (t6-t5)>(t1 - t0):
        #     print('sampling:', t1 - t0)
        #     print('formatting:', t6-t5)
    else:

        for _ in range(number_of_samples):
            output_now = np.zeros(number_of_clusters, dtype=tuple)
            for cluster_index in range(number_of_clusters):
                cluster = clusters_list[cluster_index]
                local_distro = local_columns[cluster]
                local_sample_now = np.random.multinomial(pvals=local_distro,
                                                         n=1)
                output_now[cluster_index] = tuple(local_sample_now)
            counts_dict[tuple(output_now)] += 1

        counts_dict_formatted = __format_counts_from_clusters(counts_dict=counts_dict,
                                                              clusters_list=clusters_list)

    # for cluster_index in range(number_of_clusters):
    #     cluster = clusters_list[cluster_index]
    #     local_distro = local_columns[cluster]
    #     local_sample_now = np.random.multinomial(pvals=local_distro,
    #                                              n=number_of_samples)
    #
    #
    #
    #
    #

    return counts_dict_formatted
#
# if __name__ == 'main':
#
#     local_noise_matrices_base = {
#         (0, 1, 7): array([[7.57029412e-01, 1.74052381e-01, 1.01661290e-01, 2.20660714e-02,
#                            1.97447458e-01, 4.16703125e-02, 2.80763636e-02, 5.30517241e-03],
#                           [1.76901961e-01, 5.75815873e-01, 2.31080645e-02, 4.25196429e-02,
#                            1.79110169e-01, 1.54307813e-01, 1.62727273e-02, 1.00879310e-02],
#                           [1.86215686e-02, 1.22015873e-02, 7.29356452e-01, 3.43464286e-01,
#                            6.38983051e-03, 2.61562500e-03, 2.53301818e-01, 7.51551724e-02],
#                           [1.51764706e-03, 1.30714286e-02, 1.02304839e-01, 4.27332143e-01,
#                            3.04745763e-03, 3.31718750e-03, 1.08665455e-01, 9.43500000e-02],
#                           [3.59137255e-02, 9.53571429e-02, 4.74838710e-03, 8.10535714e-03,
#                            4.08279661e-01, 1.30548438e-01, 5.86600000e-02, 1.96810345e-02],
#                           [9.04901961e-03, 1.23300000e-01, 1.08709677e-03, 6.53214286e-03,
#                            1.92888136e-01, 6.40910937e-01, 2.06563636e-02, 5.24793103e-02],
#                           [8.86274510e-04, 4.48412698e-03, 3.30709677e-02, 9.45196429e-02,
#                            1.07423729e-02, 1.02140625e-02, 4.17769091e-01, 2.78255172e-01],
#                           [8.03921569e-05, 1.71746032e-03, 4.66290323e-03, 5.54607143e-02,
#                            2.09491525e-03, 1.64156250e-02, 9.65981818e-02, 4.64686207e-01]]),
#         (2,): array([[0.96430696, 0.08032241],
#                      [0.03569304, 0.91967759]]), (3,): array([[0.91271138, 0.03638376],
#                                                               [0.08728862, 0.96361624]]),
#         (4,): array([[0.9692458, 0.05537525],
#                      [0.0307542, 0.94462475]]), (5,): array([[0.92232226, 0.77049618],
#                                                              [0.07767774, 0.22950382]]),
#         (6,): array([[0.95776162, 0.04182947],
#                      [0.04223838, 0.95817053]]), (8,): array([[0.97306899, 0.1251291],
#                                                               [0.02693101, 0.8748709]]),
#         (9,): array([[0.93448873, 0.07061062],
#                      [0.06551127, 0.92938938]]), (10,): array([[0.98673934, 0.37029401],
#                                                                [0.01326066, 0.62970599]]),
#         (11,): array([[0.97976416, 0.0407595],
#                       [0.02023584, 0.9592405]]), (12,): array([[0.9721489, 0.04683174],
#                                                                [0.0278511, 0.95316826]]),
#         (13,): array([[0.95634948, 0.0379854],
#                       [0.04365052, 0.9620146]]), (14,): array([[0.98572483, 0.04384909],
#                                                                [0.01427517, 0.95615091]]),
#         (16,): array([[0.9194975, 0.06026644],
#                       [0.0805025, 0.93973356]]),
#         (15, 17): array([[9.47176812e-01, 3.60669643e-02, 4.42798387e-02, 8.97796610e-03],
#                          [2.44862319e-02, 9.33566964e-01, 2.00887097e-03, 3.23272881e-01],
#                          [2.74797101e-02, 1.12142857e-03, 9.26055645e-01, 3.32940678e-02],
#                          [8.57246377e-04, 2.92446429e-02, 2.76556452e-02, 6.34455085e-01]]),
#         (18,): array([[0.98574926, 0.05787736],
#                       [0.01425074, 0.94212264]]), (19,): array([[0.96944702, 0.07846338],
#                                                                 [0.03055298, 0.92153662]])}
#
#     p, q, p2, q2 = 0.2, 0.3, 0.1, 0.15
#
#     p3, q3 = 0.05, 0.09
#     # local_noise_matrices_base = {(0,): np.array([[1 - p, q],
#     #                                              [p, 1 - q]]),
#     #                              (1,): np.array([[1 - p2, q2],
#     #                                              [p2, 1 - q2]]),
#     #                              (2,): np.array([[1 - p3, q3],
#     #                                              [p3, 1 - q3]]),
#     #                              }
#     number_of_samples = 10 ** 5
#     number_of_qubits = 10
#     dimension = int(2 ** number_of_qubits)
#
#     local_noise_matrices = {}
#     for cluster in local_noise_matrices_base.keys():
#         if np.any(np.array(cluster) >= number_of_qubits):
#             break
#         else:
#             local_noise_matrices[cluster] = local_noise_matrices_base[cluster].real
#
#     input_bitstring = []
#     for i in range(number_of_qubits):
#         if i % 2 == 0:
#             input_bitstring.append("0")
#         else:
#             input_bitstring.append("0")
#
#     input_bitstring = "".join(input_bitstring)
#
#     if number_of_qubits <= 10:
#         global_noise_matrix = np.eye(dimension, dtype=float)
#         for cluster, local_noise_matrix in local_noise_matrices.items():
#             print(cluster)
#             global_noise_matrix = global_noise_matrix @ quanf.embed_operator_in_bigger_hilbert_space(
#                 number_of_qubits=number_of_qubits,
#                 local_operator=local_noise_matrix,
#                 global_indices=cluster).real
#
#         true_distro = global_noise_matrix[:, int(input_bitstring, 2)]
#
#
#     t0 = time.time()
#     counts_dict_formatted_first = sample_from_product_noise_model(input_bitstring=input_bitstring,
#                                     local_noise_matrices=local_noise_matrices,
#                                     number_of_samples=number_of_samples)
#
#     t1 = time.time()
#
#
#     clusters_list = list(local_noise_matrices.keys())
#     number_of_clusters = len(clusters_list)
#
#     local_columns = {cluster: stochastic_map[:, int("".join([input_bitstring[x] for x in cluster]), 2)]
#                      for cluster, stochastic_map in local_noise_matrices.items()}
#
#     counts_dict = defaultdict(float)
#     cluster_samples_original = {}
#     for cluster_index in tqdm(range(number_of_clusters)):
#         cluster = clusters_list[cluster_index]
#         local_distro = local_columns[cluster]
#         local_samples_now = np.random.multinomial(pvals=local_distro,
#                                                   n=number_of_samples)
#         cluster_samples_original[cluster] = local_samples_now
#
#
#
#     # samples_clusters_shuffled = {}
#
#     samples_list_shuffled = []
#     for cluster, local_samples in tqdm(cluster_samples_original.items()):
#         cluster_samples_local = []
#         for index in range(len(local_samples)):
#             cluster_samples_local = cluster_samples_local + list(np.full(shape=local_samples[index],
#                                                                          fill_value=convert.integer_to_bitstring(
#                                                                              integer=index,
#                                                                              number_of_bits=len(cluster))))
#         local_samples_shuffled = anf_pt.properly_shuffle_stuff(cluster_samples_local)
#         samples_list_shuffled.append(local_samples_shuffled)
#
#     samples_list_shuffled = np.array(samples_list_shuffled)
#
#     counts_dict = defaultdict(float)
#     for sample_index in range(number_of_samples):
#         output_now = np.zeros(number_of_clusters, dtype=tuple)
#         for cluster_index in range(number_of_clusters):
#             cluster = clusters_list[cluster_index]
#             local_sample_now = samples_list_shuffled[cluster_index,sample_index]
#             output_now[cluster_index] = tuple(local_sample_now)
#         counts_dict[tuple(output_now)] += 1
#
#     counts_dict_formatted_second = __format_counts_from_clusters(counts_dict=counts_dict,
#                                                           clusters_list=clusters_list)
#
#     t2 =time.time()
#
#
#     print(t1-t0,t2-t1)
#
#     # counts_dict_formatted_first
#
#
#     if number_of_qubits <= 10:
#         sampled_direct = np.array(np.random.multinomial(pvals=true_distro / sum(true_distro),
#                                                         n=number_of_samples)) / number_of_samples
#
#         estimated_distro_first = fda.convert_counts_dictionary_to_probability_distribution(counts_dictionary=counts_dict_formatted_first)
#         estimated_distro_second = fda.convert_counts_dictionary_to_probability_distribution(
#             counts_dictionary=counts_dict_formatted_second)
#
#         print(1/2*np.linalg.norm(true_distro-sampled_direct))
#         print(1 / 2 * np.linalg.norm(true_distro - estimated_distro_first))
#         print(1 / 2 * np.linalg.norm(true_distro - estimated_distro_second))
#
#
#
