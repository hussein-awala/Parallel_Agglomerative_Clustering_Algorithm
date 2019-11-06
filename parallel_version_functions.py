import numpy as np
import math



def disimilarity_between_two_elements(x,y):

    return np.sqrt(np.sum(np.square(x-y)))


def disimilarity_between_two_clusters(x,y, data):
    dis=0.0

    for i in x:
        for j in y:
            dis=dis+disimilarity_between_two_elements(data[i],data[j])

    return dis/(len(x)*len(y))


def get_initial_clusters(data):
    clusters = []

    for i in range(data.shape[0]):
        cluster = [i]
        clusters.append(cluster)

    return clusters

def create_local_diss_dict(clusters, local_clusters_indices, data):

    local_diss_dict = {}

    if local_clusters_indices[0] == -1 and local_clusters_indices[1] == -1 :
        local_diss_dict = {}
    else:
        for i in range(local_clusters_indices[0], local_clusters_indices[1]):
            for j in range(i + 1, len(clusters)):

                key = (i, j)
                diss = disimilarity_between_two_clusters(clusters[i], clusters[j], data)
                local_diss_dict[key] = diss

    return local_diss_dict

def get_local_min_pair(local_diss_dict):
    min = math.inf
    min_key = (0,0)

    for k, v in local_diss_dict.items():
        if v < min:
            min = v
            min_key = k

    local_min_pair = np.array([min_key[0], min_key[1], min])

    return local_min_pair

def merging_min(clusters, min_pair):


    points_j = clusters[int(min_pair[1])]
    clusters[int(min_pair[0])] = clusters[int(min_pair[0])] + points_j
    del clusters[int(min_pair[1])]

    return clusters

def create_prediction(clusters, data_size):
    prediction = np.zeros(data_size)
    k = 0

    for cluster in clusters:
        for i in clusters:
            prediction[i] = k

        k = k + 1

    return prediction


def calculate_partitions(workers_size, clusters_size):
    clusters_size = clusters_size - 1

    if clusters_size >= workers_size:
        partition_size = math.floor(clusters_size / workers_size)
        remainder = clusters_size % workers_size
        actual_workers_size = workers_size

    else:
        partition_size = 1
        remainder = 0
        actual_workers_size = clusters_size

    return partition_size, remainder, actual_workers_size


def create_clusters_indices(clusters_size, partitions_size, remainder, active_workers_size, workers_size):
    p = 0
    clusters_indices = np.empty([workers_size, 2], dtype=int)

    for w in range(workers_size):

        if w < active_workers_size:

            current_partition = partitions_size

            if remainder > 0:
                current_partition = current_partition + 1
                remainder = remainder - 1

            local_assigned_clusters_indices = np.array([p, p + current_partition])
            p = p + current_partition

        else:
            local_assigned_clusters_indices = np.empty((1,2))
            local_assigned_clusters_indices[:] = -1

        clusters_indices[w] = local_assigned_clusters_indices


    return clusters_indices


def calculate_local_min_pair(clusters, local_clusters_indices, data):

    local_diss_dict = create_local_diss_dict(clusters, local_clusters_indices, data)
    local_min_pair = get_local_min_pair(local_diss_dict)

    return local_min_pair

def calculate_global_min_pair(min_pairs):

    min_index = np.argmin(min_pairs[:,2])

    return [min_pairs[min_index,0], min_pairs[min_index,1]]
