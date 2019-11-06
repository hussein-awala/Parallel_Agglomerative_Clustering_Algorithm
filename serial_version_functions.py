import numpy as np
import math



def disimilarity_between_two_elements(x,y):
"""[summary]
    calculate the euclidean distance between two points

Arguments:
    x {vector} -- a point 
    y {vector} -- a point

Returns:
    float -- euclidean distance
"""
    return np.sqrt(np.sum(np.square(x-y)))


def disimilarity_between_two_clusters(x,y, data):
    """[summary]
    calculate the average euclidean distance between two clusters

    Arguments:
        x {[type]} -- list of points
        y {[type]} -- list of points
        data {[type]} -- the observations
    
    Returns:
    float -- euclidean distance
    """
    dis=0.0

    for i in x:
        for j in y:
            dis=dis+disimilarity_between_two_elements(data[i],data[j])

    return dis/(len(x)*len(y))


def get_min_pair(diss_dict):
    return min(diss_dict, key = lambda x: diss_dict.get(x))

def get_initial_clusters(data):
    clusters = []

    for i in range(data.shape[0]):
        cluster = [i]
        clusters.append(cluster)

    return clusters

def create_diss_dict(clusters, data):
    

    local_diss_dict = {}

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):

            key = (i, j)
            diss = disimilarity_between_two_clusters(clusters[i], clusters[j], data)
            local_diss_dict[key] = diss

    return local_diss_dict

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
