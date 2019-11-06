import matplotlib.pyplot as plt
import numpy as np
import serial_version_functions as svf
import time


def agglomerative_clustring(data, k):
"""[summary]
    This functions implements the agglomerative algorithm
Arguments:
    data {numpy array} -- the observations
    k {int} -- the final number of clusters

Returns:
    clusters [List of lists] -- the final clusters
    prediction [vector] -- vector correspond to each observation belong to which cluster
"""""" """

    clusters = svf.get_initial_clusters(data)
    K = len(clusters)

    while K > k:
        print("Number of clusters: {}".format(K))
        diss_dict = svf.create_diss_dict(clusters, data)
        min_pair =  svf.get_min_pair(diss_dict)
        clusters = svf.merging_min(clusters, min_pair)
        K = len(clusters)

    print("Number of clusters: {}".format(K))
    prediction = svf.create_prediction(clusters, data.shape[0])
    return clusters, prediction



if __name__ == '__main__':

    start_time=time.time()

    data = np.load('data/features_50_2.npy')
    clusters, prediction = agglomerative_clustring(data, 4)

    print(clusters)

    print("Time : {}".format(time.time()-start_time))
