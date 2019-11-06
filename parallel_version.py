# import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import math
import parallel_version_functions as pvf
import time



comm = MPI.COMM_WORLD
rank = comm.rank
workers_size = comm.size

data = np.load('data/features_50_1000.npy')

clusters = pvf.get_initial_clusters(data)
K_curr = len(clusters)
data_size = K_curr
k = 4

if rank==0:
    start_time=time.time()

comm.Barrier()
while K_curr > k:

    clusters_indices = None
    local_clusters_indices = np.empty(2, dtype=int)
    min_pairs = None

    if rank == 0:
        print("Number of clusters: {}".format(K_curr))
        partitions_size, remainder, actual_workers_size = pvf.calculate_partitions(workers_size, K_curr)
        clusters_indices = pvf.create_clusters_indices(K_curr, partitions_size, remainder, actual_workers_size, workers_size)

    # comm.Barrier()
    comm.Scatter(clusters_indices, local_clusters_indices, root=0)

    local_min_pair = pvf.calculate_local_min_pair(clusters, local_clusters_indices, data)

    if rank == 0:
        min_pairs = np.empty([workers_size, 3])

    # comm.Barrier()
    comm.Gather(local_min_pair, min_pairs, root=0)

    if rank == 0:
        global_min_pair = pvf.calculate_global_min_pair(min_pairs)
        clusters = pvf.merging_min(clusters, global_min_pair)


    # comm.Barrier()

    clusters = comm.bcast(clusters, root=0)
    K_curr = len(clusters)
    # comm.Barrier()

if rank == 0:
    prediction = pvf.create_prediction(clusters,data_size)
    print("Number of clusters: {}".format(K_curr))
    print(clusters)
    print("Time : {}".format(time.time()-start_time))
