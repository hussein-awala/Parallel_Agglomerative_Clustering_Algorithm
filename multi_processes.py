import multiprocessing as mp
import numpy as np
import math
import multi_processes_functions as mpf
import time

start_time=time.time()
# Load the data
data = np.load('data/features_50_2.npy')
# Intialize the clusters
clusters = mpf.get_initial_clusters(data)

K = len(clusters)

# Loop until reach the desired number of clusters
while K>5:
	print("Number of clusters: {}".format(K))
	# Get the number of CPUs
	cpus = mp.cpu_count()
	# Compute the rest from the division the number of cluster over the number of CPUs
	rest = K % cpus
	# Calculate the size of each slice
	latSliceCount = int((K - rest) / cpus)
	# Prepare the result list
	results = []
	# Define a Pool object
	pool = mp.Pool()
	# loop on all the CPUs
	for i in range(cpus):
		# Calculate the start and the end of the slice
		start = i * latSliceCount
		end = ((i + 1) * latSliceCount) -1 if i != (cpus - 1) else (((i + 1) * latSliceCount) + rest) -1
		# Add the result of all the CPUs to the list
		results.append(pool.apply_async(mpf.min_of_slice, (start, end,clusters)))
	# Create a list of this mins
	mins=[res.get(timeout=100) for res in results]
	# Find the min between them
	min_iter=mpf.max_nb
	arg_min=(-1,-1)
	for i in mins:
		if(i[0]<min_iter):
			min_iter=i[0]
			arg_min=i[1]
	# Merge the 2 clusters having this min
	clusters=mpf.merging_min(clusters,arg_min)
	K = len(clusters)

print("Time : {}".format(time.time()-start_time))
