from mpi4py import MPI
import numpy as np
from time import sleep
import time
import numpy as np

max_number=10000000

def get_initial_clusters(data):
	"""
	Input: ndarray of all the data
	Output: list contains each element as a list
	"""

	clusters = []
	for i in range(data.shape[0]):
		cluster = [data[i]]
		clusters.append(cluster)

	return clusters

def disimilarity_between_two_elements(x,y):
	"""
	Input: 2 data points
	Output: distance between these 2 points
	"""

	return np.sqrt(np.sum(np.square(x-y)))


def disimilarity_between_two_clusters(x,y):
	"""
	Input: 2 clusters(list of data points)
	Output: distance between these 2 clusters
	"""
	dis=0.0

	for i in x:
		for j in y:
			dis=dis+disimilarity_between_two_elements(i,j)

	return dis/(len(x)*len(y))

def min_of_slice(start,end,clusters):
	"""
	Input: all the clusters, start index, end index
	Output: the minimum distance between 2 clusters in the range and the indexes of these clusters
	"""
	min_dist=max_nb
	arg_min=(-1,-1)
	for i in range(len(clusters)):
		for j in range(start,end+1):
			if(i==j):
				continue
			dist=disimilarity_between_two_clusters(clusters[i],clusters[j])
			if dist<min_dist:
				min_dist=dist
				arg_min=(i,j)
	return (min_dist,arg_min)

def merging_min(clusters, min_pair):
	"""
	Input: all the clusters, pair of index
	Output: list of clusters after merging the 2 clusters having the indexes in the min_pair
	"""

	points_j = clusters[int(min_pair[1])]
	clusters[int(min_pair[0])] = clusters[int(min_pair[0])] + points_j
	del clusters[int(min_pair[1])]

	return clusters

def send_two_clusters(data,start,nb_worker):
	"""
	Input: all the combinations between the clusters, start index, number of worker
	Function distribute the list of combinations between all the worker
	"""
	worker=1
	for i in range(start,min(start+nb_worker,len(data))):
		comm.send(data[i], dest=worker, tag=11)
		worker+=1
	while worker<=nb_worker: # in case if the list is distributed and we have more workers
		comm.send("MAX", dest=worker, tag=11)
		worker+=1


def stop_workers(nb_worker):
	"""
	Input: the number of the workers
	Function send the 'stop' command to the workers
	"""
	for worker in range(1,nb_worker+1):
		comm.send("STOP", dest=worker, tag=11)

def  receive_two_clusters():
	"""
	Output: receive 2 clusters with their indexes from the root
	"""
	return comm.recv(source=0, tag=11)
start_time=time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nb_worker= comm.Get_size()-1

if rank==0: #root

	# Load the data
	data = np.load('data/features_50_1000.npy')

	# Intialize the clusters
	clusters = get_initial_clusters(data)
	K=len(clusters)
	print(K)
	# Start the algorithm
	while K>2: # Loop until number of clusters=2 and we can choose any number
		print('number of clusters: {}'.format(K))
		allComb=[]
		data=max_number
		# Generate all the possible combinations
		for i in range(len(clusters)):
			for j in range(i+1,len(clusters)):
				allComb.append(((clusters[i],clusters[j]),(i,j)))
		start=0
		# Intialize the min and arg_min
		global_min=max_number
		arg_min=(-1,-1)

		# Loop to send all the combinations
		while True:
			if start<len(allComb):
				send_two_clusters(allComb,start,nb_worker)
				start+=nb_worker
			# break on the end of the list
			else:
				break
			# Compute the min between all the computed distance on all the workers
			minn=comm.reduce((data,clusters),op=MPI.MINLOC,root=0)
			min_value=minn[0]
			min_loc=minn[1]
			# update the current min and arg_min if it's possible
			if(min_value<global_min):
				global_min=min_value
				arg_min=min_loc
		# update the clusters
		clusters=merging_min(clusters,arg_min)
		# update the K
		K=len(clusters)

	print('number of clusters: {}'.format(K))
	# Send the stop command after finishing
	stop_workers(nb_worker)
	# print(clusters)
if rank>0: # The workers
	while True: # Always running until receiving a stop command from the root
		# Receive the data
		data = receive_two_clusters()
		# Stop if command stop
		if data=="STOP":
			break
		# Return max if no more clusters
		elif data=="MAX":
			data=max_number
		# Otherwise compute the distance
		else:
			clusters=data[1]
			data=data[0]
			data=disimilarity_between_two_clusters(data[0],data[1])
		#Send the values to compute the min
		minn=comm.reduce((data,clusters),op=MPI.MINLOC,root=0)

	print("Time : {}".format(time.time()-start_time))
