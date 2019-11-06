import numpy as np
import math

max_nb=10000000
def get_initial_clusters(data):
    clusters = []

    for i in range(data.shape[0]):
        cluster = [data[i]]
        clusters.append(cluster)

    return clusters

def disimilarity_between_two_elements(x,y):

    return np.sqrt(np.sum(np.square(x-y)))


def disimilarity_between_two_clusters(x,y):
    dis=0.0

    for i in x:
        for j in y:
        	dis=dis+disimilarity_between_two_elements(i,j)
    return dis/(len(x)*len(y))

def min_of_slice(start,end,clusters):
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
    points_j = clusters[int(min_pair[1])]
    clusters[int(min_pair[0])] = clusters[int(min_pair[0])] + points_j
    del clusters[int(min_pair[1])]

    return clusters