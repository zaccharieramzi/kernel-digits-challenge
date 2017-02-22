import math

import numpy as np


def kmeans(X, k):
    '''Performs the Kmeans algorithm on X with k clusters.
        Args:
            - X (ndarray): the points you want to cluster.
            - k (int): the number of cluster you want.
        Output:
            - ndarray: the centroids.
    '''
    n = X.shape[0]
    Y = np.zeros(n)
    centroids = []
    indices = np.random.permutation(n)
    j = 0
    while len(centroids) < k:
        idx = indices[j]
        candidate_centroid = X[idx]
        good_candidate = True
        for centroid in centroids:
            if np.linalg.norm(centroid - candidate_centroid) < 0.0001:
                good_candidate = False
        if good_candidate:
            centroids.append(candidate_centroid)
        j += 1
    centroids = np.vstack(centroids)
    diff = math.inf
    while diff > 0:
        # Assignment
        Y_new = np.zeros(n)
        for i in range(n):
            Y_new[i] = np.argmin(np.linalg.norm(centroids - X[i, :], axis=1))
        # Recomputation of centroids
        new_centroids = np.zeros(centroids.shape)
        for j in range(k):
            new_centroids[j, :] = np.mean(X[Y_new == j, :], axis=0)
        # Difference
        diff = np.sum(Y_new != Y)
        centroids = new_centroids
        Y = Y_new
        print(diff)
    return centroids


def vf_vector(pins, centroids):
    '''Assigns each element of pins (points of interest for one image) to the
    closest centroid (VF).
        Args:
            - pins (list): the points of interest
            - centroids (ndarray): the visual features
        Output:
            - ndarray: the visual features vector
    '''
    k = centroids.shape[0]
    vf_vector = np.zeros(k)
    for pin in pins:
        y = np.argmin(np.linalg.norm(centroids - pin, axis=1))
        vf_vector[y] += 1
    return vf_vector
