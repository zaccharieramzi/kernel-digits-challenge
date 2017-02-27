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
        Y_new = np.argmin(np.linalg.norm(X[:, None, :] - centroids, axis=2),
                          axis=1)
        # Recomputation of centroids
        new_centroids = np.zeros(centroids.shape)
        for j in range(k):
            new_centroids[j, :] = np.mean(X[Y_new == j, :], axis=0)
        # Difference
        diff = np.sum(Y_new != Y)
        centroids = new_centroids
        Y = Y_new
        print("Wrongly clusterized pins: {}".format(diff))
    return centroids, Y


def vf_vector(pins, centroids, pin_to_im, n):
    '''Assigns each element of pins (points of interest for one image) to the
    closest centroid (VF).
        Args:
            - pins (list): the points of interest
            - centroids (ndarray): the visual features
            - pin_to_im (dict): the maps from a particular pin to the image
            it belongs to.
            - n (int): the number of images.
        Output:
            - ndarray: the visual features vector
    '''
    k = centroids.shape[0]
    vf_vector = np.zeros((n, k))
    pins = np.vstack(pins)

    y = np.argmin(np.linalg.norm(pins[:, None, :] - centroids, axis=2), axis=1)

    for idx, pin in enumerate(pins):
        y_pin = y[idx]
        vf_vector[pin_to_im[idx], y_pin] += 1
    return vf_vector


def gaussian_vf_vector(pins, pi, mu, sigma, pin_to_im, n):
    '''Assigns each element of pins (points of interest for one image) to the
    closest centroid (VF).
        Args:
            - pins (list): the points of interest
            - pi (ndarray): the frequencies of each gaussian
            - mu (ndarray): the mean of each gaussian
            - sigma (ndarray): the covariance matrix of each gaussian
            - pin_to_im (dict): the maps from a particular pin to the image
            it belongs to.
            - n (int): the number of images.
        Output:
            - ndarray: the visual features vector
    '''
    k = pi.shape[0]
    vf_vector = np.zeros((n, k))
    pins = np.vstack(pins)

    soft_assign = np.empty((pins.shape[0], k))
    for j in range(k):
        soft_assign[:, j] = pi[j] * npgauss(pins, mu[j], sigma[j])
    # [:, None] -> dividing each row by a different value
    soft_assign[:] = soft_assign[:] / np.sum(soft_assign, axis=1)[:, None]

    for idx, pin in enumerate(pins):
        vf_vector[pin_to_im[idx]] += soft_assign[idx]
    return vf_vector


def npgauss(X, mu, sig):
    '''
    compute normal law on a numpy vector
    args :
        - X ndarray (n_data, data_size) input data
        - mu ndarray (data_size) mean of the normal law
        - sig ndarray (data_size, data_size) covariance matrix of the normal
        law
    return :
        - ndarray (n_data)
    '''
    det = np.sqrt(np.abs(np.linalg.det(sig)))
    exposant = -1*((X - mu).dot(np.linalg.inv(sig))*(X - mu)).sum(axis=1)/2
    return np.exp(exposant)/(2*np.pi*det)


def em_full(data, k, max_iter=100):
    '''
    Computes gaussian mixture models through an EM algorithm
    args :
        - X ndarray (n_data, data_size) input data
        - k int, number of ellipsoids
    return :
        - ndarray (n_data)
    '''
    n_data, data_size = data.shape
    q = np.empty((data.shape[0], k))
    pi = np.empty(k)
    mu = np.empty((k, data_size))
    sigma = np.empty((k, data_size, data_size))

    # initialisation
    [centroids, labels] = kmeans(data, k)
    for c in range(len(centroids)):
        # mu is initialised with the centroid position
        mu[c, :] = data[labels == c].mean(axis=0)
        # pi with the proportion of elements in the cluster
        pi[c] = labels[labels == c].shape[0] / len(labels)
        # sigma with the sum of the squared distance to the centroid
        sigma[c] = np.cov(data[labels == c].T)
    sigma2 = sigma.copy()
    for i in range(max_iter):
        # Estimation
        for j in range(k):
            q[:, j] = pi[j] * npgauss(data, mu[j], sigma[j])
        # [:, None] -> dividing each row by a different v
        q[:] = q[:] / np.sum(q, axis=1)[:, None]
        # Maximisation
        for j in range(k):
            # update pi mu and sigma
            pi[j] = q[:, j].sum() / q.sum()
            mu[j] = (q[:, j][:, None] * data).sum(axis=0) / q[:, j].sum()

            sigma[j] = (q[:, j][:, None] * (data-mu[j])).T.dot(data-mu[j])
            sigma[j] /= q[:, j].sum()
        # we stop if the centroids converged
        if np.linalg.norm(mu - centroids) < 10e-3:
            break
        else:
            centroids = mu.copy()
    print("after {:d} iterations".format(i))
    # for each point the highest q value indicates cluster
    labels = np.argmax(q, axis=1)
    # plot(data, mu, sigma, labels)
    return pi, mu, sigma, q
