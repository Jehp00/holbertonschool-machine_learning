#!/usr/bin/env python3
"""Module optimization"""


def normalization_constants(X):
    """
    Calculates the normalization constants of np.ndarray matrix

    :param X: X [numpy.ndarray of shape (m, nx)]:
            matrix to find normalization constants for
            m: number of data points
            nx: the number of features
    :return: the mean and standard deviation of each feature, respectively
    """
    me = X.mean(axis=0)
    sigma = X.std(axis=0)
    return (me, sigma)
