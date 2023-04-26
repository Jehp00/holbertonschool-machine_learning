#!/usr/bin/env python3
"""module  optimization"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way

    :param X: numpy.ndarray of shape (m, nx) to shuffle
    :param Y: numpy.ndarray of shape (m, ny) to shuffle
    :return: shuffled X and Y matrices
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]

    return (X_shuffled, Y_shuffled)
