#!/usr/bin/env python3
"""This module contains the function mean_cov"""
import numpy as np


def mean_cov(X):
    """Calculates the mean and coveriance of a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    n = X.shape[0]

    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.sum(X, axis=0) / n

    Z = (X - mean)
    cov = np.matmul(Z.T, Z) / (n - 1)

    return mean.reshape(1, -1), cov
