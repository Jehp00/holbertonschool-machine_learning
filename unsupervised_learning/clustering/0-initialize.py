#!/usr/bin/env python3
"""This module contains the function initialize"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means"""
    if not isinstance(k, int) or k <= 0:
        return None

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    N, D = X.shape

    max = np.max(X, axis=0)
    min = np.min(X, axis=0)

    res = np.random.uniform(min, max, (k, D))
    return res
