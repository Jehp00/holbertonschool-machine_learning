#!/usr/bin/env python3
"""this module includes the function pca"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset"""
    X_mean = X - np.mean(X, axis=0)
    u, Sigma, vh = np.linalg.svd(X_mean)
    w = vh.T
    wr = w[:, :ndim]

    return np.matmul(X_mean, wr)
