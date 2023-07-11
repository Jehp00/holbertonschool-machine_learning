#!/usr/bin/env python3
"""this module includes the function pca"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""
    u, Sigma, vh = np.linalg.svd(X, full_matrices=False)
    comulative_var = np.cumsum(Sigma) / np.sum(Sigma)

    r = (np.argwhere(comulative_var >= var))[0, 0]
    w = vh.T
    wr = w[:, :r + 1]

    return wr
