#!/usr/bin/env python3
"""K means"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    K-means on a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    N, D = X.shape

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    C = np.random.uniform(X_min, X_max, (k, D))

    for i in range(iterations):
        cdois = np.copy(C)
        cdois_entended = C[:, np.newaxis]

        distance = np.sqrt(((X - cdois_entended) ** 2).sum(axis=2))

        clss = np.argmin(distance, axis=0)

        for cl in range(k):
            if X[clss == cl].size == 0:
                C[cl] = np.random.uniform(X_min, X_max, size=(1, D))
            else:
                C[cl] = X[clss == cl].mean(axis=0)

        cdois_entended = C[:, np.newaxis]
        distance = np.sqrt(((X - cdois_entended) ** 2).sum(axis=2))
        clss =np.argmin(distance, axis=0)

        if (cdois == C).all():
            break

        return C, clss
