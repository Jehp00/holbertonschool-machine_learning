#!/usr/bin/env python3
"""conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X

    for i in range(L):
        W = weights['W' + str(i + 1)]
        A = cache['A' + str(i)]
        B = weights['b' + str(i + 1)]

        Z = np.dot(W, A) + B

        D = np.where(np.random.rand(Z.shape[0], Z.shape[1]) < keep_prob, 1, 0)

        if i == L - 1:
            soft_max = np.exp(Z)
            cache['A' + str(i + 1)] = (soft_max / np.sum(soft_max, axis=0,
                                                         keepdims=True))
        else:
            tanh = np.tanh(Z)
            cache['A' + str(i + 1)] = (tanh / keep_prob) * D
            cache['D' + str(i + 1)] = D

    return cache
