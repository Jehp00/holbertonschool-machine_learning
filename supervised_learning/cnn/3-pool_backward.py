#!/usr/bin/env python3
"""Module convolutional neurak netwwork"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network
    """
    res = []
    m, new_h, new_w, new_c = dA.shape
    m, prev_h, prev_w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros((m, prev_h, prev_w, c))

    for x in range(new_h):
        for y in range(new_w):
            for k in range(c):
                for case in range(m):
                    i = x * sh
                    j = y * sw
                    if mode == 'max':
                        a_prev_slice = A_prev[case, i: i + kh, j: j + kw, k]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        res = mask * dA[case, x, y, k]
                    elif mode == 'avg':
                        res = dA[case, x, y, k] / (kh * kw)

                    dA_prev[case, i: i + kh, j: j + kw, k] += res
    return dA_prev
