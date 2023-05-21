#!/usr/bin/env python3
"""Module convolutional neurak netwwork"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape[0], kernel_shape[1]

    sh, sw = stride[0], stride[1]

    output_h = int((h_prev - kh) / sh + 1)
    output_w = int((w_prev - kw) / sw + 1)

    output = np.zeros((m, output_h, output_w, c_prev))

    examples = np.arange(m)

    for x in range(output_h):
        for y in range(output_w):
            if mode == 'max':
                res = np.max(A_prev[examples,
                             (x * sh):(x * sh) + kh,
                             (y * sw):(y * sw) + kw],
                             axis=(1, 2))
            elif mode == 'avg':
                res = np.mean(A_prev[examples,
                              (x * sh):(x * sh) + kh,
                              (y * sw):(y * sw) + kw],
                              axis=(1, 2))

            output[examples, x, y] = res

    return output
