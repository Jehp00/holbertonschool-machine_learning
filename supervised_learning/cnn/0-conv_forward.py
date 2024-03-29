#!/usr/bin/env python3
"""Module convolutional neurak netwwork"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional layer of a neural
       network."""
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]

    kh = W.shape[0]
    kw = W.shape[1]
    c_new = W.shape[3]

    pad_h = 0
    pad_w = 0

    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        pad_h = ((((h_prev - 1) * sh) + kh - h_prev) // 2)
        pad_w = ((((w_prev - 1) * sw) + kw - w_prev) // 2)

    output_h = int((h_prev + (2 * pad_h) - kh) / sh + 1)
    output_w = int((w_prev + (2 * pad_w) - kw) / sw + 1)

    output = np.zeros((m, output_h, output_w, c_new))

    examples = np.arange(m)

    output_pad = np.pad(A_prev, pad_width=((0, 0), (pad_h, pad_h),
                                           (pad_w, pad_w), (0, 0)),
                        mode='constant')

    for x in range(output_h):
        for y in range(output_w):
            for k in range(c_new):
                output[examples,
                       x,
                       y, k] = np.sum(output_pad[examples,
                                      (x * sh):(x * sh) + kh,
                               (y * sw):(y * sw) + kw]
                               * W[:, :, :, k],
                               axis=(1, 2, 3))

    activation = activation(output + b)

    return activation
