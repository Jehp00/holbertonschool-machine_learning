#!/usr/bin/env python3
"""Module convolutional neurak netwwork"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional layer of a neural
       network."""
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]
    c_new = W.shape[2]

    pad_h = 0
    pad_w = 0

    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        pad_h = ((((h_prev - 1) * sh) + kh - h_prev) // 2)
        pad_w = ((((w_prev - 1) * sw) + kw - w_prev) // 2)

    opt_h = int((h_prev + (2 * pad_h) - kh) / sh + 1)
    opt_w = int((w_prev + (2 * pad_w) - kw) / sw + 1)

    opt = np.zeros((m, opt_h, opt_w, c_new))

    cases = np.arange(m)

    opt_pad = np.pad(A_prev, pad_width=((0, 0), (pad_h, pad_h),
                                        (pad_w, pad_w), (0, 0)),
                     mode='constant')

    for x in range(opt_h):
        for y in range(opt_w):
            for k in range(c_new):
                opt[cases,
                    x,
                    y, k] = np.sum(opt_pad[cases,
                                           (x * sh):(x * sh) + kh,
                                           (y * sw):(y * sw) + kw]
                                   * W[:, :, :, k],
                                   axis=(1, 2, 3))
    active = activation(opt + b)

    return active
