#!/usr/bin/env python3
"""Module convolutional neurak netwwork"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network
    """
    m, new_h, w_new, c_new = dZ.shape
    m, prev_h, prev_w, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = int(np.ceil(((prev_h - 1) * sh + kh - prev_h) / 2))
        pad_w = int(np.ceil(((prev_w - 1) * sw + kw - prev_w) / 2))

    else:
        pad_h, pad_w = 0, 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h),
                                 (pad_w, pad_w), (0, 0)), mode='constant')

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(m):
        for h in range(new_h):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = A_prev_pad[i,
                                         vert_start:vert_end,
                                         horiz_start:horiz_end,
                                         :]
                    dA_prev_pad[i,
                                vert_start:vert_end,
                                horiz_start:horiz_end,
                                :] += W[:,
                                        :,
                                        :,
                                        c] * dZ[i,
                                                h,
                                                w,
                                                c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
    dA_prev = dA_prev_pad[:, pad_h:prev_h + pad_h, pad_w:prev_w + pad_w + pad_w, :]
    return dA_prev, dW, db
