#!/usr/bin/env python3
"""This module contains the function convolve_grayscale"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a convolution on grayscale images with custom padding"""
    m = images.shape[0]
    img_h = images.shape[1]
    img_w = images.shape[2]

    kernel_h = kernels.shape[0]
    kernel_w = kernels.shape[1]
    kernel_n = kernels.shape[3]

    pad_h = 0
    pad_w = 0

    sh = stride[0]
    sw = stride[1]

    if isinstance(padding, tuple):
        pad_h = padding[0]
        pad_w = padding[1]

    if padding == 'same':
        pad_h = int((((img_h - 1) * sh + kernel_h - img_h) / 2) + 1)
        pad_w = int((((img_w - 1) * sw + kernel_w - img_w) / 2) + 1)

    output_h = int((img_h + 2 * pad_h - kernel_h) / sh + 1)
    output_w = int((img_w + 2 * pad_w - kernel_w) / sw + 1)

    output_image = np.zeros((m, output_h, output_w, kernel_n))

    image = np.arange(m)

    images_pad = np.pad(images, pad_width=((0, 0), (pad_h, pad_h),
                                           (pad_w, pad_w), (0, 0)),
                        mode='constant')

    for x in range(output_h):
        for y in range(output_w):
            for k in range(kernel_n):
                output_image[image, x, y, k] = np.sum(
                    images_pad[image, (x * sh):(x * sh) + kernel_h,
                                      (y * sw):(y * sw) + kernel_w] *
                    kernels[:, :, :, k],
                    axis=(1, 2, 3))

    return output_image
