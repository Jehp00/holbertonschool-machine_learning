#!/usr/bin/env python3
"""This module contains the function convolve_grayscale_valid"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolution on grayscale images"""
    m = images.shape[0]
    img_h = images.shape[1]
    image_w = images.shape[2]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    output_h = img_h - kernel_h + 1
    output_w = image_w - kernel_w + 1

    output_image = np.zeros((m, output_h, output_w))

    image = np.arange(m)

    for x in range(output_h):
        for y in range(output_w):
            output_image[image, x, y] = np.sum(
                images[image, x:x + kernel_h, y:y + kernel_w] * kernel,
                axis=(1, 2))

    return output_image
