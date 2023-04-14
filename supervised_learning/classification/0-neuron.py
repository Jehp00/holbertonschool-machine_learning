#!/usr/bin/env python3
"""module first neuron"""
import numpy as np


class Neuron:
    """
    Class Neuron
    """

    def __init__(self, nx):
        """
        :param nx: Number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(0, 1, size=(1, nx))
        self.b = 0
        self.A = 0
