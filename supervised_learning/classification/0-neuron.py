#!/usr/bin/env python3
"""module first neuron"""
import numpy.random


class Neuron:
    """

    """

    def __init__(self, nx):
        """
        :param nx:
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = numpy.random.normal(0, 1, size=(1, nx))
        self.b = 0
        self.A = 0
