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
        self.__W = np.random.normal(0, 1, size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        :param X: numpy.ndarray with shape (nx, m) that contains
        the input data
        :return: the forward propagation of the neuron
        """
        z = np.dot(self.W, X) + self.b
        self.__A = 1/(1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        :param Y: labels for the input data
        :param A: output of the neuron for each example
        :return: the cost
        """
        lo = Y * np.log(A) + (1 - Y) * np.log(1.0000001- A)
        cost = -(np.sum(lo) / lo.shape[1])
        return cost
