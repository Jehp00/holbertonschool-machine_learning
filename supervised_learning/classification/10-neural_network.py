#!/usr/bin/env python3
"""Module Neuronal Network"""

import numpy as np


class NeuralNetwork:
    """
        neural network with one hidden layer
        performing binary classification
    """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(0, 1, size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network"""
        z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2
