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
        lo = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -(np.sum(lo) / lo.shape[1])
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        z = np.where(self.forward_prop(X) < 0.5, 0, 1)
        return z, self.cost(Y, self.forward_prop(X))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        dz = A - Y
        m = X.shape[1]
        dw = (1 / m) * np.dot(dz, X.T)
        db = (1 / m) * np.sum(dz)
        self.__W = self.__W - alpha * dw
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        :param X: contains the input data
        :param Y: contains the correct labels for the input data
        :param iterations: number of iterations to train over
        :param alpha:
        :return: the evaluation of the training data
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A, alpha)

        return self.evaluate(X, Y)
