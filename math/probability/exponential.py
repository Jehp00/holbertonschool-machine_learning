#!/usr/bin/env python3
"""Module exponential probability"""


class Exponential:
    """
    Exponential Distribution
    """
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = 1 / float(sum(data) / len(data))

    def pdf(self, x):
        """
        :param x: time period
        :return: the value of the pdf for given x
        """
        e = 2.7182818285
        if x < 0:
            return 0
        return self.lambtha * e**(-self.lambtha * x)

    def cdf(self, x):
        """
        :param x: time period
        :return: the value of the cdf for given x
        """
        e = 2.7182818285
        if x < 0:
            return 0
        return 1 - e**(-self.lambtha * x)
