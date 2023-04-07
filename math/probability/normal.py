#!/usr/bin/env python3
""" Module normal class"""


class Normal:
    """Normal distribution"""
    def __init__(self, data=None, mean=0, stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = (sum((x - (sum(data) / len(data))) **
                           2 for x in data) / len(data))**0.5

    def z_score(self, x):
        """
        :param x: value
        :return: z_score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        :param z: value
        :return: x_value of z
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        :param x: value
        :return: pdf of the x
        """
        pi = 3.1415926536
        e = 2.7182818285
        constant = 1 / (self.stddev * (2 * pi) ** 0.5)
        return constant * (e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2))

    def cdf(self, x):
        """
        :param x: value
        :return: cdf of the x
        """
        pi = 3.1415926536
        norm_x = (x - self.mean) / (self.stddev * (2 ** 0.5))
        term_1 = ((norm_x ** 3) / 3)
        term_2 = ((norm_x ** 5) / 10)
        term_3 = ((norm_x ** 7) / 42)
        term_4 = ((norm_x ** 9) / 216)
        erf = (2 / (pi ** 0.5)) * (norm_x - term_1 + term_2 - term_3 + term_4)
        return 0.5 * (1 + erf)
