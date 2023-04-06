#!/usr/bin/env python3
"""Module poisson class"""


class Possion:
    """
    Poisson Distribution
    """
    def __int__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

    def pmf(self, k):
        """PMF of a given number of success"""
        e = 2.7182818285
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0

        fact = 1
        for i in range(1, k+1):
            fact = fact * i

        return ((e**(-self.lambtha))*(self.lambtha**k))/fact

    def cdf(self, k):
        """CDF for a given number of success"""
        e = 2.7182818285
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0

        res = 0
        for i in range(k+1):
            fact = 1
            for i in range(1, i+1):
                fact = fact * i
            res += ((e**(-self.lambtha))*(self.lambtha**i)/fact
        return res
