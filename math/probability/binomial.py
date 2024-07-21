#!/usr/bin/env python3
""" This module contains Binomial """


class Binomial():
    """ Defines a Binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ Initializes a Binomial """
        self.n = int(n)
        self.p = float(p)

        if data is None:
            if self.n < 1:
                raise ValueError("n must be a positive value")
            elif self.p <= 0 or self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1 - variance / mean
            n = int(round(mean / p))
            p = float(mean / n)

        self.n = n
        self.p = p

    def pmf(self, k):
        """ Calculates the PMF for a given k """
        k = int(k)
        if k < 0:
            return 0
        return (self.__f(self.n) / (self.__f(k) * self.__f(self.n - k))) * \
               (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """ Calculates the CDF for a given k """
        k = int(k)
        if k < 0:
            return 0

        cp = 0
        for i in range(k + 1):
            cp += self.pmf(i)

        return cp

    @staticmethod
    def __f(k):
        """ Calculates k! """
        k = int(k)
        return 1 if k == 0 or k == 1 else k * Binomial.__f(k - 1)
