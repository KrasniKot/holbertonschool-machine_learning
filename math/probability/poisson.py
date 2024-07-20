#!/usr/bin/env python3
""" This module contains the class Poisson """


class Poisson():
    """ Defines a Poisson distribution """

    E = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Initializes a Poisson """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                # λ = Sum(number of events) / length of the interval
                self.lambtha = (sum(data) / len(data))

    def pmf(self, k):
        """ Calculates the PMF for a certain number of occurrences
            - k: number of ocurrences
        """
        k = int(k)
        if k < 0:
            return 0
        return (self.E ** -self.lambtha * self.lambtha ** k) / self.__f(k)

    def cdf(self, k):
        """
        Calculates the CDF for
        a certain number of occurrences
        """
        k = int(k)
        return sum([self.pmf(i) for i in range(k + 1)])

    @staticmethod
    def __f(k):
        """ Calculates k! """
        k = int(k)
        return 1 if k == 0 or k == 1 else k * Poisson.__f(k - 1)
