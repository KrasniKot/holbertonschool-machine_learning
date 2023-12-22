#!/usr/bin/env python3
""" This module contains the class Exponential """


class Exponential():
    """ Defines an Exponential distribution """

    E = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Initializes an Exponential """
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
                self.lambtha = (1 / (sum(data) / len(data)))

    def pdf(self, k):
        """ Calculates the PMF for a given time period """
        k = int(k)
        if k < 0:
            return 0

        return (self.lambtha * self.E) ** (-self.lambtha * k)
