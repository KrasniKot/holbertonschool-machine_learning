#!/usr/bin/env python3
""" This module contains the class Normal """


class Normal():
    """ Defines a Normal distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ Initializes a Normal """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum((x - self.mean) ** 2
                               for x in data) / len(data)) ** 0.5
