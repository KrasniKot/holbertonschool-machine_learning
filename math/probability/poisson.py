#!/usr/bin/env python3
""" This module contains the class Poisson """


class Poisson():
    """ Defines a Poisson distribution """

    def __init__(self, data=None, lambtha=1.):
        """ Initializes a Poisson """
        if not data and not lambtha or lambtha <= 0:
