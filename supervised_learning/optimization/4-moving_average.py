#!/usr/bin/env python3
""" This module contains moving_average(),
    which returns a list of the moving averages of the given matrix.

    requires:
        - matplotlib.
        - numpy.
"""


def moving_average(data, beta):
    """ Calculates the moving averages for a given data.
        - data: list of data to calculate the moving average of.
        - beta: weight used for the moving average.
    """
    ws = 0
    mvav = []

    for i in range(len(data)):
        ws = ((ws * beta) + ((1 - beta) * data[i]))
        mvav.append(v / (1 - (beta ** (i + 1))))

    return mvav
