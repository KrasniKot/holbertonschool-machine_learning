#!/usr/bin/env python3
""" This module contains summation_i_squared() """


def summation_i_squared(n):
    """ Calculates the sum of i² from 1 up to n """
    if type(n) == int and n > 0:
        return n * (n + 1) * (2 * n + 1) // 6
