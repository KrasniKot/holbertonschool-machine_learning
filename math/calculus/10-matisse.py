#!/usr/bin/env python3
""" This module contains poly_derivate() """


def poly_derivative(poly):
    """ Derivates a polynomial """
    if poly and type(poly) is list:
        if len(poly) == 1:
            return [0]
        derivative = [i * poly[i] for i in range(1, len(poly))]
        return [0] if not derivative else derivative
