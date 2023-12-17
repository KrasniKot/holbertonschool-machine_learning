#!/usr/bin/env python3
""" This module contains poly_integral """


def poly_integral(poly, C=0):
    """ Calculates the integral of a list of coefficients """
    if type(poly) is list and all(type(c) in (int, float) for c in poly):
        if len(poly) == 1 and poly[0] == 0:
            return [0]

        integral = [C] + [
            c // (i + 1) if c % (i + 1) == 0 else c / (i + 1)
            for i, c in enumerate(poly)
        ]
        return integral
