#!/usr/bin/env python3
""" This module contains poly_integral """

def poly_integral(poly):
    """ Calculates the integral of a list of coefficients """
    if type(poly) is not list or any(type(coeff) not in (int, float) for coeff in poly):
        return None

    return [0] + [int(coeff / (i + 1)) for i, coeff in enumerate(poly)]
