#!/usr/bin/env python3
""" This module contains poly_integral """

def poly_integral(poly):
    """ Calculates the integral of a list of coefficients """
    if type(poly) is not list or any(type(c) not in (int, float) for c in poly):
        return None

    integral = [C] + [
        int(coeff / (i + 1)) if coeff != 0 else 0
        for i, coeff in enumerate(poly)
    ]

    return integral
