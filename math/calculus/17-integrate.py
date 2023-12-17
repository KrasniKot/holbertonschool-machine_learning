#!/usr/bin/env python3
""" This module contains poly_integral """

def poly_integral(poly):
    """ Calculates the integral of a list of coefficients """
    if type(poly) is not list or any(type(c) not in (int, float) for c in poly):
        return None

    integral = [0] + [int(c / (i + 1)) if c != 0 else 0 for i, c in enumerate(poly)]
    return integral
