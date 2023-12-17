#!/usr/bin/env python3
""" This module contains poly_integral """


def poly_integral(poly, C=0):
    """ Calculates the integral of a list of coefficients """
    if type(poly) is list or any(type(c) in (int, float) for c in poly):
        
        integral = [C] + [
            c / (i + 1) if c != 0 else 0
            for i, c in enumerate(poly)
            ]
            
        return integral
