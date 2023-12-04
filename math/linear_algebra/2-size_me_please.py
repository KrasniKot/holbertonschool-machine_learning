#!/usr/bin/env python3
""" This modulecontains matrix_shape() """


def matrix_shape(matrix: list) -> list:
    """ Calculates the shape of a matrix """
    shape = [len(matrix)]

    if type(matrix[0]) == list:
        shape += matrix_shape(matrix[0])

    return shape
