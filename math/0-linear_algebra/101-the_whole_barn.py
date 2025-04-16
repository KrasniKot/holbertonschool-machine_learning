#!/usr/bin/env python3
""" This module contains:
        * shape(): Gets the shape of a matrix
        * add_matrixes(): Performs the additions of two matrices
"""


def shape(matrix: list) -> list:
    """ Calculates the shape of a matrix """
    s = [len(matrix)]

    if type(matrix[0]) == list:
        s += shape(matrix[0])

    return s


def add_matrices(mat1: list, mat2: list) -> list:
    """ Adds two matrices """
    if shape(mat1) == shape(mat2):
        if type(mat1[0]) == list:
            return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
