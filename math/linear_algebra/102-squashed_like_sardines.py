#!/usr/bin/env python3
""" This module concatenates two matrices and contains:
        * shape(): Gets the shape of a matrix
        * cat_matrices(): Concatenates two matrices
"""


def shape(matrix: list) -> list:
    """ Calculates the shape of a matrix """
    s = [len(matrix)]

    if type(matrix[0]) == list:
        s += shape(matrix[0])

    return s


def cat_matrices(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis """
    if shape(mat1) == shape(mat2):
        if axis == 0:
            return mat1 + mat2
        elif axis == 1:
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
        else:
            return [cat_matrices(r, r2, axis=axis-1)
                    for r, r2 in zip(mat1, mat2)]
