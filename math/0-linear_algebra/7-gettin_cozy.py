#!/usr/bin/env python3
""" This module contains cat_matrices2D() """


def cat_matrices2D(mat1: list, mat2: list, axis: int = 0) -> list:
    """ Concatenates two matrices along a specific axis """
    if axis == 1 and len(mat1) == len(mat2):
        return [mat1[x] + mat2[x] for x in range(len(mat1))]
    elif axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
