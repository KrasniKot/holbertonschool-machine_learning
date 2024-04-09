#!/usr/bin/env python3
""" This module contains add_matrices2D() """


def add_matrices2D(mat1: list, mat2: list) -> list:
    """ Adds two matrices
        - mat1: matrix a
        - mat2: matrix b
    """
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        return [[mat1[i][j] + mat2[i][j] for i in range(len(mat1))]
                for j in range(len(mat1[0]))]
