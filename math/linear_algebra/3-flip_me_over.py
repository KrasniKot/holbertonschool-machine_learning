#!/usr/bin/env python3
""" This module contains matrix_transpose() """


def matrix_transpose(matrix: list) -> list:
    """ Transposes a matrix """
    return [[matrix[i][j] for i in range(len(matrix))]
            for j in range(len(matrix[0]))]
