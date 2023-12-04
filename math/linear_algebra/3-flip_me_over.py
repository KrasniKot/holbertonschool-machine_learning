#!/usr/bin/env python3
""" This module contains matrix_transpose() """


def matrix_transpose(matrix: list) -> list:
    """ Transposes a matrix """
    m = [row[:] for row in matrix]

    for i in range(len(m)):
        for j in range(i + 1, len(m[i])):
            m[i][j], m[j][i] = m[j][i], m[i][j]

    return m
