#!/usr/bin/env python3
""" This module contains matrix_transpose() """


def matrix_transpose(matrix):
    """ Transposes a matrix """
    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]
