""" This module contains matrix_transpose() """


def matrix_transpose(matrix: list) -> list:
    """ Transposes the given matrix """
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]
