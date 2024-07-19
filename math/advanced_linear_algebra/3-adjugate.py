#!/usr/bin/env python3
""" This module contains the function ajugate(),
    that calculates the adjugate matrix of a matrix
"""


def adjugate(matrix):
    """ Calculates the adjugate of a matrix
        - matrix: matrix to calculate its adjugate
    """
    cmatrix = cofactor(matrix)

    for i in range(len(cmatrix)):
        for j in range(i + 1, len(cmatrix[0])):
            cmatrix[i][j], cmatrix[j][i] = cmatrix[j][i], cmatrix[i][j]

    return cmatrix


def cofactor(matrix):
    """ Calculates the cofactor of a matrix
        - matrix: matrix to calculate its cofactor
    """
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if not is_square(matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    # Special case for [[]] matrix
    if len(matrix) == 1 and len(matrix[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')

    # Special case for [[x]] matrix
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    # Special case for 2*2 matrix
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return [[matrix[1][1], -matrix[1][0]], [-matrix[0][1], matrix[0][0]]]

    cofactor_matrix = []

    for i in range(len(matrix)):
        cofactor_matrix.append([])

        for j in range(len(matrix[0])):
            minor = get_minor_slice(matrix, i, j)

            det = determinant(minor)
            cofactor_matrix[i].append((-1)**(i + j) * det)

    return cofactor_matrix


def is_square(matrix):
    """ Checks if a matrix is square
        - matrix: matrix to get checked
    """
    # Get the number of rows
    num_rows = len(matrix)

    # Check if all rows have the same number of columns
    for row in matrix:
        if len(row) != num_rows and len(row) != 0 and num_rows != 1:
            return False

    return True


def determinant(matrix):
    """ Recursively calculates the determinant of a matrix """
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(len(matrix)):
        minor = get_minor_slice(matrix, 0, c)
        det += ((-1) ** c) * matrix[0][c] * determinant(minor)

    return det


def get_minor_slice(matrix, i, j):
    """ Returns a minor for the given position
        - i: row
        - j: column
    """
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
