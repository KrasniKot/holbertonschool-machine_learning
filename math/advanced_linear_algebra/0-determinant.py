#!/usr/bin/env python3
""" This module contains determinant(),
    that calculates the determinant of a matrix
"""


def determinant(matrix):
    """ Calculates the determinant of a matrix
        - matrix: matrix to calculate its determinant
    """
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if not is_square(matrix):
        raise ValueError('matrix must be a square matrix')

    # Special case for [[]] matrix
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    # Special case for [[x]] matrix
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(len(matrix)):
        det += ((-1) ** c) * matrix[0][c] * determinant(minor(matrix, 0, c))
    return det


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


def minor(matrix, i, j):
    """ Returns a minor for the given position
        - i: row
        - j: column
    """
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
