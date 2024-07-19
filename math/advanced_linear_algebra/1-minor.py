#!/usr/bin/env python3
""" This module contains the function minor(),
    that calculates the minor matrix of a matrix
"""


def minor(matrix):
    """ Calculates the determinant of a minor matrix
        and builds a matrix of determinants
        - matrix: matrix to calculate its determinant
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
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    minor_matrix = []

    for i in range(len(matrix)):
        minor_matrix.append([])

        for j in range(len(matrix[0])):
            # Get minor for element ij
            minor = get_minor_slice(matrix, i, j)

            # Calculate determinant and append it to minor_matrix
            det = determinant(minor)
            minor_matrix[i].append(det)

    return minor_matrix


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

mat = [[98, 89, 0, 17, 3],
      [13, 14, 30, 43, 13],
      [13, -13, -14, -15, 56],
      [9, 5, 8, 6, 57],
      [92, 34, 3, -3, -89]]
