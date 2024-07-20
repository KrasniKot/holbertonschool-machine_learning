""" This module contains the function definiteness()
    that evaluates the definiteness of a matrix
"""

import numpy as np


def definiteness(matrix):
    """ Evaluates the definiteness of a given matrix
        - matrix: matrix whose definiteness should be evaluated
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    mat = matrix

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.size == 0:
        return

    if matrix.shape[0] != matrix.shape[1]:
        return

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return
