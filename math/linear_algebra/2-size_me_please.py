""" This module contains matrix_shape() """


def matrix_shape(matrix: list) -> list:
    """ Calculates recursively the shape of a matrix
        - matrix ..... matrix to calculate its shape
    """
    shape = [len(matrix)]

    if type(matrix[0]) is list: shape += matrix_shape(matrix[0])

    return shape
