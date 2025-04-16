""" This module contains matrix_shape() """


def matrix_shape(matrix: list) -> list:
    """ Calculates recursively the shape of a matrix
        - matrix ..... matrix to calculate its shape

        >>> Given matrix shape
    """
    shape = [len(matrix)]

    if type(matrix[0]) is list: shape += matrix_shape(matrix[0])

    return shape


if __name__ == '__main__':
    # Get shape for first matrix (2, 2)
    mat1 = [[1, 2], [3, 4]]
    print('Shape matrix 1:', matrix_shape(mat1))

    # Get shape for second matrix (2, 3, 5)
    mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],[[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
    print('Shape matrix 2:', matrix_shape(mat2))
