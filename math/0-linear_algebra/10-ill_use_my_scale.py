""" This module contains np_shape() """

import numpy as np


def np_shape(matrix) -> tuple:
    """ Returns the shape of a np matrix """
    return matrix.shape


if __name__ == '__main__':
    # Declare some matrices
    mat1 = np.array([1, 2, 3, 4, 5, 6])
    mat2 = np.array([])
    mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])

    # Print their shapes
    print('mat1 shape:', np_shape(mat1))  # Shape: (6,)
    print('mat2 shape:', np_shape(mat2))  # Shape: (0,)
    print('mat3 shape:', np_shape(mat3))  # Shape: (2, 2, 5)