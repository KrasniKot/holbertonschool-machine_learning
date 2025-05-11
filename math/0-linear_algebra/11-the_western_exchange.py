""" This module contains np_transpose() """

import numpy as np


def np_transpose(matrix):
    """ Trasnposes a matrix """
    return matrix.T


if __name__ == '__main__':

    # Declare some matrices
    mat1 = np.array([1, 2, 3, 4, 5, 6])
    mat2 = np.array([])
    mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])

    # Print their transposes
    print('-'*40)
    print(f'Matrix 1:\n{mat1}')
    print(f'Transpose of Matrix 1:\n{np_transpose(mat1)}')
    print('-'*40)
    print(f'Matrix 2:\n{mat2}')
    print(f'Transpose of Matrix 2:\n{np_transpose(mat2)}')
    print('-'*40)
    print(f'Matrix 3:\n{mat3}')
    print(f'Transpose of Matrix 3:\n{np_transpose(mat3)}')
