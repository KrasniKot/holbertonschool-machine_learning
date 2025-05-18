""" This module contains np_matmul() """

import numpy as np


def np_matmul(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """ Multiplies two np.ndarray matrices """
    return mat1 @ mat2


if __name__ == '__main__':
    # Declare some numpy arrays
    mat1 = np.array([[11, 22, 33], [44, 55, 66]])
    mat2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mat3 = np.array([[7], [8], [9]])

    # Perform matrix multiplication
    print('matrix 1 @ matrix 2:\n', np_matmul(mat1, mat2))
    print('\nmatrix 1 @ matrix 3:\n', np_matmul(mat1, mat3))
