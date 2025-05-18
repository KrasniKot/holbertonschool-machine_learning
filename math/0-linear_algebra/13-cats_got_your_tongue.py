""" This module contains np_cat() """

import numpy as np


def np_cat(mat1: np.ndarray, mat2: np.ndarray, axis: int = 0) -> np.ndarray:
    """ Concatenates two arrays along a specific axis """
    return np.concatenate((mat1, mat2), axis)


if __name__ == '__main__':
    # Define some numpy arrays
    mat1 = np.array([[11, 22, 33], [44, 55, 66]])
    mat2 = np.array([[1, 2, 3], [4, 5, 6]])
    mat3 = np.array([[7], [8]])

    # Perform some concatenations
    print('Concatenation mat1 and mat2 (axis 0):\n', np_cat(mat1, mat2))          # [[11, 22, 33], [44, 55, 66], [1, 2, 3], [4, 5, 6]]
    print('\nConcatenation mat1 and mat2 (axis 1):\n', np_cat(mat1, mat2, axis=1))  # [[11, 22, 33, 1, 2, 3], [44, 55, 66, 4, 5, 6]]
    print('\nConcatenation mat1 and mat1 (axis 1):\n', np_cat(mat1, mat3, axis=1))  # [[11, 22, 33, 7], [44, 55, 66, 8]]