#!/usr/bin/env python3
""" this module contains np_elementwise() """

import numpy as np
from numpy.typing import NDArray


def np_elementwise(mat1: NDArray, mat2: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """ Performs addition, substraction, multiplication and division of mat1 and mat2 """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2


if __name__ == '__main__':
    # Define two arrays
    mat1 = np.array([[11, 22, 33], [44, 55, 66]])
    mat2 = np.array([[1, 2, 3], [4, 5, 6]])

    print('Matrix 1:\n', mat1)
    print('Matrix 2:\n', mat2)

    # Perform the element-wise operations
    print('-' * 40, 'matrix to matrix operations', '-' * 40)
    add, sub, mul, div = np_elementwise(mat1, mat2)
    print("Add (Element-wise Addition):\n", add)
    print("Sub (Element-wise Subtraction):\n", sub)
    print("Mul (Hadamard Product):\n", mul)
    print("Div (Element-wise Division):\n", div)

    # Perform element-wise operations with a scalar
    print('-' * 40, 'matrix to scalar (2) operations', '-' * 40)
    add, sub, mul, div = np_elementwise(mat1, 2)
    print("Add (Element-wise Addition with scalar):\n", add)
    print("Sub (Element-wise Subtraction with scalar):\n", sub)
    print("Mul (Hadamard Product with scalar):\n", mul)
    print("Div (Element-wise Division with scalar):\n", div)
