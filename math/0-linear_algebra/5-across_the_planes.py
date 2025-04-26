#!/usr/bin/env python3
""" This module contains add_matrices2D() """

ms = __import__('2-size_me_please').matrix_shape  # Function to obtain the matrix shape
aa = __import__('4-line_up').add_arrays           # Function to add up two matrices (1d)


def add_matrices2D(mat1, mat2) :
    """ Adds two matrices
        - mat1 - list[list[Union[int, float]]] .... matrix a
        - mat2 - list[list[Union[int, float]]] .... matrix b

        >>> Performs element-wise addition on two 2d matrices - list[list[Union[int, float]]]
    """
    # Check both matrices have same dimensions and return the element-wise addition
    if ms(mat1) == ms(mat2): return [aa(mat1[index], mat2[index]) for index in range(len(mat1))]


if __name__ == '__main__':
    # Declare two matrices
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]

    # Print addition between mat1 and mat2
    print('Mat1 + Mat2:', add_matrices2D(mat1, mat2))  # mat1 + mat2 = [[6, 8], [10, 12]]

    # Print both matrices
    print('Mat1:', mat1)
    print('Mat2:', mat2)

    # This should print <<None>> since shapes do not match
    print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))  # (2, 2) + (2, 3) = None
