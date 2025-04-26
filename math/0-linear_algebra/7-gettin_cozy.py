#!/usr/bin/env python3
""" This module contains cat_matrices2D() """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis
        > mat1: list[list[Union[int, float]]] .... First matrix to concatenate
        > mat2: list[list[Union[int, float]]] .... Second matrix to concatenate

        >>> A new matrix, being the result of the concatenation of mat1 and mat2 - list[list[Union[int, float]]]
    """
    assert axis == 0 or axis == 1, f'Axis must be either 0 or 1, not {axis}'

    conc_approach = {0: concatenate0, 1: concatenate1}  # Select the concatenation axis
    return conc_approach[axis](mat1, mat2)              # Perform the concatenation


def concatenate0(mat1, mat2):
    """ Concatenates two matrices along first axis, rows
        > mat1: list[list[Union[int, float]]] .... First matrix to concatenate
        > mat2: list[list[Union[int, float]]] .... Second matrix to concatenate

        >>> A new matrix, being the result of the concatenation of mat1 and mat2 - list[list[Union[int, float]]]
    """
    if len(mat1[0]) == len(mat2[0]): return mat1 + mat2


def concatenate1(mat1, mat2):
    """ Concatenates two matrices along second axis, columns
        > mat1: list[list[Union[int, float]]] .... First matrix to concatenate
        > mat2: list[list[Union[int, float]]] .... Second matrix to concatenate

        >>> A new matrix, being the result of the concatenation of mat1 and mat2 - list[list[Union[int, float]]]
    """
    if len(mat1) == len(mat2): return [row1 + row2 for row1, row2 in zip(mat1, mat2)]


if __name__ == "__main__":
    # Define some matrices
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6]]
    mat3 = [[7], [8]]

    # Perform the concatenation in both axis
    print('Matrix1 + Matrix2, along first axis:', cat_matrices2D(mat1, mat2))           # [[1, 2], [3, 4], [5, 6]]
    print('Matrix1 + Matrix3, along second axis:', cat_matrices2D(mat1, mat3, axis=1))  # [[1, 2, 7], [3, 4, 8]]

