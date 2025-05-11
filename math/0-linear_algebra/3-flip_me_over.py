""" This module contains matrix_transpose() """


def matrix_transpose(matrix: list[int]) -> list:
    """ Transposes the given matrix
        - matrix .... given matrix to be transposed

        >>> Transposed matrix
    """
    return list(zip(*matrix))


if __name__ == '__main__':
    # Assigning, printing and transposing first matrix
    mat1 = [[1, 2], [3, 4]]
    print('Matrix 1:', mat1)
    print(matrix_transpose(mat1))
    print('-' * 50)

    # Assigning, printing and transposing second matrix
    mat2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
    print('Matrix 2:', mat2)
    print(matrix_transpose(mat2))
