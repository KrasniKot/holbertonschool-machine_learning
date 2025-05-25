""" This module contains:
        * shape(): Gets the shape of a matrix
        * add_matrixes(): Performs the additions of two matrices
"""


def get_shape(matrix: list) -> list:
    """ Recursively calculates the shape of a matrix """
    matrix_shape = [len(matrix)]

    if isinstance(matrix[0], list): matrix_shape += get_shape(matrix[0])
    return matrix_shape


def add_matrices(mat1: list, mat2: list) -> list:
    """ Recursively adds two matrices """
    if get_shape(mat1) == get_shape(mat2):  # Verify shapes match

        # Return the addition
        if isinstance(mat1[0], list): return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]
        return [mat1[i] + mat2[i] for i in range(len(mat1))]


if __name__ == '__main__':
    # 1st dimensional matrices
    mat1 = [1, 2, 3]
    mat2 = [4, 5, 6]
    print(f'Addition equals:\n{add_matrices(mat1, mat2)}')

    # 2nd dimensional matrices
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print(f'\nAddition equals:\n{add_matrices(mat1, mat2)}')

    # 4th dimensional matrices
    mat1 = [[[[1, 2, 3, 4], [5, 6, 7, 8]],
             [[9, 10, 11, 12], [13, 14 ,15, 16]],
             [[17, 18, 19, 20], [21, 22, 23, 24]]],
            [[[25, 26, 27, 28], [29, 30, 31, 32]],
             [[33, 34, 35, 36], [37, 38, 39, 40]],
             [[41, 42, 43, 44], [45, 46, 47, 48]]]]
    mat2 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
             [[19, 110, 111, 112], [113, 114 ,115, 116]],
             [[117, 118, 119, 120], [121, 122, 123, 124]]],
            [[[125, 126, 127, 128], [129, 130, 131, 132]],
             [[133, 134, 135, 136], [137, 138, 139, 140]],
             [[141, 142, 143, 144], [145, 146, 147, 148]]]]
    mat3 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
             [[117, 118, 119, 120], [121, 122, 123, 124]]],
            [[[125, 126, 127, 128], [129, 130, 131, 132]],
             [[141, 142, 143, 144], [145, 146, 147, 148]]]]

    print(f'\nAddition equals:\n{add_matrices(mat1, mat2)}')
    print(f'\nAddition equals:\n{add_matrices(mat1, mat3)}')
