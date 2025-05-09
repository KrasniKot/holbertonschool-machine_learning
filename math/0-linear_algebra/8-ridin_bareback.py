""" This module contains mat_mul() """


def mat_mul(mat1: list, mat2: list) -> list:
    """ Multiplies two matrices """

    if mat1 and mat2 and len(mat1[0]) == len(mat2):
        mult = []
        for ymat1 in range(len(mat1)):
            multrow = []
            for col in range(len(mat2[0])): multrow.append(sum(mat1[ymat1][row] * mat2[row][col] for row in range(len(mat1[0]))))
            mult.append(multrow)

        return mult


if __name__ == '__main__':

    # First example
    mat1 = [[1, 2], [3, 4], [5, 6]]
    mat2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
    print('First test:', mat_mul(mat1, mat2))  # Expected: [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]]

    # Square Matrix Test (3x3)
    mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mat2 = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    print('Square Matrix Test (3x3):', mat_mul(mat1, mat2))  # Expected: [[30, 24, 18], [84, 69, 54], [138, 114, 90]]

    # Non-Square Matrices Test (3x2 * 2x3)
    mat1 = [[1, 2], [3, 4], [5, 6]]
    mat2 = [[7, 8, 9], [10, 11, 12]]
    print('Non-Square Matrices Test (3x2 * 2x3):', mat_mul(mat1, mat2))  # Expected: [[27, 30, 33], [61, 68, 75], [95, 106, 117]]

    # Rectangular Matrices Test (2x3 * 3x1)
    mat1 = [[1, 2, 3], [4, 5, 6]]
    mat2 = [[7], [8], [9]]
    print('Rectangular Matrices Test (2x3 * 3x1):', mat_mul(mat1, mat2))  # Expected: [[50], [122]]

    # Zero Matrix Test
    mat1 = [[1, 2], [3, 4], [5, 6]]
    mat2 = [[0, 0], [0, 0]]
    print('Zero Matrix Test:', mat_mul(mat1, mat2))  # Expected: [[0, 0], [0, 0], [0, 0]]

    # Single Element Matrix Test
    mat1 = [[2]]
    mat2 = [[3]]
    print('Single Element Matrix Test:', mat_mul(mat1, mat2))  # Expected: [[6]]

    # Non-Commutative Property Test
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print('Non-Commutative Test A * B:', mat_mul(mat1, mat2))  # Expected: [[19, 22], [43, 50]]
    print('Non-Commutative Test B * A:', mat_mul(mat2, mat1))  # Expected: [[23, 34], [31, 46]]

    # Case 1: 2x3 * 2x2
    mat1 = [[1, 2, 3], [4, 5, 6]]
    mat2 = [[7, 8], [9, 10]]
    print('Incompatible Dimensions (2x3 * 2x2):', mat_mul(mat1, mat2))  # Expected: None

    # Case 2: 3x3 * 1x2
    mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mat2 = [[1, 2]]
    print('Incompatible Dimensions (3x3 * 1x2):', mat_mul(mat1, mat2))  # Expected: None

    # Case 3: Empty matrix
    mat1 = []
    mat2 = [[1, 2], [3, 4]]
    print('Empty Matrix Test:', mat_mul(mat1, mat2))  # Expected: None
