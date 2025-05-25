""" This module contains np_slice() """

import numpy as np
from typing import Any, Dict, Tuple

def np_slice(matrix: np.ndarray, axes: Dict[int, Tuple[int | None, int | None]]) -> np.ndarray:
    """ Slices a matrix along specific axes
        > matrix .... matrix to be sliced
        > axes ...... dictionary where the key represents the axis to slice along and
                      the value is a tuple representing the slice to make along that axis.
    """
    ######## Initialise a future list of slices
    slices = [slice(None)] * matrix.ndim
    # slice(start, end, step) * number of axes the matrix has
    # In the case of: mat[slice(None), slice(None)]
    # First slice indicates the slice for rows, second for columns and so on...
    ########

    # Re-assign for each axis in <<axes>> the corresponding slice value
    for axis, axslice in axes.items(): slices[axis] = slice(*axslice)  # Unpack the slice

    return matrix[tuple(slices)]  # For slicing operations a tuple is expected, not a list


if __name__ == '__main__':

    # First example
    mat1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    print(f'First matrix:\n{mat1}')
    print(f'Matrix slice:\n{np_slice(mat1, axes={1: (1, 3)})}\n')
    print('-' * 30)

    # Second example
    mat2 = np.array([[[1, 2, 3, 4, 5],[6, 7, 8, 9, 10]],[[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]], [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
    print(f'Second matrix:\n{mat2}')
    print(f'Matrix slice:\n{np_slice(mat2, axes={0: (2,), 2: (None, None, -2)})}')
