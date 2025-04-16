#!/usr/bin/env python3
""" This module contains add_arrays() """

ms = __import__('2-size_me_please').matrix_shape


def add_arrays(arr1: list, arr2: list) -> list:
    """ Adds two arrays if they are the same shape
        - arr1 .... array a
        - arr2 .... array b

        >>> Element wise addition of both matrices
    """
    if ms(arr1) == ms(arr2): return [arr1[index] + arr2[index] for index in range(len(arr1))]


if __name__ == '__main__':
    add_arrays = __import__('4-line_up').add_arrays  # Import the function

    # Declare two arrays
    arr1, arr2 = [1, 2, 3, 4],  [5, 6, 7, 8]

    # Print the element wise addition
    print('Array addition (arr1, arr2):', add_arrays(arr1, arr2))

    # Print both arrays
    print('Array 1:', arr1)
    print('Array 2:', arr2)

    # This should print <<None>> because of the different shapes: (4, ) vs (3, )
    print(add_arrays(arr1, [1, 2, 3]))  
