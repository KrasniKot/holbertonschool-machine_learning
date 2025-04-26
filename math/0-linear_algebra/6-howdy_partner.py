#!/usr/bin/env python3
""" This module contains def cat_arrays() """


def cat_arrays(arr1: list, arr2: list) -> list:
    """  Concatenates two arrays """
    return arr1 + arr2


if __name__ == '__main__':
    print('-' * 10, 'Two arrays', '-' * 10)
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [6, 7, 8]

    print('Array 1:', arr1)
    print('Array 2:', arr2)

    print('\nConcatenation...', cat_arrays(arr1, arr2))  # [1, 2, 3, 4, 5, 6, 7, 8]

    print('-' * 10, 'Two empty arrays', '-' * 10)
    arr1 = []
    arr2 = []

    print('Array 1:', arr1)
    print('Array 2:', arr2)

    print('\nConcatenation...', cat_arrays(arr1, arr2))  # []

    print('-' * 10, 'One empty array', '-' * 10)
    arr1 = [1, 2, 3, 4, 5]
    arr2 = []

    print('Array 1:', arr1)
    print('Array 2:', arr2)

    print('\nConcatenation...', cat_arrays(arr1, arr2))  # [1, 2, 3, 4, 5]
