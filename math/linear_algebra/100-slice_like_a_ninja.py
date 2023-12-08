#!/usr/bin/env python3
""" This module contains np_slice() """


def np_slice(matrix, axes={}):
    """ Slices a matrix along specific axes """
    sm = [slice(None, None, None)] * matrix.ndim
    for k, v in sorted(axes.items()):
        sm[k] = slice(*v)

    return matrix[tuple(sm)]
