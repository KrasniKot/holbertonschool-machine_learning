#!/usr/bin/env python3
""" This module contains np_cat() """

import numpy as np

def np_cat(mat1: np.ndarray, mat2: np.ndarray, axis: int = 0) -> np.ndarray:
    """ Concatenates two arrays along a specific axis """
    return np.concatenate((mat1, mat2), axis)
