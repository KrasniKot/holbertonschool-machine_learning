#!/usr/bin/env python3
""" This module contains np_matmul() """

import numpy as np


def np_matmul(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """ Multiplies two np.ndarray matrices """
    return np.dot(mat1, mat2)
