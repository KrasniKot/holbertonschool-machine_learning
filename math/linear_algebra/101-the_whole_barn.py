#!/usr/bin/env python3
""" This module contains add_matrices() """

import numpy as np


def add_matrices(mat1: list, mat2: list) -> list:
    """ Adds two matrices """
    if np.array(mat1).shape == np.array(mat2).shape:
        return np.array(mat1) + np.array(mat2)
