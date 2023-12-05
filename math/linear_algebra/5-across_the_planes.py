#!/usr/bin/env python3
""" This module contains add_matrices2D() """


def add_matrices2D(mat1: list, mat2: list) -> list:
    """ Adds two matrices """
    if (len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0])):
        return [[mat1[r][e] + mat2[r][e] for e in range(len(mat1[0]))]
                for r in range(len(mat1))]
