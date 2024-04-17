#!/usr/bin/env python3
""" This module contains mat_mul() """


def mat_mul(mat1: list, mat2: list) -> list:
    """ Multiplies two matrices """
    if len(mat1[0]) == len(mat2):
        return [
            [sum(mat1[i][k] * mat2[k][j] for k in range(len(mat1[0])))
             for j in range(len(mat2[0]))]
            for i in range(len(mat1))
        ]
