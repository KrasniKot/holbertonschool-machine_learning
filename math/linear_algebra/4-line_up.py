#!/usr/bin/env python3
""" This module contains add_arrays() """


def add_arrays(arr1, arr2):
    """ Adds two arrays if they are the same shape """
    if len(arr1) == len(arr2):
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
