#!/usr/bin/env python3
""" Numpy to Pandas DataFrame """

import pandas as pd


def from_numpy(array):
    """ Create a pd.DataFrame from a np.ndarray """
    return pd.DataFrame(array, columns=[chr(i + 65) for i in range(len(array[0]))])
