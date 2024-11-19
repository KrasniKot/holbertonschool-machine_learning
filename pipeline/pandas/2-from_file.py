#!/usr/bin/env python3
""" Have a function create a DataFrame from a file """
import pandas as pd


def from_file(filename, delimiter):
    """ Create a DataFrame from a file
        - filename ....... file to load from
        - delimiter ...... column separator
    """
    return pd.read_csv(filename, delimiter=delimiter)
