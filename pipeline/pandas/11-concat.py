#!/usr/bin/env python3
""" Have a function concatenate two DataFrames """

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """ Concatenates two DataFrame objects
    - df1 .... DataFrame to append to (coinbase).
    - df2 .... DataFrame to select rows from and prepend (bitstamp).
    """
    # Ensure both DataFrames are indexed on their 'Timestamp' columns
    df1, df2 = index(df1), index(df2)

    # Filter df2 for rows with timestamps up to and including 1417411920
    # Concatenate the filtered rows from df2 and df1, adding keys
    return pd.concat([df2.loc[:1417411920], df1], keys=['bitstamp', 'coinbase'])  # noqa
