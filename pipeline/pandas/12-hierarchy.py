#!/usr/bin/env python3
""" Have a function concatenate two DataFrames """

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """ Rearranges the MultiIndex of concatenated DataFrames and filters rows based on a timestamp range """
    # Filter both DataFrames for the specified timestamp range
    df1 = df1.loc[(df1['Timestamp'] >= 1417411980) & (df1['Timestamp'] <= 1417417980)]
    df2 = df2.loc[(df2['Timestamp'] >= 1417411980) & (df2['Timestamp'] <= 1417417980)]

    # Ensure both DataFrames are indexed on their 'Timestamp' columns
    dff1, dff2 = index(df1), index(df2)

    # Concatenate the filtered DataFrames with hierarchical keys
    concatenated = pd.concat([dff2, dff1], keys=['bitstamp', 'coinbase'])

    # Swap the MultiIndex levels to make Timestamp the first level
    # Sort by Timestamp to ensure chronological order
    return concatenated.swaplevel(0, 1).sort_index(level=0)
