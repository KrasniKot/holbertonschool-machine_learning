#!/usr/bin/env python3
""" Select every 60 rows """


def slice(df):
    """ Slice a DataFrame df """
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[:: 60]
