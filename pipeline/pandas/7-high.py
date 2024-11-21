#!/usr/bin/env python3
""" Sort by High price in descending order """


def high(df):
    """ Sorts a DataFrame df by High price in descending order """
    return df.sort_values(by='High', ascending=False)
