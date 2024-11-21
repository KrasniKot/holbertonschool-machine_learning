#!/usr/bin/env python3
""" Convert the tail of a DataFrame to numpy """

import pandas as pd


def array(df):
    """ Convert the tail of a DataFrame df to numpy """
    return df.loc[df.index[-10:], ['High', 'Close']].to_numpy()