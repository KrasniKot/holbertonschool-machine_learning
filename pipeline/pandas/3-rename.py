#!/usr/bin/env python3
""" Have a function rename a column """

import pandas as pd


def rename(df):
    """ Renames a column in the df """
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    return df[['Datetime', 'Close']]
