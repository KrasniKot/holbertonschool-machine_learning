#!/usr/bin/env python3
""" Have a function rename a column """

import pandas as pd


def rename(df):
    """ Renames a column in the df """
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)  # Rename column Timestamp to Datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')   # Convert timestamp values to datatime values

    return df[['Datetime', 'Close']]                            # Stick only with the Datetime and Close columns
