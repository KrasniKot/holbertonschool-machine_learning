#!/usr/bin/env python3
""" Have a function provide statistics about a DataFrame """


def analyze(df):
    """ Computes descriptive statistics for all columns except the Timestamp column """
    # Exclude the 'Timestamp' column and compute statistics for the remaining columns
    # Compute descriptive statistics
    return df.drop(columns=['Timestamp']).describe()
