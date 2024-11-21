#!/usr/bin/env python3
""" Have a function prune a DataFrame """


def prune(df):
    """ Prunes a DataFrame df """
    return df.dropna(subset=['Close'])