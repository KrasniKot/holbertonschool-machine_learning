#!/usr/bin/env python3
""" Have a function set the Timestamp as the index for a DataFrame """


def index(df):
    """ Sets the Timestamp as the index for a DataFrame """
    return df.set_index('Timestamp')
