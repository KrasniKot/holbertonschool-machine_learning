#!/usr/bin/env python3
""" Have a function reverse the chronological order and transpose a df """


def flip_switch(df):
    """ Reverses the chronological order and transpose a df """
    return df.sort_index(ascending=False).T
