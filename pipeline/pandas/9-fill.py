#!/usr/bin/env python3
""" Have a function handle missing values in a DataFrame """


def fill(df):
    """ Handles missing values in a DataFrame """
    # Remove Weighted_Price column
    df.drop(columns=['Weighted_Price'], inplace=True)

    # Use the previous value of Close to fill the next missing one
    df['Close'] = df['Close'].ffill()

    # Fill missing values in High, Low, and Open columns with the corresponding Close value in the same row # noqa
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])

    # Set missing values in Volume_(BTC) and Volume_(Currency) to 0
    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col] = df[col].fillna(0)

    return df
