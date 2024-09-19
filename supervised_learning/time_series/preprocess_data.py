#!/usr/bin/env python3
""" This module contains a set of fuctions to preprocess the data """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_dataset(df, column_name, title):
    """ Plots the specified column of the dataset
        - df: DataFrame containing the dataset
        - column_name: name of the column to plot
        - title: plot title
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")

    plt.figure(figsize=(12, 6))

    # Plot the data with the index as the x-axis
    plt.plot(df.index, df[column_name], label=column_name)

    plt.xlabel('Start Time')
    plt.ylabel(column_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def load_and_normalize(datasets):
    """ Loads the given datasets normalizes them,
        returns a single combined dataset
        - datasets: list of datasets to load and normalize
    """
    processed_datasets = []

    primaryds = None
    for dataset in datasets:
        print(f"Loading and normalizing dataset:", dataset)

        dataset = pd.read_csv(dataset)

        # Time stamps are converted to seconds, and is set as index
        dataset = dataset.set_index(
            pd.to_datetime(dataset['Timestamp'], unit='s')
            )
        dataset = dataset.drop('Timestamp', axis=1)

        # Databases are combined
        if primaryds is None:
            primaryds = dataset
        else:
            primaryds.combine_first(dataset)

    # Dates previous to 2017, 1, 1 are excluded
    primaryds = primaryds[primaryds.index >= pd.Timestamp(2017, 1, 1)]

    # Missing column values fixed: continuous timeseries
    primaryds.ffill(inplace=True)

    # Aggregate all the records to 1 per hour
    primaryds = primaryds.resample('h').mean()

    print()
    return primaryds


def drop_by_correlation(dataset, threshold, do_not_drop):
    """ Decides which columns are useful based on
        the correlation with the close price
        - dataset: dataset to evaluate
        - threshold: minimum accepted correlation rate
        - do_not_drop: columns that must not be dropped
    """
    # Correlation matrix with respect to the close column
    correlations = dataset.corr()['Close']

    print(correlations)

    # Features with correlation values less than the threshold
    features_below_threshold = correlations[
        abs(correlations) < threshold
        ].index.tolist()

    print("\nFeatures below threshold:", features_below_threshold)
    print("Dropping:", features_below_threshold, "from dataset")

    # Drop all the columns that are below the threshold
    #    but those found in do_not_drop
    dataset.drop(columns=[
        col for col in features_below_threshold if col not in do_not_drop
        ], inplace=True)

    print()

    return dataset


if __name__ == "__main__":
    # Datasets are loaded, normalized and combined
    dataset = load_and_normalize(["bitstampUSD.csv", "coinbaseUSD.csv"])

    # Close price for combined dataset is plotted
    print(dataset)
    plot_dataset(dataset, "Close", "Close Price Over Time")

    # Columns with lower correlation with close price are dropped
    dataset = drop_by_correlation(dataset, 0.25, ["start_time"])
    dataset.drop(columns=["Weighted_Price"])

    # Differential logarithm to make the data stationarity
    #    log(current_value / previous_value)
    print("Making dataset stationary")
    dataset = dataset.apply(lambda x: np.log(x / x.shift(1)), axis=0)
    dataset = dataset.dropna()

    # Clip values to be within [-0.10, 0.10]
    dataset['Close'] = dataset['Close'].clip(lower=-0.10, upper=0.10)
    print("Clipping 'Close' column")

    plot_dataset(
        dataset, "Close", "Close Price Over Time (Stationary, Clipped)")

    print("Saving preprocessed data into ./preprocessed_data.csv")

    dataset.to_csv('preprocessed_data.csv', index=False)
