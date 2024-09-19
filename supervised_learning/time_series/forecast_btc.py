#!/usr/bin/env python3
""" This module builds, trains and validates a LSTM model """

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense


def build_lstm_model_functional(input_shape, layers, dropouts):
    """ Builds an LSTM model with L2 regularization
        - input_shape (tuple): Shape of the input data (timesteps, features).
        - layers: list containing the number of nodes per layer
        - dropouts: list containing the dropout rate for each layer
        - l2_lambda: L2 regularization factor
    """
    # Input layer
    inputs = Input(shape=input_shape)
    x = inputs

    # Iterate over all but last element
    for i, nodes in enumerate(layers[:-1]):
        # All but the last LSTM layer should return sequences
        return_sequences = (i < len(layers) - 2)

        x = LSTM(nodes, return_sequences=return_sequences)(x)

        if i < len(dropouts):
            x = Dropout(dropouts[i])(x)  # Apply dropout

    x = Dense(layers[-1], activation='relu')(x)

    output = Dense(1)(x)  # Last layer will return the predicted close price

    # Keras model
    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer='adam', loss='mse')

    return model


def split_dataset(dataset):
    """ Splits a dataset into training, validation and testing datasets
        and returns them standardized
    """
    print("Splitting and standardizing datasets")

    n = len(dataset)
    trainds = dataset[:int(n * 0.80)]
    valds = dataset[int(n * 0.80): int(n * 0.9)]
    testds = dataset[int(n * 0.9):]

    return trainds, valds, testds


def standardize(X, mean, std):
    """ Standardize the data using mean and std """
    return (X - mean) / std


def create_sequences(data, input_width, label_width, shift):
    X, y = [], []

    for i in range(len(data) - input_width - shift):
        # Append the input sequence, using .iloc for positional indexing
        X.append(data.iloc[i:i + input_width].values)

        # Append the label (close price for the next time step)
        y.append(data.iloc[i + input_width + shift - 1]['Close'])

    #                   Reshape y to (batch_size, 1)
    return np.array(X), np.array(y).reshape(-1, 1)


def standardize_datasets(datasets, mean, std):
    """ Standardizes the provided datasets """
    print("Standardizing datasets")

    return (standardize(dataset, mean, std) for dataset in datasets)

    print("Creating sequences")

    X, y = [], []  # empty lists to store input (X) and output (y) sequences

    # Loop through the data to create sequences
    for start in range(len(data) - input_width - label_width + 1):
        end = start + input_width          # end of the input sequence
        label_end = end + label_width      # end of the label sequence

        # Append the input sequence (data[start:end]) to X
        # data[start:end] takes a window of input_width from the data
        X.append(data[start:end].values)

        # Append the label sequence (data[end:label_end]) to y
        # data[end:label_end] is the corresponding label sequence
        # after the input
        y.append(data[end:label_end].values)

    # Convert the list of sequences into NumPy arrays for training
    return np.array(X), np.array(y)


def create_tf_dataset(X, y, batch_size=64):
    """ Create a tf.data.Dataset from the sequences and targets """
    print("Creating datasets")

    X = X.astype('float32')
    y = y.astype('float32')

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def plot_datasets(datasets, labels, colors):
    """ Plots the given datasets Close column """
    plt.figure(figsize=(10, 6))

    for dataset, label, color in zip(datasets, labels, colors):
        start_time = dataset.index  # Extract the start_time column
        close_price = dataset['Close']  # Extract the close price column
        plt.scatter(
            start_time, close_price, label=label, color=color, alpha=0.7)

    plt.title(
        'Stationary Close Price for Training, Validation, and Test Datasets')
    plt.legend()
    plt.show()


def plot_loss(history):
    """Plot training and validation loss"""

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(
        history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Load and preprocess data
    dataset = pd.read_csv("preprocessed_data.csv")
    print(dataset.head(10))

    # Split dataset
    trainds, valds, testds = split_dataset(dataset)

    datasets = [trainds, valds, testds]
    labels = ['Train', 'Validation', 'Test']
    colors = ['blue', 'green', 'red']

    # Plot the datasets
    plot_datasets(datasets, labels, colors)

    # Standardize all datasets
    tmean, tstd = trainds.mean(axis=0), trainds.std(axis=0)
    trainds, valds, testds = standardize_datasets(
        [trainds, valds, testds], tmean, tstd)

    # Parameters for sequence creation
    input_width = 24  # Number of time steps in input sequences
    label_width = 24   # Number of time steps in label sequences
    shift = 1         # Shift to create the next sample

    # Create sequences
    X_train, y_train = create_sequences(
        trainds, input_width, label_width, shift)
    X_val, y_val = create_sequences(valds, input_width, label_width, shift)
    X_test, y_test = create_sequences(testds, input_width, label_width, shift)

    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(X_train, y_train)
    val_dataset = create_tf_dataset(X_val, y_val)
    test_dataset = create_tf_dataset(X_test, y_test)

    # Build model
    input_shape = (input_width, dataset.shape[1])  # (timesteps, features)
    model = build_lstm_model_functional(
        input_shape, layers=[8, 8, 32], dropouts=[0.2, 0.1])

    # Train model
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=50)

    # Plot the training and validation loss
    plot_loss(history)

    # Saving the model
    model.save('lstm_model.keras')

    # Evaluate model
    test_loss = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
