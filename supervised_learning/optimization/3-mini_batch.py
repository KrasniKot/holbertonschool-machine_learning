#!/usr/bin/env python3
""" This module contains train_mini_batch(),
    which trains a nn using mini_batch gradient descent.

    requires:
        - numpy,
        - tensoflow.
"""

import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ Trains a nn using mini-bacth gradient descent,
        - X_train: np.ndarray of shape (m, 784) containing the training data,
        - Y_train one-hot np.ndarray of shape (m, 10)
            containing the training labels,
        - X_valid: np.ndarray of shape (m, 784) containing the validation data,
        - Y_valid: one-hot np.ndarray of shape (m, 10)
            containing the validation labels,
        - batch_size: number of data points in a batch
        - epochs: number of times the training should pass throughout dataset,
        - load_path: path from which to load the model,
        - save_path path where the model should be saved after training to,
    """
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(session, load_path)
        
        # Access placeholders and ops
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # Number of steps per epoch
        steps_per_epoch = X_train.shape[0] // batch_size

        for epoch in range(epochs):

            X_train, Y_train = shuffle_data(X_train, Y_train)

            for step in range(steps_per_epoch):
                # Get mini-batch
                start = step * batch_size
                end = start + batch_size
                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                # Train the model
                feed_dict = {x: X_batch, y: Y_batch}
                session.run(train_op, feed_dict)

            # Print statistics
            train_accuracy = session.run(accuracy, feed_dict={x: X_train, y: Y_train})
            train_loss = session.run(loss, feed_dict={x: X_train, y: Y_train})
            valid_accuracy = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            valid_loss = session.run(loss, feed_dict={x: X_valid, y: Y_valid})

            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Training - Loss: {train_loss}, Accuracy: {train_accuracy}')
            print(f'Validation - Loss: {valid_loss}, Accuracy: {valid_accuracy}')

        # Save the model
        saver.save(session, save_path)

    return save_path
