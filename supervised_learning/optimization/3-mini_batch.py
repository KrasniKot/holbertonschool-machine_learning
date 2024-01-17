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
    with tf.Session() as sess:
        # Get model and restore session
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        # Get needed tensors
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = Y_train.shape[0]  # Corrected line
        batches_per_epoch = range(0, m, batch_size)

        for epoch in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)

            train_cost, train_accuracy = sess.run(
                    [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            print(f'After {epoch + 1} epochs:')
            print(f'\tTraining Cost: {train_cost}')
            print(f'\tTraining Accuracy: {train_accuracy}')

            # Validation
            valid_cost, valid_accuracy = sess.run(
                    [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print(f'\tValidation Cost: {valid_cost}')
            print(f'\tValidation Accuracy: {valid_accuracy}')

            for step in batches_per_epoch:
                # Get mini-batch
                X_batch = X_train[step: step + batch_size]
                Y_batch = Y_train[step: step + batch_size]

                _, step_cost, step_accuracy = sess.run(
                        [train_op, loss, accuracy], feed_dict={
                            x: X_batch, y: Y_batch})

                if step % 100 == 0:
                    print(f'\tStep {step}:')
                    print(f'\t\tCost: {step_cost}')
                    print(f'\t\tAccuracy: {step_accuracy}')

        saver.save(sess, save_path)

    return save_path
