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
    m = X_train.shape[0]

    with tf.Session() as session:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(session, load_path)

        x, y = tf.get_collection('x')[0], tf.get_collection('y')[0]
        acc = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        trop = tf.get_collection('train_op')[0]

        for epoch in range(epochs + 1):
            feed_dict = {x: X_train, y: Y_train}
            tacc, toss = session.run([acc, loss], feed_dict)
            vacc, voss = session.run(
                    [acc, loss], feed_dict={x: X_valid, y: Y_valid})

            print(f'After {epoch} epochs:')
            print(f'\tTraining Cost: {toss}')
            print(f'\tTraining Accuracy: {tacc}')
            print(f'\tValidation Cost: {voss}')
            print(f'\tValidation Accuracy: {vacc}')

            if epoch != epochs:
                start = 0
                end = batch_size

                Xt, Yt = shuffle_data(X_train, Y_train)

                for i in range(1, round(len(X_train) / batch_size) + 2):
                    feed_dict = {x: Xt[start:end], y: Yt[start:end]}
                    b_train = session.run(trop, feed_dict)

                    if i % 100 == 0:
                        bcost, bacc = session.run([loss, acc], feed_dict)

                        print(f'\tStep {i}:')
                        print(f'\t\tCost: {bcost}')
                        print(f'\t\tAccuracy: {bacc}')

                    start = start + batch_size
                    if (m - start) < batch_size:
                        end = end + (m - start)
                    else:
                        end = end + batch_size

        return saver.save(session, save_path)
