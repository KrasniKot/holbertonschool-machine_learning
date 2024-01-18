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

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        acc = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        trop = tf.get_collection('train_op')[0]

        for epoch in range(epochs + 1):
            tcost, tacc = sess.run(
                    [loss, acc], feed_dict={x: X_train, y: Y_train})
            vcost, vacc = sess.run(
                    [loss, acc], feed_dict={x: X_valid, y: Y_valid})

            print(f'After {epochs} epochs:')
            print(f'\tTraining Cost: {tcost}')
            print(f'\tTraining Accuracy: {tacc}')
            print(f'\tValidation Cost: {vcost}')
            print(f'\tValidation Accuracy: {vacc}')

            Xt, Yt = shuffle_data(X_train, Y_train)

            sts = 0
            ends = batch_size

            Xt, Yt = shuffle_data(X_train, Y_train)

            if m % batch_size == 0:
                bpe = m // batch_size
            else:
                bpe = (m // batch_size) + 1

            for i in range(bpe):
                feed_dict = {x: Xt[sts:ends], y: Yt[sts:ends]}
                sess.run(trop, feed_dict)

                if step % 100 == 0:
                    bcost = session.run(loss, feed_dict)
                    bacc = session.run(acc, feed_dict)
                    print(f'\tStep {i}:')
                    print(f'\t\tCost: {bcost}')
                    print(f'\t\tAccuracy: {bacc}')

                sts = sts + batch_size
                if (m - sts) < batch_size:
                    ends = ends + (m - sts)
                else:
                    ends = ends + batch_size

        return (saver.save(sess, save_path))
