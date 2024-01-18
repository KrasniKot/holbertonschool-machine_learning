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

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for epoch in range(epochs + 1):
            feed_dict = {x: X_train, y: Y_train}
            t_accur = session.run(accuracy, feed_dict)
            t_loss = session.run(loss, feed_dict)
            vaccur = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
            v_loss = session.run(loss, feed_dict={x: X_valid, y: Y_valid})

            print('After {} epochs:'.format(epoch))
            print('\tTraining Cost: {}'.format(t_loss))
            print('\tTraining Accuracy: {}'.format(t_accur))
            print('\tValidation Cost: {}'.format(v_loss))
            print('\tValidation Accuracy: {}'.format(vaccur))

            if epoch != epochs:
                start = 0
                end = batch_size

                X_trainS, Y_trainS = shuffle_data(X_train, Y_train)

                for i in range(1, round(len(X_train) / batch_size) + 2):
                    train_batch = X_trainS[start:end]
                    train_label = Y_trainS[start:end]
                    feed_dict = {x: train_batch, y: train_label}
                    b_train = session.run(train_op, feed_dict)

                    if i % 100 == 0:
                        b_cost = session.run(loss, feed_dict)
                        b_accuracy = session.run(accuracy, feed_dict)
                        print('\tStep {}:'.format(i))
                        print('\t\tCost: {}'.format(b_cost))
                        print('\t\tAccuracy: {}'.format(b_accuracy))

                    start = start + batch_size
                    if (m - start) < batch_size:
                        end = end + (m - start)
                    else:
                        end = end + batch_size

        return saver.save(session, save_path)
