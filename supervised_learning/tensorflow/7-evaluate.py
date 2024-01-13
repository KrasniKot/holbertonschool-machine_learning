#!/usr/bin/env python3
""" This module contains evaluate()
which returns A, the accuarcy and the cost
"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a nn
     - X : numpy.ndarray containing the input data to evaluate
     - Y : numpy.ndarray containing the one-hot labels for X,
     - save_path: location to load the model from
    """
    with tf.Session() as sess:
        save = tf.train.import_meta_graph(save_path + ".meta")
        save.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        a = tf.get_collection("y_pred")[0]
        acc = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        A = sess.run(a, feed_dict={x: X, y: Y})
        cost = sess.run(loss, feed_dict={x: X, y: Y})
        acc = sess.run(acc, feed_dict={x: X, y: Y})

        return A, acc, cost
