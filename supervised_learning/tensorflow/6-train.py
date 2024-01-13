#!/usr/bin/env python3
""" This module trains and saves the nn classifier """

import tensorflow.compat.v1 as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """ Builds, trains, and saves a neural network classifier
        - X_train: numpy.ndarray containing the training input data,
        - Y_train: numpy.ndarray containing the training labels,
        - X_valid: numpy.ndarray containing the validation input data,
        - Y_valid: numpy.ndarray containing the validation labels,
        - layer_sizes: list containing the number of nodes in each layer,
        - activations: list containing the activation functions for each layer,
        - alpha: learning rate,
        - iterations: number of iterations to train over,
        - save_path: designates where to save the model,
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    a = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', a)
    acc = calculate_accuracy(y, a)
    tf.add_to_collection('accuracy', acc)
    loss = calculate_loss(y, a)
    tf.add_to_collection('loss', loss)
    top = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', top)

    init = tf.global_variables_initializer()
    save = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            ct, acct = sess.run(
                    [loss, acc], feed_dict={x: X_train, y: Y_train})
            ctv, accv = sess.run(
                    [loss, acc], feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {ct}")
                print(f"\tTraining Accuracy: {acct}")
                print(f"\tValidation Cost: {ctv}")
                print(f"\tValidation Accuracy: {accv}")

            if i < iterations:
                sess.run(top, feed_dict={x: X_train, y: Y_train})

        mdlpath = save.save(sess, save_path)

    return mdlpath
