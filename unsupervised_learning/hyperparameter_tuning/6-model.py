""" Defines a simple model to be trained on IRIS dataset and optimized
    with Bayessian Optimization

Model Architecture:
    1. Dense Layer, input shape (28, 28)

Requires:
    - tensorflow
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2


def build_simple_model(learning_rate, layers, dropout_rate, batch_size, l2_reg, ishape):
    """ Builds a simple model
        - learning_rate: learning rate for the model
        - layers: number of nodes for each layer
        - dropout_rate: dropout rate
        - batch_size: number of samples to be processed during training
        - l2_reg: L2 regularization parameter
        - ishape: input shape
    """
    model = Sequential([])

    # Add the first layer with L2 regularization
    L0 = Dense(units=int(layers[0]), activation="relu", input_shape=(ishape,), kernel_regularizer=l2(l2_reg))
    model.add(L0)

    # Add subsequent layers
    for layer_idx, nodes in enumerate(layers[1:]):
        model.add(Dense(units=nodes, activation="relu", kernel_regularizer=l2(l2_reg)))

    # Add the output layer
    model.add(Dense(units=3, activation="softmax"))

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
