#!/usr/bin/env python3
""" This module contains conv_forward(),
    that performs forward propagation over a convolutional layer of a nn:

    requires:
        - numpy.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ Performs forward prop over a nn convolutional layer:
        - A_prev: np.ndarray of shape (m, h_prev, w_prev, c_prev),
                  contains the previous layer output.
            - m: number of images,
            - h_prev: height of the previous layer,
            - w_prev: width of the previous layer,
            - c_prev: number of chanels in the previous layer.
        - W: np.ndarray of shape (kh, kw, c_prev, c_new),
             contains the kernels for the convolution.
            - kh: kernel height,
            - kw: kernel width,
            - c_new: number of chanels in the output.
        - b: np.ndarray of shape (1, 1, 1, c_new),
             contains the biases applied to the convolution.
        - activation: activation function applied to the convolution.
        - padding: string that is either "same" or "valid",
                   indicating the type of padding used.
        - stride: tuple of (sh, sw), contains the strides for the convolution.
            - sh: stride height,
            - sw: stride width.
    """
    m, prvh, prvw, prvc = A_prev.shape
    kh, kw, _, och = W.shape
    sh, sw = stride

    if padding == 'same':
        padh = int((((prvh - 1) * sh - prvh + kh) / 2))
        padw = int((((prvw - 1) * sw - prvw + kw) / 2))
    else:
        padh = padw = 0

    # output dimensions
    oh = int(((prvh + (2 * padh) - kh) / sh)) + 1
    ow = int(((prvw + (2 * padw) - kw) / sw)) + 1

    # Initialize output volume and pad the previous layer output
    Z = np.zeros([m, oh, ow, och])
    Apad = np.pad(A_prev,
                  pad_width=((0, 0), (padh, padh), (padw, padw), (0, 0)),
                  mode='constant', constant_values=0)

    for h in range(oh):  # loop over the output vertical axis
        for w in range(ow):  # loop over the output horizontal axis
            for c in range(och):  # loop over the output channels

                # get the current kernel position
                x = h * sh
                y = w * sw

                # get the current output slice
                aslice = Apad[:, x:x+kh, y:y+kw, :]

                # get the slice convolved
                Z[:, h, w, c] = (aslice * W[:, :, :, c]).sum(axis=(1, 2, 3))

    Z = Z + b  # add bias

    return activation(Z)  # activated output
