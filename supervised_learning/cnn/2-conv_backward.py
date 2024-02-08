#!/usr/bin/env python3
""" This module contains conv_backward(),
    that performs back propagation over a convolutional layer of a nn:

    requires:
        - numpy.
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Performs back propagation over a convolutional layer of a nn:
        - dZ: np.ndarray of shape (m, h_new, w_new, c_new),
              contains the partial derivatives with respect to the
              unactivated output of the convolutional layer.
            - m: number of examples,
            - h_new: height of the output,
            - w_new: width of the output,
            - c_new: output number of channels.
        - A_prev: np.ndarray of shape (m, h_prev, w_prev, c_prev),
                  contains the output of the previous layer.
            - h_prev: height of the previous layer,
            - w_prev: width of the previous layer,
            - c_prev: previous layer number of channels
        - W: np.ndarray of shape (kh, kw, c_prev, c_new),
             contains the kernels for the convolution.
            - kh: kernel height,
            - kw: kernel width.
        - b: np.ndarray of shape (1, 1, 1, c_new),
             contains the biases applied to the convolution.
        - padding: string, either "same" or "valid",
                   indicating the type of padding used.
        - stride: tuple of (sh, sw), contains the strides for the convolution.
    """
    # Extracting dimensions
    m, oh, ow, och = dZ.shape
    m, prvh, prvw, prvc = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Initializing gradients
    prvda = np.zeros_like(A_prev)
    dw = np.zeros_like(W)
    db = np.zeros_like(b)

    # Calculating padding width and height
    if padding == "same":
        padh = ((prvh - 1) * sh - prvh + kh) // 2
        padw = ((prvw - 1) * sw - prvw + kw) // 2
    elif padding == "valid":
        padh = padw = 0

    # Padding A_prev
    prvapad = np.pad(A_prev, ((0, 0), (padh, padh), (padw, padw), (0, 0)),
                     mode='constant')

    # Backpropagation
    for i in range(m):
        for h in range(oh):
            for w in range(ow):
                for k in range(och):
                    # Appling stride
                    x = h*sh
                    y = w*sh

                    # Calculating gradients
                    prvda[i, x: x+kh, y: y+kw, :] += W[:, :, :, k] *\
                        dZ[i, h, w, k]
                    dw[:, :, :, k] += prvapad[i, x: x+kh, y: y+kw, :] *\
                        dZ[i, h, w, k]
                    db[:, :, :, k] += dZ[i, h, w, k]

    return prvda, dw, db
