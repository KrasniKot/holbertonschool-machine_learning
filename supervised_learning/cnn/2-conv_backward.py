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
    m, oh, ow, och = dZ.shape
    m, prvh, prvw, prvc = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == "valid":
        ph = pw = 0
    else:
        ph = (((prvh - 1) * sh) + kh - prvh) // 2 + 1
        pw = (((prvw - 1) * sw) + kw - prvw) // 2 + 1

    pdd = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                 "constant", constant_values=0)

    dA_prev = np.zeros((m, prvh + (2 * ph), prvw + (2 * pw), prvc))
    dw = np.zeros((kh, kw, prvc, och))

    for i in range(m):
        for c in range(och):
            for h in range(oh):
                for w in range(ow):
                    x = h * sh
                    y = w * sw

                    dA_prev[i, x: x + kh, y: y + kw, :] += (
                        dZ[i, h, w, c] * W[:, :, :, c])
                    dw[:, :, :, c] += (
                        pdd[i, x: x + kh, y: y + kw, :] *
                        dZ[i, h, w, c])

    if padding == "same":
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    return dA_prev, dw, db
