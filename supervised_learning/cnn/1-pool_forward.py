#!/usr/bin/env python3
""" This module contains pool_forward(),
    that performs forward propagation over a pooling layer of a nn:

    requires:
        - numpy.
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Performs forward propagation over a pooling layer of a nn:
        - A_prev: np.ndarray of shape (m, h_prev, w_prev, c_prev),
                  contains the previous layer output.
            - m: number of examples,
            - h_prev: height of the previous layer,
            - w_prev: width of the previous layer,
            - c_prev: number of channels of the previous layer.
        - kernel_shape: tuple of (kh, kw),
                        contains the kernel size for the pooling.
            - kh: kernel height,
            - kw: kernel width.
        - stride: tuple of (sh, sw), contains the strides for the pooling.
            - sh: stride height,
            - sw: stride width.
        - mode: string, either "max" or "avg",
                indicates whether to perform maximum or average pooling.
    """
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Determining pooling mode
    p = np.max if mode == "max" else np.average

    # Getting output dimensions
    oh = int((h - kh) / sh) + 1
    ow = int((w - kw) / sw) + 1

    # Setting output volume dimensions
    Z = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            # Getting kernel position
            x = i * sh
            y = j * sw

            # Applying pertinent pooling
            Z[:, i, j, :] = p(A_prev[:, x:x+kh, y:y+kw, :], axis=(1, 2))

    return Z
