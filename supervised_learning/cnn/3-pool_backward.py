#!/usr/bin/env python3
""" This module contains pool_backward(),
    that performs back propagation over a pooling layer of a nn.

    requires:
        - numpy.
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Performs  back propagation over a pooling layer of a nn:
        - dA:
        - A_prev:
        - kernel_shape:
        - stride:
        - mode:
    """
    m, oh, ow, och = dA.shape
    m, prvh, prvw, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros((m, prvh, prvw, och))

    for i in range(m):
        for c in range(och):
            for h in range(oh):
                for w in range(ow):
                    x = h * sh
                    y = w * sw

                    if mode == 'max':
                        pool = A_prev[i, x: x + kh, y: y + kw, c]
                        mask = np.where(pool == np.max(pool), 1, 0)
                    elif mode == 'avg':
                        mask = np.ones((kh, kw))
                        mask /= (kh * kw)

                    dA_prev[i, x: x + kh, y: y + kw, c] += (
                        mask * dA[i, h, w, c])
    return dA_prev
