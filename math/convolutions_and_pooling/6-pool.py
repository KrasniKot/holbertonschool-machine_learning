#!/usr/bin/env python3
""" This module contains convolve_channels,
    that performs pooling on an image

    requires:
        - numpy.
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ Perfoms pooling over an img with channels
        - images: numpy.ndarray with shape (m, h, w) containing
            multiple grayscale images
            - m: the number of images
            - h: height in pixels of the images
            - w: width in pixels of the images
            - c: number the channels in the image
        - kernel_shape: tuple of (kh, kw) containing
            the kernel shape of the pooling
            - kh: the height of the kernel
            -kw: the width of the kernel
        - stride: is a tuple of (sh, sw)
            -sh is the stride for the height of the image
            - sw is the stride for the width of the image
        - mode: indicates the type of pooling
            - max: max pooling
            - avg: average pooling
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    oh = int(1 + ((h - kh) / sh))
    ow = int(1 + ((w - kw) / sw))

    co = np.zeros((m, oh, ow, c))

    image = np.arange(m)

    for x in range(oh):
        for y in range(ow):
            if mode == 'max':
                co[image, x, y] = (np.max(images[image,
                                                 x * sh:((x * sh) + kh),
                                                 y * sw:((y * sw) + kw)],
                                          axis=(1, 2)))

            elif mode == 'avg':
                co[image, x, y] = (np.mean(images[image,
                                                  x * sh:((x * sh) + kh),
                                                  y * sw:((y * sw) + kw)],
                                           axis=(1, 2)))

    return co
