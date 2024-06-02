#!/usr/bin/env python3
""" This module contains convolve_channels,
    that performs a convolution on an image with channels

    requires:
        - numpy.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """ Performs a convolution on an image with channels
        - images: numpy.ndarray with shape (m, h, w) containing
            multiple grayscale images
            - m: the number of images
            - h: height in pixels of the images
            - w: width in pixels of the images
        - kernel: numpy.ndarray with shape (kh, kw, c) containing
            the kernel for the convolution
            - kn: the height of the kernel
            - kw: the width of the kernel
            - c:  the number of channels in the image
        - padding: is either a tuple with shape (ph, pw), 'same, 'valid'
            - ph: is the padding for the height of the image
            - pw: is the padding for the width of the image
        - stride: is a tuple with shape (sh, sw)
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    padw = 0
    padh = 0

    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        padh = int((((h - 1) * sh + kh - h) / 2) + 1)
        padw = int((((w - 1) * sw + kw - w) / 2) + 1)
    if type(padding) is tuple:
        padh = padding[0]
        padw = padding[1]

    padimg = np.pad(images, pad_width=((0, 0), (padh, padh),
                                       (padw, padw), (0, 0)),
                       mode='constant')

    oh = int(((h + 2 * padh - kh) / sh) + 1)
    ow = int(((w + 2 * padw - kw) / sw) + 1)

    co = np.zeros((m, oh, ow))

    image = np.arange(m)

    for x in range(oh):
        for y in range(ow):
            co[image, x, y] = (np.sum(padimg[image,
                                      x * sh:((x * sh) + kh),
                                      y * sw:((y * sw) + kw)] * kernel,
                                      axis=(1, 2, 3)))
    return co
