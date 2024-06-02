#!/usr/bin/env python3
""" This module contains convolve_channels,
    that performs a convolution on an image with channels and multiple kernels

    requires:
        - numpy.
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ Performs a convolution of img with channels and kernels
        - images: numpy.ndarray with shape (m, h, w) containing
            multiple grayscale images
            - m: the number of images
            - h: height in pixels of the images
            - w: width in pixels of the images
        - kernels: numpy.ndarray with shape (kh, kw, c, nc) containing
            the kernel for the convolution
            - kn: the height of the kernel
            - kw: the width of the kernel
            - c:  the number of channels in the image
            - nc: number of kernels
        - padding: is either a tuple of shape (ph, pw), 'same, 'valid'
            - ph: is the padding for the height of the image
            - pw: is the padding for the width of the image
        - stride: is a tuple of shape (sh, sw)
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nc = kernels.shape[3]

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

    co = np.zeros((m, oh, ow, nc))

    image = np.arange(m)

    for x in range(oh):
        for y in range(ow):
            for z in range(nc):
                co[image, x, y, z] = (np.sum(padimg[image,
                                                    x * sh:((x * sh) + kh),
                                                    y * sw:((y * sw) + kw)] *
                                             kernels[:, :, :, z],
                                             axis=(1, 2, 3)))
    return co
