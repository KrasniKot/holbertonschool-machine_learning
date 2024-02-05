#!/usr/bin/env python3
""" This module contains convolve_grayscale()
    that performs a convolution on grayscale images:

    requires:
        - numpy.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ Performs a convolution on grayscale image with with a custom stride:
        - images: np.ndarray with shape (m, h, w) containing grayscale images,
            - m: number of images;
            - h: height in pixels of the images;
            - w: width in pixels of the images;
        - kernel: np.ndarray with shape (kh, kw) containing the kernel to use,
            - kh height of the kernel;
            - kw: width of the kernel.
        - padding: either a tuple of (ph, pw), 'same', or 'valid',
        - stride: tuple of (sh, sw);
            - sh: stride for the height of the image;
            - sw: stride for the width of the image.
    """
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    if padding == "same":
        padh = ((ih - 1) * stride[0] - ih + kh) / 2
        padw = ((iw - 1) * stride[1] - iw + kw) / 2
    elif padding == "valid":
        padh = padw = 0
    else:
        padh, padw = stride

    oh = ((h + (2 * padh) - kh) / sh) + 1
    ow = ((w + (2 * padw) - kw) / sw) + 1

    co = np.zeros([m, oh, ow])
    imgpd = np.pad(images, pad_width=((0, 0), (padw, padw), (padh, padh)),
                   mode='constant', constant_values=0)

    for i in range(oh):
        for j in range(ow):
            img = imgpd[:, x: i * sh + kh, y: j * sw + kw]
            co[:, i, j] = np.multiply(img, kernel).sum(axis=(1, 2))

    return co
