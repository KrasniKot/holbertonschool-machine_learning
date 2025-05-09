#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
#import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

crop_image = __import__('1-crop').crop_image

tf.random.set_seed(1)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
for image, _ in doggies.shuffle(10).take(1):
    plt.imshow(crop_image(image, (200, 200, 3)))
    plt.show()
