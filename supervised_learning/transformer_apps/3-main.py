#!/usr/bin/env python3

Dataset = __import__('3-dataset').Dataset
import tensorflow as tf

tf.random.set_seed(0)
data = Dataset(32, 40)

print("TRAIN\n\n")
for pt, en in data.data_train.take(1):
    print(pt, en)

print("VALID\n\n")
for pt, en in data.data_valid.take(1):
    print(pt, en)