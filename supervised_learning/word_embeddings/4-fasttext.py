#!/usr/bin/env python3
""" Function that creates and trains a genism fastText model """
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """ Creates and trains a genism fastText model and returns it
        - sentences: list of sentences to be trained on
        - size: dimensionality of the embedding layer
        - min_count: minimum number of occurrences of a word for use
                     in training
        - window: maximum distance between the current and predicted
                  word within a sentence
        - negative: size of negative sampling
        - cbow: boolean to determine the training type; True is for CBOW
                False is for Skip-gram
        - iterations: number of iterations to train over
        - seed: seed for the random number generator
        - workers: number of worker threads to train the model
    """
    model = FastText(sentences=sentences,
                     sg=cbow,
                     vector_size=size,
                     negative=negative,
                     window=window,
                     min_count=min_count,
                     seed=seed,
                     workers=workers,
                     epochs=iterations)
    return model
