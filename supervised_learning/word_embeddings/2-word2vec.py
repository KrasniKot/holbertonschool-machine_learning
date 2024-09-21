#!/usr/bin/env python3
""" Create a function that trains a gensim word2vec model """

import gensim


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """ Trains a word2vec model using gensim, and returns the trained model
        - sentences: list of sentences to be trained on
        - size: dimensionality of the embedding layer
        - min_count: minimum number of occurrences of a word for use in
                     training
        - window: maximum distance between the current and predicted word
                  within a sentence
        - negative: the size of negative sampling
        - cbow: boolean to determine the training type; True is for CBOW;
                False is for Skip-gram
        - iterations:  number of iterations to train over
        - seed: seed for the random number generator
        - workers: number of worker threads to train the model
    """
    model = gensim.models.Word2Vec(sentences, vector_size=size, window=window,
                                   min_count=min_count, negative=negative,
                                   workers=workers, sg=cbow, seed=seed,
                                   epochs=iterations)
    return model
